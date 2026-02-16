#!/usr/bin/env python3
"""
Command Gateway Node — Thin Orchestrator.

Wires together the modular subsystems:
  • CuroboPlanner       — GPU-accelerated motion planning
  • TrajectoryExecutor  — RTDE / ROS2 trajectory dispatch
  • JointStateManager   — mode-aware /joint_states publishing
  • ProtocolExecutor    — multi-pose protocol execution
  • RPCHandlers         — WebSocket RPC implementations
"""

import asyncio
import json
import time
import traceback
from typing import Dict

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor

from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState

import tf2_ros
from tf2_ros import Buffer, TransformListener

try:
    import websockets
    from websockets.server import serve
except ImportError:
    raise ImportError("websockets>=12.0 required: pip install websockets")

# ── Package imports ──────────────────────────────────────────────
from simforge_gateway_nvidia.config import (
    ROBOT_CONFIG, RTDE_AVAILABLE, CUROBO_AVAILABLE,
    ConnectedClient, RobotStateInfo, CONFIG_DIR, ENV_NAME,
)
from simforge_gateway_nvidia.rtde_controller import URRTDEController
from simforge_gateway_nvidia.joint_state_manager import JointStateManager
from simforge_gateway_nvidia.curobo_planner import CuroboPlanner
from simforge_gateway_nvidia.trajectory_executor import TrajectoryExecutor
from simforge_gateway_nvidia.protocol_executor import ProtocolExecutor
from simforge_gateway_nvidia.rpc_handlers import RPCHandlers


class CommandGatewayNode(Node):
    """WebSocket <-> ROS2 bridge with NVIDIA cuRobo GPU motion planning."""

    def __init__(self):
        super().__init__("command_gateway")

        # ── Parameters ───────────────────────────────────────────
        self.declare_parameter("websocket_port", 8766)
        self.declare_parameter("websocket_host", "0.0.0.0")
        self.declare_parameter("max_clients", 5)
        self.declare_parameter("max_velocity_scaling", 1.0)
        self.declare_parameter("max_acceleration_scaling", 1.0)
        self.declare_parameter("config_dir", str(CONFIG_DIR))
        self.declare_parameter("interpolation_dt", 0.02)

        self.ws_port = self.get_parameter("websocket_port").value
        self.ws_host = self.get_parameter("websocket_host").value
        self.max_clients = self.get_parameter("max_clients").value
        self.max_velocity_scaling = self.get_parameter("max_velocity_scaling").value
        self.max_acceleration_scaling = self.get_parameter("max_acceleration_scaling").value
        self.interpolation_dt = self.get_parameter("interpolation_dt").value

        from pathlib import Path
        self.config_dir = Path(self.get_parameter("config_dir").value)
        self.cb_group = ReentrantCallbackGroup()

        # ── Internal state ───────────────────────────────────────
        self._ws_clients: Dict[str, ConnectedClient] = {}
        self._current_mode = "simulation"
        self._robot_program_running: Dict[str, bool] = {
            n: False for n in ROBOT_CONFIG
        }

        # ── Publishers ───────────────────────────────────────────
        self.heartbeat_pub = self.create_publisher(
            String, "/safety/heartbeat", 10
        )
        self.estop_pub = self.create_publisher(
            String, "/safety/emergency_stop", 10
        )

        # ── TF2 ──────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subsystem: Joint State Manager ───────────────────────
        self._js_mgr = JointStateManager(self)

        self.create_subscription(
            JointState, "/joint_states",
            self._js_mgr.on_joint_states, 10,
        )

        # ── Subsystem: RTDE controllers ──────────────────────────
        self._rtde_controllers: Dict[str, URRTDEController] = {}
        if RTDE_AVAILABLE:
            for rn, cfg in ROBOT_CONFIG.items():
                self._rtde_controllers[rn] = URRTDEController(
                    robot_name=rn, ip=cfg["ip"],
                    logger=self.get_logger(),
                )
            self.get_logger().info(
                "ur_rtde available -- direct RTDE control enabled"
            )
        else:
            self.get_logger().warn(
                "ur_rtde NOT available -- real robot control will use "
                "ROS2 FollowJointTrajectory"
            )

        # ── Subsystem: Protocol Executor (created early for stop_check ref)
        # Placeholder — wired up after planner + executor are ready
        self._proto_exec = None

        # ── Subsystem: Trajectory Executor ───────────────────────
        self._executor = TrajectoryExecutor(
            self,
            rtde_controllers=self._rtde_controllers,
            stop_check_fn=lambda: (
                self._proto_exec.stop_requested
                if self._proto_exec else False
            ),
            joint_state_manager=self._js_mgr,
        )

        # ── Subsystem: cuRobo Planner ────────────────────────────
        self._planner = CuroboPlanner(
            self,
            interpolation_dt=self.interpolation_dt,
            max_velocity_scaling=self.max_velocity_scaling,
            max_acceleration_scaling=self.max_acceleration_scaling,
            config_dir=self.config_dir,
        )
        urdf = self._get_urdf_from_topic()
        self._planner.init(urdf)

        # Wire planner into executor so execute_home can use
        # cuRobo collision-aware joint planning
        self._executor.set_planner(self._planner)

        # ── Subsystem: Protocol Executor ─────────────────────────
        self._proto_exec = ProtocolExecutor(
            self, self._planner, self._executor, self._js_mgr,
        )

        # ── Subsystem: RPC Handlers ──────────────────────────────
        self._rpc = RPCHandlers(
            self, self._planner, self._executor,
            self._proto_exec, self._js_mgr,
        )

        # ── Robot-program-running subscriptions ──────────────────
        self._resend_program_clients: Dict = {}
        for rn, cfg in ROBOT_CONFIG.items():
            prefix = cfg["prefix"]
            topic = f"/{prefix}io_and_status_controller/robot_program_running"
            self.create_subscription(
                Bool, topic,
                lambda msg, rn=rn: self._on_robot_program_running(rn, msg),
                10,
            )
            srv = f"/{prefix}io_and_status_controller/resend_robot_program"
            self._resend_program_clients[rn] = self.create_client(
                Trigger, srv, callback_group=self.cb_group,
            )

        self.ws_server = None

        # ── Startup summary ──────────────────────────────────────
        self.get_logger().info(
            f"Command Gateway (cuRobo) initialised -- "
            f"WS on {self.ws_host}:{self.ws_port}"
        )
        self.get_logger().info(f"Environment: {ENV_NAME}")
        self.get_logger().info(f"Robots: {list(ROBOT_CONFIG.keys())}")
        self.get_logger().info(
            f"Motion scaling: vel={self.max_velocity_scaling}, "
            f"accel={self.max_acceleration_scaling}"
        )
        self.get_logger().info(f"cuRobo available: {CUROBO_AVAILABLE}")

    # ── URDF from /robot_description ─────────────────────────────

    def _get_urdf_from_topic(self, timeout_sec: float = 30.0):
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

        urdf_data = {"value": None}

        def _cb(msg):
            urdf_data["value"] = msg.data

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        sub = self.create_subscription(
            String, "/robot_description", _cb, qos,
        )
        self.get_logger().info(
            "Waiting for URDF from /robot_description (TRANSIENT_LOCAL)..."
        )
        start = time.time()
        while urdf_data["value"] is None and (time.time() - start) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.5)
        self.destroy_subscription(sub)

        if urdf_data["value"]:
            self.get_logger().info(
                f"Got URDF ({len(urdf_data['value'])} bytes)"
            )
        else:
            self.get_logger().error(
                f"Timed out ({timeout_sec}s) waiting for /robot_description"
            )
        return urdf_data["value"]

    # ── Robot program state ──────────────────────────────────────

    def _on_robot_program_running(self, robot_name: str, msg: Bool):
        prev = self._robot_program_running.get(robot_name, False)
        self._robot_program_running[robot_name] = msg.data
        if msg.data != prev:
            self.get_logger().info(
                f"Robot program running [{robot_name}]: {msg.data}"
            )

    # ── WebSocket server ─────────────────────────────────────────

    async def start_websocket_server(self):
        self.get_logger().info(
            f"Starting WebSocket server on {self.ws_host}:{self.ws_port}"
        )
        self.ws_server = await serve(
            self._handle_client, self.ws_host, self.ws_port,
            ping_interval=30, ping_timeout=300,
        )
        self.get_logger().info("WebSocket server started")

    async def _handle_client(self, websocket, path: str = None):
        cid = f"client_{id(websocket)}"
        if len(self._ws_clients) >= self.max_clients:
            self.get_logger().warning(f"Max clients -- rejecting {cid}")
            await websocket.close(1013, "Max clients reached")
            return

        client = ConnectedClient(
            client_id=cid, websocket=websocket,
            connected_at=time.time(), last_activity=time.time(),
        )
        self._ws_clients[cid] = client
        self.get_logger().info(f"Client connected: {cid}")

        try:
            async for message in websocket:
                await self._process_message(client, message)
        except websockets.ConnectionClosed as e:
            self.get_logger().info(f"Client disconnected: {cid} -- {e}")
        except Exception as e:
            self.get_logger().error(f"Client error: {cid} -- {e}")
        finally:
            self._ws_clients.pop(cid, None)
            if self._proto_exec and self._proto_exec.running:
                self.get_logger().info(
                    f"Client {cid} disconnected while proto_sim running -- stopping"
                )
                self._proto_exec.stop_requested = True
                self._proto_exec.running = False

    async def _process_message(self, client: ConnectedClient, raw: str):
        try:
            msg = json.loads(raw)
            client.last_activity = time.time()
            t = msg.get("type", "")

            if t == "heartbeat":
                await self._on_heartbeat(client, msg)
            elif t == "ping":
                await self._on_ping(client, msg)
            elif t == "rpc":
                await self._rpc.dispatch(client, msg)
            elif t == "move_robot":
                await self._on_move_robot(client, msg)
            elif t == "emergency_stop":
                await self._on_estop(client, msg)
            elif t == "soft_stop":
                if self._proto_exec:
                    self._proto_exec.stop_requested = True
                await self._reply(
                    client, msg.get("request_id"),
                    success=True, message="Soft stop requested",
                )
            elif t == "get_robot_state":
                await self._on_get_state(client, msg)
            else:
                await self._send_error(
                    client, msg.get("request_id"),
                    f"Unknown message type: {t}",
                )
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Bad JSON from {client.client_id}: {e}")
        except Exception as e:
            self.get_logger().error(
                f"Error processing msg: {e}\n{traceback.format_exc()}"
            )
            await self._send_error(client, None, str(e))

    # ── Simple message handlers ──────────────────────────────────

    async def _on_heartbeat(self, client, msg):
        client.heartbeat_count += 1
        hb = String()
        hb.data = (
            f"{client.client_id}:{msg.get('sequence', 0)}:"
            f"{msg.get('latency_ms', 0)}"
        )
        self.heartbeat_pub.publish(hb)
        await client.websocket.send(json.dumps({
            "type": "heartbeat_ack",
            "request_id": msg.get("request_id"),
            "sequence": msg.get("sequence", 0),
            "server_time_ns": time.time_ns(),
        }))

    async def _on_ping(self, client, msg):
        await client.websocket.send(json.dumps({
            "type": "pong",
            "request_id": msg.get("request_id"),
            "timestamp_ns": time.time_ns(),
        }))

    async def _on_estop(self, client, msg):
        rname = msg.get("robot")
        self.get_logger().warning(f"EMERGENCY STOP: {rname or 'ALL'}")
        m = String()
        m.data = rname or "all"
        self.estop_pub.publish(m)
        if self._proto_exec:
            self._proto_exec.stop_requested = True
        await client.websocket.send(json.dumps({
            "type": "emergency_stop_active",
            "request_id": msg.get("request_id"),
            "robots_stopped": (
                [rname] if rname else list(ROBOT_CONFIG.keys())
            ),
        }))

    async def _on_get_state(self, client, msg):
        rn = msg.get("robot", list(ROBOT_CONFIG.keys())[0])
        cfg = ROBOT_CONFIG.get(rn)
        if not cfg:
            await self._send_error(
                client, msg.get("request_id"), f"Unknown robot: {rn}"
            )
            return
        st = self._js_mgr.robot_states.get(rn, RobotStateInfo())
        await client.websocket.send(json.dumps({
            "type": "robot_state",
            "request_id": msg.get("request_id"),
            "robot": rn,
            "joint_names": cfg["joints"],
            "joint_positions": st.joint_positions,
            "joint_velocities": st.joint_velocities,
            "last_update": st.last_update,
        }))

    async def _on_move_robot(self, client, msg):
        rid = msg.get("request_id")
        rn = msg.get("robot_name", list(ROBOT_CONFIG.keys())[0])
        target = msg.get("target_joints")
        if rn not in ROBOT_CONFIG:
            await self._send_error(client, rid, f"Unknown robot: {rn}")
            return
        if not target or len(target) != 6:
            await self._send_error(
                client, rid, "target_joints with 6 values required"
            )
            return

        current = self._js_mgr.robot_states[rn].joint_positions
        if not current or len(current) != 6:
            current = list(ROBOT_CONFIG[rn]["home_position"])

        result = await self._planner.plan_to_joints(
            rn, target, current_joints=current,
        )
        if result is None:
            await self._send_error(client, rid, "Planning failed")
            return
        ros_traj = self._planner.result_to_ros_trajectory(result, rn)
        if ros_traj is None:
            await self._send_error(client, rid, "Trajectory conversion failed")
            return
        ok = await self._executor.execute(rn, ros_traj)
        await self._reply(client, rid, success=ok)

    # ── Utilities ────────────────────────────────────────────────

    async def _reply(self, client, rid, **kwargs):
        await client.websocket.send(json.dumps({
            "type": "response", "request_id": rid, **kwargs,
        }))

    async def _send_error(self, client, rid, error):
        try:
            await client.websocket.send(json.dumps({
                "type": "error", "request_id": rid, "error": error,
            }))
        except Exception as e:
            self.get_logger().error(f"Failed to send error: {e}")


# ── Entry point ──────────────────────────────────────────────────


async def main():
    rclpy.init()

    node = CommandGatewayNode()

    # Use SingleThreadedExecutor driven cooperatively from the asyncio
    # loop.  The previous MultiThreadedExecutor.spin() ran in a
    # background thread and BUSY-POLLED with 4 threads, starving the
    # Python GIL so badly that cuRobo planning (which alternates
    # between CUDA kernels and Python orchestration) slowed from
    # 0.2 s to 32 s per plan.
    #
    # Now we call spin_once(timeout_sec=0) every 1 ms from an asyncio
    # task.  This processes ROS2 callbacks cooperatively and yields
    # the GIL between iterations, giving cuRobo planning threads
    # fair access.
    ros_executor = SingleThreadedExecutor()
    ros_executor.add_node(node)

    await node.start_websocket_server()

    async def _spin_ros2():
        """Cooperatively spin ROS2 inside the asyncio event loop."""
        while rclpy.ok():
            ros_executor.spin_once(timeout_sec=0)
            await asyncio.sleep(0.001)  # 1 ms yield → ~1000 callbacks/s

    spin_task = asyncio.create_task(_spin_ros2())

    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        spin_task.cancel()
        ros_executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
