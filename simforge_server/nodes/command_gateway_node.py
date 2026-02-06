#!/usr/bin/env python3
"""
Command Gateway Node

Bridges WebSocket commands from Mac client to ROS 2 Action Servers.

This node:
1. Accepts WebSocket connections from Mac client
2. Forwards heartbeats to Safety Watchdog
3. Translates JSON commands to ROS 2 Actions
4. Streams feedback back to client
5. Handles emergency stop requests

The WebSocket server runs on port 8765 by default.
"""

import asyncio
import json
import time
import logging
import math
import numpy as np
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String, Header
from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose, Point, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

# Import FollowJointTrajectory action for UR robot driver
from rclpy.action import ActionClient
try:
    from control_msgs.action import FollowJointTrajectory
    from control_msgs.msg import JointTolerance
    from builtin_interfaces.msg import Duration as RosDuration
    HAS_FOLLOW_JOINT_TRAJECTORY = True
except ImportError:
    HAS_FOLLOW_JOINT_TRAJECTORY = False

# Import MoveIt ExecuteTrajectory action (the proper way to execute MoveIt plans)
try:
    from moveit_msgs.action import ExecuteTrajectory as ExecuteTrajectoryAction
    HAS_EXECUTE_TRAJECTORY = True
except ImportError:
    HAS_EXECUTE_TRAJECTORY = False

# Import custom messages for real robot control
try:
    from simforge_msgs.srv import MoveJoints
    HAS_MOVE_JOINTS = True
except ImportError:
    HAS_MOVE_JOINTS = False

import tf2_ros
from tf2_ros import Buffer, TransformListener

# Local imports for pose generation - try multiple import strategies
try:
    # Try installed package import first
    from simforge_server.utils.pose_generation import (
        ProtoSimParameters,
        ProtoPose,
        generate_proto_poses,
        transform_pose_to_world,
        count_poses,
        rpy_deg_to_quat_xyzw,
    )
except ImportError:
    try:
        # Try relative path import (for development)
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from utils.pose_generation import (
            ProtoSimParameters,
            ProtoPose,
            generate_proto_poses,
            transform_pose_to_world,
            count_poses,
            rpy_deg_to_quat_xyzw,
        )
    except ImportError:
        # Fallback: define minimal stubs if pose_generation not available
        import logging
        logging.getLogger(__name__).warning("pose_generation module not found, using fallback")
        from dataclasses import dataclass
        from typing import Sequence, Dict, Tuple, List
        
        @dataclass(frozen=True)
        class ProtoSimParameters:
            horiz: Sequence[float]
            vert: Sequence[float]
            distance: Sequence[float]
            roll: Sequence[float]
            pitch: Sequence[float]
            yaw: Sequence[float]
        
        @dataclass(frozen=True)
        class ProtoPose:
            parameters: Dict[str, float]
            position_m: Tuple[float, float, float]
            orientation_deg: Tuple[float, float, float]
            orientation_quat_xyzw: Tuple[float, float, float, float]
            def get_name(self) -> str:
                p = self.parameters
                return f"H{p['horiz']:.0f}_V{p['vert']:.0f}_D{p['distance']:.0f}_R{p['roll']:.0f}_P{p['pitch']:.0f}_Y{p['yaw']:.0f}"
        
        def generate_proto_poses(params: ProtoSimParameters) -> List[ProtoPose]:
            # Fallback: generate simple poses
            poses = []
            for h in params.horiz:
                for v in params.vert:
                    for p in params.pitch:
                        for y in params.yaw:
                            for d in params.distance:
                                for r in params.roll:
                                    poses.append(ProtoPose(
                                        parameters={'horiz': h, 'vert': v, 'distance': d, 'roll': r, 'pitch': p, 'yaw': y},
                                        position_m=(0.0, d/1000.0, 0.0),
                                        orientation_deg=(r, p, y),
                                        orientation_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                                    ))
            return poses
        
        def transform_pose_to_world(pose, target_pos, target_quat):
            return pose.position_m, pose.orientation_quat_xyzw
        
        def count_poses(params):
            return len(params.horiz) * len(params.vert) * len(params.distance) * len(params.roll) * len(params.pitch) * len(params.yaw)
        
        def rpy_deg_to_quat_xyzw(r, p, y):
            return (0.0, 0.0, 0.0, 1.0)

try:
    import websockets
    from websockets.server import serve, WebSocketServerProtocol
except ImportError:
    raise ImportError("websockets package required. Install with: pip install websockets>=12.0")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConnectedClient:
    """Information about a connected WebSocket client."""
    client_id: str
    websocket: WebSocketServerProtocol
    connected_at: float
    last_activity: float
    heartbeat_count: int = 0


class CommandGatewayNode(Node):
    """
    Command Gateway - bridges WebSocket to ROS 2 Actions.
    
    This node is the main entry point for Mac client commands.
    """

    def __init__(self):
        super().__init__('command_gateway')
        
        # Declare parameters
        self.declare_parameter('websocket_port', 8766)
        self.declare_parameter('websocket_host', '0.0.0.0')
        self.declare_parameter('max_clients', 5)
        self.declare_parameter('robot_name', 'nakul_ur5e')
        
        self.ws_port = self.get_parameter('websocket_port').value
        self.ws_host = self.get_parameter('websocket_host').value
        self.max_clients = self.get_parameter('max_clients').value
        self.default_robot = self.get_parameter('robot_name').value
        
        # Callback group for async operations
        self.callback_group = ReentrantCallbackGroup()
        
        # Connected clients
        self._connected_clients: Dict[str, ConnectedClient] = {}
        
        # Action clients for robots (registered dynamically)
        self.move_action_clients: Dict[str, ActionClient] = {}
        self.task_action_clients: Dict[str, ActionClient] = {}
        
        # === Publishers ===
        # Forward heartbeats to safety watchdog
        self.heartbeat_pub = self.create_publisher(
            String,
            '/safety/heartbeat',
            10
        )
        
        # Emergency stop publisher
        self.estop_pub = self.create_publisher(
            String,
            '/safety/emergency_stop',
            10
        )
        
        # Joint state publisher for simulation visualization
        self.joint_state_pub = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )
        
        # Current simulated joint positions (UR5e home position)
        self.sim_joint_positions = [0.0, -math.pi/2, 0.0, -math.pi/2, 0.0, 0.0]
        self.sim_joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Real robot joint positions (from UR driver joint_state_broadcaster)
        # These are updated by subscribing to /joint_states
        self.real_robot_joint_positions = None  # None until first message received
        self._real_robot_joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._real_robot_joint_state_callback,
            10
        )
        
        # Robot program running state (from UR driver)
        # The robot CANNOT execute trajectories unless the external control program is running
        self._robot_program_running = False
        try:
            from std_msgs.msg import Bool
            self._robot_program_sub = self.create_subscription(
                Bool,
                '/io_and_status_controller/robot_program_running',
                self._robot_program_running_callback,
                10
            )
            self.get_logger().info("Subscribed to robot_program_running topic")
        except Exception as e:
            self.get_logger().warn(f"Could not subscribe to robot_program_running: {e}")
        
        # Resend robot program service client
        self._resend_program_client = self.create_client(
            Trigger,
            '/io_and_status_controller/resend_robot_program'
        )
        
        # Proto-sim control flags
        self._proto_sim_running = False
        self._proto_sim_stop_requested = False
        
        # TF2 for frame lookups (target object position)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # IK solver (lazy initialized)
        self._ik_solver = None
        
        # === MoveIt 2 Service Clients for IK and Planning ===
        # Import MoveIt message types
        try:
            from moveit_msgs.srv import GetPositionIK, GetMotionPlan, GetPlanningScene
            from moveit_msgs.msg import RobotState, Constraints, JointConstraint, PositionIKRequest, MoveItErrorCodes
            self._has_moveit = True
            
            # IK service client
            self.compute_ik_client = self.create_client(
                GetPositionIK,
                '/compute_ik',
                callback_group=self.callback_group
            )
            
            # Motion planning service client
            self.plan_motion_client = self.create_client(
                GetMotionPlan,
                '/plan_kinematic_path',
                callback_group=self.callback_group
            )
            
            # Planning scene client
            self.planning_scene_client = self.create_client(
                GetPlanningScene,
                '/get_planning_scene',
                callback_group=self.callback_group
            )
            
            # Planning scene publisher for adding collision objects
            from moveit_msgs.msg import PlanningScene, CollisionObject
            from shape_msgs.msg import SolidPrimitive
            self.planning_scene_pub = self.create_publisher(
                PlanningScene,
                '/planning_scene',
                10
            )
            self._collision_objects_added = False
            
            self.get_logger().info("MoveIt 2 service clients initialized")
        except ImportError as e:
            self._has_moveit = False
            self.compute_ik_client = None
            self.plan_motion_client = None
            self.planning_scene_client = None
            self.get_logger().warn(f"MoveIt 2 not available: {e}. Using fallback IK.")
        
        # === Service Clients ===
        # Service to pause robot_control_node's joint state publishing during simulation
        self.pause_joint_pub_client = self.create_client(
            Trigger,
            '/robot/pause_joint_publishing',
            callback_group=self.callback_group
        )
        
        self.resume_joint_pub_client = self.create_client(
            Trigger,
            '/robot/resume_joint_publishing',
            callback_group=self.callback_group
        )
        
        # Protective stop service
        self.protective_stop_client = self.create_client(
            Trigger,
            '/robot/protective_stop',
            callback_group=self.callback_group
        )
        
        # Real robot MoveJoints service client
        if HAS_MOVE_JOINTS:
            self.move_joints_client = self.create_client(
                MoveJoints,
                '/robot/move_joints',
                callback_group=self.callback_group
            )
            self.get_logger().info("MoveJoints service client initialized for real robot control")
        else:
            self.move_joints_client = None
            self.get_logger().warn("MoveJoints service not available - real robot control disabled")
        
        # FollowJointTrajectory action client for UR Robot Driver
        # This is the preferred method for controlling the real robot with proper trajectory execution
        self.follow_trajectory_client = None
        if HAS_FOLLOW_JOINT_TRAJECTORY:
            # Try scaled_joint_trajectory_controller first (UR driver default), then fall back to joint_trajectory_controller
            self.follow_trajectory_client = ActionClient(
                self,
                FollowJointTrajectory,
                '/scaled_joint_trajectory_controller/follow_joint_trajectory',
                callback_group=self.callback_group
            )
            self.get_logger().info("FollowJointTrajectory action client initialized for UR robot driver")
        
        # MoveIt ExecuteTrajectory action client (preferred for executing MoveIt-planned trajectories)
        # This goes through MoveIt's execution pipeline which handles controller management,
        # start state validation, and proper trajectory forwarding.
        self.execute_trajectory_client = None
        if HAS_EXECUTE_TRAJECTORY:
            self.execute_trajectory_client = ActionClient(
                self,
                ExecuteTrajectoryAction,
                '/execute_trajectory',
                callback_group=self.callback_group
            )
            self.get_logger().info("MoveIt ExecuteTrajectory action client initialized")
        
        # === Register default robot ===
        self.register_robot(self.default_robot)
        
        # WebSocket server (started separately)
        self.ws_server = None
        
        self.get_logger().info(
            f"Command Gateway initialized - WebSocket on {self.ws_host}:{self.ws_port}"
        )

    def register_robot(self, robot_name: str):
        """Register action clients for a robot."""
        self.get_logger().info(f"Registering robot: {robot_name}")
        
        # MoveRobot action client
        # Note: Using String as placeholder until simforge_msgs is built
        # In production, this would be: ActionClient(self, MoveRobot, f'/{robot_name}/move_robot')
        
        # For now, we'll handle this without action clients since we need the custom messages
        pass

    async def start_websocket_server(self):
        """Start the WebSocket server."""
        self.get_logger().info(f"Starting WebSocket server on {self.ws_host}:{self.ws_port}")
        
        self.ws_server = await websockets.serve(
            self.handle_client,
            self.ws_host,
            self.ws_port,
            ping_interval=30,  # Send ping every 30s
            ping_timeout=300,  # Allow 5 minutes for long-running proto-sim
        )
        
        self.get_logger().info("WebSocket server started")

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str = None):
        """Handle a WebSocket client connection.
        
        Note: path parameter is optional for websockets 12.0+ compatibility.
        """
        client_id = f"client_{id(websocket)}"
        
        # Check max clients
        if len(self._connected_clients) >= self.max_clients:
            self.get_logger().warning(f"Max clients reached, rejecting {client_id}")
            await websocket.close(1013, "Max clients reached")
            return
        
        # Register client
        client = ConnectedClient(
            client_id=client_id,
            websocket=websocket,
            connected_at=time.time(),
            last_activity=time.time(),
        )
        self._connected_clients[client_id] = client
        
        self.get_logger().info(f"Client connected: {client_id} from {websocket.remote_address}")
        
        try:
            async for message in websocket:
                await self.process_message(client, message)
                
        except websockets.ConnectionClosed as e:
            self.get_logger().info(f"Client disconnected: {client_id} - {e}")
        except Exception as e:
            self.get_logger().error(f"Client error: {client_id} - {e}")
        finally:
            # Unregister client
            self._connected_clients.pop(client_id, None)
            self.get_logger().info(f"Client removed: {client_id}")

    async def process_message(self, client: ConnectedClient, message: str):
        """Process a message from a client."""
        try:
            msg = json.loads(message)
            msg_type = msg.get('type', '')
            
            client.last_activity = time.time()
            
            if msg_type == 'heartbeat':
                await self.handle_heartbeat(client, msg)
            elif msg_type == 'ping':
                await self.handle_ping(client, msg)
            elif msg_type == 'move_robot':
                await self.handle_move_robot(client, msg)
            elif msg_type == 'execute_task':
                await self.handle_execute_task(client, msg)
            elif msg_type == 'emergency_stop':
                await self.handle_emergency_stop(client, msg)
            elif msg_type == 'protective_stop':
                await self.handle_protective_stop(client, msg)
            elif msg_type == 'get_robot_state':
                await self.handle_get_robot_state(client, msg)
            elif msg_type == 'rpc':
                await self.handle_rpc(client, msg)
            else:
                await self.send_error(client, msg.get('request_id'), f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON from {client.client_id}: {e}")
        except Exception as e:
            self.get_logger().error(f"Error processing message: {e}")
            await self.send_error(client, None, str(e))

    async def handle_heartbeat(self, client: ConnectedClient, msg: Dict[str, Any]):
        """Forward heartbeat to safety watchdog."""
        client.heartbeat_count += 1
        
        # Extract client ID from message if provided
        msg_client_id = msg.get('client_id', client.client_id)
        
        # Publish to ROS 2 topic
        heartbeat_msg = String()
        heartbeat_msg.data = f"{msg_client_id}:{msg.get('sequence', 0)}:{msg.get('latency_ms', 0)}"
        self.heartbeat_pub.publish(heartbeat_msg)

    async def handle_ping(self, client: ConnectedClient, msg: Dict[str, Any]):
        """Handle ping request."""
        response = {
            'type': 'result',
            'request_id': msg.get('request_id'),
            'timestamp_ns': time.time_ns(),
            'server_time': time.time(),
        }
        await client.websocket.send(json.dumps(response))

    async def handle_move_robot(self, client: ConnectedClient, msg: Dict[str, Any]):
        """Handle move robot command."""
        request_id = msg.get('request_id')
        robot_name = msg.get('robot_name', 'ur20')
        
        self.get_logger().info(
            f"Move request from {client.client_id} for {robot_name}"
        )
        
        # Extract parameters
        motion_type = msg.get('motion_type', 0)
        target_joints = msg.get('target_joints')
        target_pose = msg.get('target_pose')
        velocity_scale = msg.get('velocity_scale', 0.5)
        acceleration_scale = msg.get('acceleration_scale', 0.5)
        collision_check = msg.get('collision_check_enabled', True)
        
        # TODO: In production, this would send to the ROS 2 action server
        # For now, simulate the motion execution
        
        # Send feedback updates
        for progress in [0.1, 0.3, 0.5, 0.7, 0.9]:
            feedback = {
                'type': 'feedback',
                'request_id': request_id,
                'progress': progress,
                'status': 'executing',
                'distance_to_goal': (1.0 - progress) * 0.5,
                'time_remaining': (1.0 - progress) * 2.0,
            }
            await client.websocket.send(json.dumps(feedback))
            await asyncio.sleep(0.2)
        
        # Send result
        result = {
            'type': 'result',
            'request_id': request_id,
            'success': True,
            'message': 'Motion completed',
            'final_joint_positions': target_joints or [0.0] * 6,
            'execution_time_sec': 1.0,
            'planning_time_sec': 0.1,
            'replan_count': 0,
        }
        await client.websocket.send(json.dumps(result))

    async def handle_execute_task(self, client: ConnectedClient, msg: Dict[str, Any]):
        """Handle VLA task execution."""
        request_id = msg.get('request_id')
        instruction = msg.get('instruction', '')
        robot_name = msg.get('robot_name', 'ur20')
        
        self.get_logger().info(
            f"Task request from {client.client_id}: '{instruction}'"
        )
        
        # TODO: In production, this would:
        # 1. Get current camera image
        # 2. Call VLA inference service
        # 3. Execute waypoints via MoveRobot action
        # 4. Stream feedback
        
        # Simulate task execution
        steps = ['approach', 'grasp', 'lift', 'move', 'place']
        
        for i, step in enumerate(steps):
            feedback = {
                'type': 'feedback',
                'request_id': request_id,
                'overall_progress': (i + 1) / len(steps),
                'current_step': step,
                'current_step_index': i,
                'total_steps': len(steps),
                'step_progress': 1.0,
                'confidence': 0.9,
                'status': f'Executing: {step}',
            }
            await client.websocket.send(json.dumps(feedback))
            await asyncio.sleep(0.5)
        
        result = {
            'type': 'result',
            'request_id': request_id,
            'success': True,
            'message': f'Task completed: {instruction}',
            'total_time_sec': len(steps) * 0.5,
        }
        await client.websocket.send(json.dumps(result))

    async def handle_emergency_stop(self, client: ConnectedClient, msg: Dict[str, Any]):
        """Handle emergency stop."""
        self.get_logger().fatal(
            f"EMERGENCY STOP from {client.client_id}"
        )
        
        # Publish to ROS 2
        estop_msg = String()
        estop_msg.data = f"E-STOP from {msg.get('client_id', client.client_id)}"
        self.estop_pub.publish(estop_msg)
        
        # Notify all clients
        notification = {
            'type': 'emergency_stop_active',
            'triggered_by': client.client_id,
            'timestamp': time.time(),
        }
        await self.broadcast(json.dumps(notification))

    async def handle_protective_stop(self, client: ConnectedClient, msg: Dict[str, Any]):
        """Handle protective stop."""
        self.get_logger().warning(
            f"Protective stop from {client.client_id}: {msg.get('reason', 'No reason')}"
        )
        
        # Call protective stop service
        if self.protective_stop_client.service_is_ready():
            request = Trigger.Request()
            future = self.protective_stop_client.call_async(request)
            # Don't wait for result - protective stop should be immediate

    async def handle_get_robot_state(self, client: ConnectedClient, msg: Dict[str, Any]):
        """Handle get robot state request."""
        request_id = msg.get('request_id')
        robot_name = msg.get('robot_name', 'ur20')
        
        # TODO: In production, get actual robot state
        # For now, return simulated state
        
        result = {
            'type': 'result',
            'request_id': request_id,
            'robot_state': {
                'robot_name': robot_name,
                'joint_positions': [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
                'joint_velocities': [0.0] * 6,
                'robot_mode': 1,  # IDLE
                'protective_stop_active': False,
                'emergency_stop_active': False,
            }
        }
        await client.websocket.send(json.dumps(result))

    async def handle_rpc(self, client: ConnectedClient, msg: Dict[str, Any]):
        """Handle generic RPC calls from client."""
        request_id = msg.get('request_id')
        method = msg.get('method', '')
        params = msg.get('params', {})
        
        self.get_logger().info(f"RPC call from {client.client_id}: {method}")
        
        try:
            if method == 'get_environment_info':
                result = await self.rpc_get_environment_info(params)
            elif method == 'get_robot_status':
                result = await self.rpc_get_robot_status(params)
            elif method == 'prepare_mode':
                result = await self.rpc_prepare_mode(params)
            elif method == 'run_proto_sim':
                # Run long-running proto_sim in a separate task to not block message processing
                # This allows heartbeats to continue being processed
                asyncio.create_task(self.rpc_run_proto_sim(client, request_id, params))
                return  # run_proto_sim sends its own responses
            elif method == 'stop_proto_sim':
                result = await self.rpc_stop_proto_sim(params)
            elif method == 'check_collision':
                result = await self.rpc_check_collision(params)
            elif method == 'plan_motion':
                result = await self.rpc_plan_motion(params)
            else:
                result = {'success': False, 'error': f'Unknown RPC method: {method}'}
            
            response = {
                'type': 'rpc_result',
                'request_id': request_id,
                **result,
            }
            await client.websocket.send(json.dumps(response))
            
        except Exception as e:
            self.get_logger().error(f"RPC error: {e}")
            await self.send_error(client, request_id, str(e))

    async def rpc_get_environment_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get available robots and objects with TF data in base_link frame.
        
        Returns transforms for all known objects so the client can compute
        poses locally in the robot's coordinate frame.
        """
        objects = ['face_link', 'table_link', 'shop_floor']
        object_transforms = {}
        
        for obj_name in objects:
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link',
                    obj_name,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                object_transforms[obj_name] = {
                    'position': [
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z,
                    ],
                    'orientation': [
                        transform.transform.rotation.x,
                        transform.transform.rotation.y,
                        transform.transform.rotation.z,
                        transform.transform.rotation.w,
                    ],
                }
            except Exception as e:
                self.get_logger().warn(f"Could not get transform for {obj_name}: {e}")
                object_transforms[obj_name] = None
        
        return {
            'success': True,
            'robots': [self.default_robot],
            'objects': objects,
            'object_transforms': object_transforms,
            'reference_frame': 'base_link',
        }

    async def rpc_get_robot_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get the status of the real robot connection.
        
        Returns information about whether the real robot is available and connected,
        which helps the client determine which modes are available.
        
        Returns:
            - real_robot_available: Whether the UR driver is running and responsive
            - simulation_available: Always True (simulation is always available)
            - available_modes: List of available modes based on connections
            - connection_details: Details about action server availability
        """
        real_robot_available = False
        connection_details = {}
        
        # Check FollowJointTrajectory action server (UR driver)
        if self.follow_trajectory_client:
            try:
                # Quick check if action server is responding
                is_ready = self.follow_trajectory_client.wait_for_server(timeout_sec=1.0)
                real_robot_available = is_ready
                connection_details['follow_trajectory_action'] = 'available' if is_ready else 'not_available'
            except Exception as e:
                connection_details['follow_trajectory_action'] = f'error: {str(e)}'
        else:
            connection_details['follow_trajectory_action'] = 'not_initialized'
        
        # Check MoveJoints service (legacy fallback)
        if self.move_joints_client:
            connection_details['move_joints_service'] = 'available'
        else:
            connection_details['move_joints_service'] = 'not_available'
        
        # Check MoveIt availability
        connection_details['moveit'] = 'available' if self._has_moveit else 'not_available'
        
        # All modes are always selectable — the client decides.
        # We report connection status so the GUI can show an indicator.
        available_modes = ['simulation', 'real', 'both']
        
        # Also check MoveIt ExecuteTrajectory (our primary execution path)
        if self.execute_trajectory_client:
            try:
                exec_ready = self.execute_trajectory_client.wait_for_server(timeout_sec=1.0)
                connection_details['execute_trajectory_action'] = 'available' if exec_ready else 'not_available'
            except Exception as e:
                connection_details['execute_trajectory_action'] = f'error: {str(e)}'
        
        # Check robot_program_running flag
        connection_details['robot_program_running'] = self._robot_program_running
        
        return {
            'success': True,
            'real_robot_available': real_robot_available,
            'simulation_available': True,
            'available_modes': available_modes,
            'connection_details': connection_details,
            'current_joint_positions': self.real_robot_joint_positions if real_robot_available else self.sim_joint_positions,
            'position_source': 'real_robot' if real_robot_available else 'simulation',
        }

    async def rpc_prepare_mode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-flight check for a requested execution mode.
        
        Called by the client before starting a protocol run to verify that the
        requested mode (simulation/real/both) can actually execute. For 'real'
        and 'both' modes this checks the UR driver, controller, and robot
        program status with generous timeouts.
        
        Returns:
            ready: True if the mode can be used right now
            message: Human-readable status message
            can_retry: True if a retry might succeed (e.g. robot booting)
        """
        mode = params.get('mode', 'simulation')
        
        if mode == 'simulation':
            return {'success': True, 'ready': True, 'message': 'Simulation mode always available'}
        
        # For real / both — run thorough checks
        checks = []
        all_ok = True
        
        # 1. FollowJointTrajectory action server (UR driver)
        fjt_ok = False
        if self.follow_trajectory_client:
            fjt_ok = self.follow_trajectory_client.wait_for_server(timeout_sec=3.0)
        checks.append(('UR Driver (FollowJointTrajectory)', fjt_ok))
        if not fjt_ok:
            all_ok = False
        
        # 2. MoveIt ExecuteTrajectory action
        exec_ok = False
        if self.execute_trajectory_client:
            exec_ok = self.execute_trajectory_client.wait_for_server(timeout_sec=3.0)
        checks.append(('MoveIt ExecuteTrajectory', exec_ok))
        if not exec_ok:
            all_ok = False
        
        # 3. Robot program running
        checks.append(('Robot Program Running', self._robot_program_running))
        if not self._robot_program_running:
            all_ok = False
            # Try to resend
            self.get_logger().info("Robot program not running — attempting resend for prepare_mode")
            try:
                ready = await self._ensure_robot_ready(timeout=10.0)
                if ready:
                    checks[-1] = ('Robot Program Running', True)
                    all_ok = fjt_ok and exec_ok  # re-evaluate
            except Exception:
                pass
        
        # Build message
        lines = []
        for name, ok in checks:
            mark = '✓' if ok else '✗'
            lines.append(f"{mark} {name}")
        msg = '\n'.join(lines)
        
        if all_ok:
            return {
                'success': True,
                'ready': True,
                'message': f'Robot ready for {mode} mode\n{msg}',
            }
        else:
            return {
                'success': True,
                'ready': False,
                'message': f'Robot NOT ready for {mode} mode:\n{msg}',
                'can_retry': True,
            }

    async def rpc_run_proto_sim(
        self,
        client: ConnectedClient,
        request_id: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run protocol simulation with collision-checked robot movement.
        
        The server receives pre-computed poses from the client in base_link frame.
        Each pose is validated via MoveIt IK and collision checking.
        Poses that would cause collision are REJECTED (not executed).
        
        Parameters (new architecture - poses from client):
        - poses: List of poses in base_link frame, each with:
            - name: Pose name (e.g., "H0_V0_D250_R-90_P0_Y0")
            - position: [x, y, z] in meters
            - orientation: [qx, qy, qz, qw] quaternion
            - parameters: Original sampling parameters
        
        Parameters (legacy - server generates poses):
        - horiz, vert, distance, roll, pitch, yaw: Sampling parameters
        - target_object: Target frame name for pose transformation
        
        Common parameters:
        - idle_time: Time to wait at each pose (seconds)
        - mode: 'simulation', 'real', or 'both'
        - move_speed: Movement speed scale (0.0-1.0)
        """
        if self._proto_sim_running:
            return {
                'success': False,
                'error': 'Protocol simulation already running',
            }
        
        self._proto_sim_running = True
        self._proto_sim_stop_requested = False
        
        robot_name = params.get('robot_name', self.default_robot)
        idle_time = params.get('idle_time', 2.0)
        mode = params.get('mode', 'simulation')
        # Default to conservative speed for real robot safety.
        # The velocity_scale is multiplied by joint_limits.yaml max_velocity values.
        # With max_velocity=1.0 rad/s (shoulder) and 1.5 rad/s (wrist), a scale of 0.3
        # gives actual velocities of 0.3 and 0.45 rad/s respectively — safe for UR5e.
        # Cap at 0.3 for real/both modes to prevent velocity-limit protective stops.
        requested_speed = params.get('move_speed', 0.3)
        if mode in ('real', 'both'):
            move_speed = min(0.3, requested_speed)
        else:
            move_speed = min(0.5, requested_speed)
        
        # Pre-flight validation for real robot modes
        if mode in ('real', 'both'):
            self.get_logger().info(f"Pre-flight check for mode={mode}...")
            fjt_ok = False
            if self.follow_trajectory_client:
                fjt_ok = self.follow_trajectory_client.wait_for_server(timeout_sec=3.0)
            if not fjt_ok:
                self._proto_sim_running = False
                return {
                    'success': False,
                    'error': ('Real robot not reachable — the UR driver FollowJointTrajectory '
                              'action server is not responding. Check that the robot is powered '
                              'on, connected, and the external control program is loaded on the '
                              'teach pendant.'),
                }
            # Ensure robot program is running (attempts resend if needed)
            robot_ready = await self._ensure_robot_ready(timeout=15.0)
            if not robot_ready:
                self._proto_sim_running = False
                return {
                    'success': False,
                    'error': ('Robot program not running. Please load and start the external '
                              'control program on the teach pendant, then try again.'),
                }
            self.get_logger().info("Pre-flight check passed — robot ready")
        
        # Check if client provided pre-computed poses (new architecture)
        client_poses = params.get('poses', None)
        
        if client_poses:
            # NEW: Client-side pose generation - poses are already in base_link frame
            self.get_logger().info(f"Proto-sim: {robot_name}, mode={mode}, client-computed poses")
            self.get_logger().info(f"  Received {len(client_poses)} pre-computed poses in base_link frame")
            world_poses = client_poses
            total_poses = len(world_poses)
        else:
            # LEGACY: Server-side pose generation (for backwards compatibility)
            self.get_logger().warn("Using legacy server-side pose generation - consider updating client")
            target_object = params.get('target_object', 'face_link')
            horiz = params.get('horiz', [0])
            vert = params.get('vert', [0])
            distance = params.get('distance', [250, 350, 450, 550])
            roll = params.get('roll', [-90])
            pitch = params.get('pitch', [-45, -30, 0, 15])
            yaw = params.get('yaw', [-30, 0, 30])
            randomize = params.get('randomize', False)
            
            self.get_logger().info(f"Proto-sim: {robot_name} -> {target_object}, mode={mode}")
            self.get_logger().info(f"  horiz={horiz}, vert={vert}, distance={distance}")
            self.get_logger().info(f"  roll={roll}, pitch={pitch}, yaw={yaw}")
            
            # Get target object transform
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link', target_object, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=2.0)
                )
                target_position = (
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                )
                target_orientation = (
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                )
            except Exception as e:
                self._proto_sim_running = False
                return {
                    'success': False,
                    'error': f"Cannot find target object transform: {e}",
                }
            
            # Generate and transform poses
            proto_params = ProtoSimParameters(
                horiz=horiz, vert=vert, distance=distance,
                roll=roll, pitch=pitch, yaw=yaw,
            )
            local_poses = generate_proto_poses(proto_params)
            if randomize:
                import random
                random.shuffle(local_poses)
            
            # Convert to world poses format
            world_poses = []
            for pose in local_poses:
                world_pos, world_quat = transform_pose_to_world(
                    pose, target_position, target_orientation
                )
                world_poses.append({
                    'name': pose.get_name(),
                    'position': list(world_pos),
                    'orientation': list(world_quat),
                    'parameters': pose.parameters,
                })
            total_poses = len(world_poses)
            self.get_logger().info(f"  Generated {total_poses} poses")
        
        self.get_logger().info(f"  idle_time={idle_time}, move_speed={move_speed}")
        
        # Add collision objects to planning scene
        self._add_collision_objects_to_planning_scene()
        
        # Pause real robot's joint state publishing during simulation
        if mode in ('simulation', 'both'):
            try:
                await self._pause_real_robot_joint_publishing()
            except Exception as e:
                self.get_logger().error(f"Failed to pause joint publishing: {e}")
        
        # === FIRST: Move robot to home position before starting protocol ===
        home_position = [0.0, -math.pi/2, 0.0, -math.pi/2, 0.0, 0.0]
        self.get_logger().info("Moving to HOME position before starting protocol...")
        
        # Determine current position for planning
        if mode in ('both', 'real') and self.real_robot_joint_positions:
            current_joints = self.real_robot_joint_positions
            self.get_logger().info(f"Current REAL robot position: {[f'{j:.2f}' for j in current_joints]}")
            self.sim_joint_positions = list(current_joints)
        else:
            current_joints = self.sim_joint_positions
            self.get_logger().info(f"Current SIM position: {[f'{j:.2f}' for j in current_joints]}")
        
        # Check if already at home (within tolerance)
        at_home = all(abs(current_joints[i] - home_position[i]) < 0.05 for i in range(6))
        
        if not at_home:
            # Plan trajectory to home
            init_home_trajectory = None
            if self._has_moveit and self.plan_motion_client:
                try:
                    init_home_trajectory = self._plan_motion_moveit_sync(
                        current_joints, home_position,
                        velocity_scale=move_speed,
                        acceleration_scale=move_speed,
                    )
                    if init_home_trajectory:
                        # MoveIt with TOTG should provide timing, but log it for debugging
                        waypoints = init_home_trajectory.get('waypoints', [])
                        time_from_start = init_home_trajectory.get('time_from_start', [])
                        total_time = time_from_start[-1] if time_from_start else 0.0
                        self.get_logger().info(f"Planned initial home trajectory: {len(waypoints)} waypoints, {total_time:.2f}s")
                except Exception as e:
                    self.get_logger().warn(f"Failed to plan initial home trajectory: {e}")
            
            if init_home_trajectory:
                # Execute on real robot first if in 'both' or 'real' mode
                if mode in ('both', 'real'):
                    self.get_logger().info("Sending initial home trajectory to real robot...")
                    home_success, home_duration = await self._execute_real_robot_trajectory(
                        init_home_trajectory,
                        velocity_scale=move_speed,
                        wait_for_completion=True,  # Wait for real robot to complete
                    )
                    if home_success:
                        self.sim_joint_positions = list(home_position)
                    else:
                        self.get_logger().warn("Initial home trajectory failed on real robot")
                
                # Execute simulation (this blocks for trajectory duration, syncing with real robot)
                if mode in ('simulation', 'both'):
                    await self._execute_trajectory(init_home_trajectory, publish_rate=50.0)
                elif mode == 'real':
                    # For real-only mode, wait for trajectory to complete
                    time_from_start = init_home_trajectory.get('time_from_start', [])
                    if time_from_start:
                        wait_duration = time_from_start[-1]
                        self.get_logger().info(f"Waiting {wait_duration:.2f}s for robot to reach HOME...")
                        await asyncio.sleep(wait_duration + 2.0)  # Add buffer
                
                self.sim_joint_positions = list(home_position)
                self.get_logger().info("Robot now at HOME position, starting protocol...")
            else:
                self.get_logger().warn("Could not plan path to home, starting from current position")
        else:
            self.get_logger().info("Robot already at HOME position, starting protocol...")
        
        # Execute poses
        completed = 0
        failed = 0
        collision_rejected = 0
        real_failed = 0
        client_disconnected = False
        
        async def safe_send_feedback(feedback_data):
            nonlocal client_disconnected
            if client_disconnected:
                return False
            try:
                await client.websocket.send(json.dumps(feedback_data))
                return True
            except Exception:
                client_disconnected = True
                return False
        
        try:
            self.get_logger().info(f"Starting pose execution: {total_poses} poses, collision-checked")
            
            for i, pose_data in enumerate(world_poses):
                if self._proto_sim_stop_requested:
                    self.get_logger().info("Proto-sim stopped by user request")
                    break
                
                pose_name = pose_data.get('name', f'pose_{i}')
                position = tuple(pose_data['position'])
                orientation = tuple(pose_data['orientation'])
                
                self.get_logger().info(f"Pose {i+1}/{total_poses}: {pose_name}")
                self.get_logger().info(f"  Position: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
                
                # Solve IK for this pose, using current joints as seed for continuity
                target_joints = self._solve_ik_for_pose(position, orientation, seed_joints=self.sim_joint_positions)
                
                if target_joints is None:
                    self.get_logger().warn(f"IK failed for pose {pose_name}")
                    failed += 1
                    await safe_send_feedback({
                        'type': 'rpc_feedback',
                        'request_id': request_id,
                        'current_pose_index': i,
                        'total_poses': total_poses,
                        'current_pose_name': pose_name,
                        'status': 'ik_failed',
                    })
                    continue
                
                # Send progress update - moving
                await safe_send_feedback({
                    'type': 'rpc_feedback',
                    'request_id': request_id,
                    'current_pose_index': i,
                    'total_poses': total_poses,
                    'current_pose_name': pose_name,
                    'progress_percent': (i / total_poses) * 100,
                    'status': 'moving',
                })
                
                # Execute movement with COLLISION CHECKING
                # First, plan the trajectory with MoveIt (this gives us timing)
                sim_success = True
                real_success = True
                trajectory_data = None
                
                self.get_logger().info(f"Moving to joints: {[f'{j:.2f}' for j in target_joints]}")
                
                # Determine start position for planning
                # For "both" or "real" mode, use real robot's actual position
                # For "simulation" mode, use sim position
                if mode in ('both', 'real') and self.real_robot_joint_positions:
                    start_joints = self.real_robot_joint_positions
                    self.get_logger().info(f"Planning from REAL robot state: {[f'{j:.2f}' for j in start_joints]}")
                    # CRITICAL: Sync simulation to real robot position before planning
                    self.sim_joint_positions = list(self.real_robot_joint_positions)
                else:
                    start_joints = self.sim_joint_positions
                    self.get_logger().info(f"Planning from SIM state: {[f'{j:.2f}' for j in start_joints]}")
                
                # Plan trajectory first (needed for both simulation and real robot)
                if self._has_moveit and self.plan_motion_client:
                    try:
                        trajectory_data = self._plan_motion_moveit_sync(
                            start_joints, target_joints,
                            velocity_scale=move_speed,
                            acceleration_scale=move_speed,
                        )
                        if trajectory_data:
                            waypoints = trajectory_data.get('waypoints', [])
                            self.get_logger().info(f"MoveIt planned {len(waypoints)} waypoints")
                            
                            # MoveIt with AddTimeOptimalParameterization should provide timing
                            # Log timing info for debugging
                            time_from_start = trajectory_data.get('time_from_start', [])
                            if time_from_start:
                                self.get_logger().info(f"MoveIt trajectory timing: {time_from_start[-1]:.2f}s total")
                            else:
                                self.get_logger().warn("MoveIt did not provide trajectory timing - will use fallback")
                        else:
                            self.get_logger().warn("MoveIt planning failed - no collision-free path")
                    except Exception as e:
                        self.get_logger().warn(f"MoveIt planning error: {e}")
                
                # If planning failed and we require collision check, reject this pose
                if trajectory_data is None:
                    self.get_logger().warn(f"Pose {pose_name} REJECTED due to collision")
                    collision_rejected += 1
                    await safe_send_feedback({
                        'type': 'rpc_feedback',
                        'request_id': request_id,
                        'current_pose_index': i,
                        'total_poses': total_poses,
                        'current_pose_name': pose_name,
                        'status': 'collision_rejected',
                    })
                    continue
                
                # CRITICAL: Execute trajectory on real robot with WAIT FOR COMPLETION
                # This ensures the robot reaches the target before we plan the next trajectory.
                # The simulation will run in parallel with the same duration.
                
                real_success = True
                real_duration = 0.0
                
                if mode in ('both', 'real'):
                    # Re-read the real robot state right before execution.
                    # Planning may have taken many seconds; the state used
                    # for planning could be stale, causing PATH_TOLERANCE_VIOLATED.
                    if self.real_robot_joint_positions:
                        fresh_start = list(self.real_robot_joint_positions)
                        drift = max(abs(fresh_start[j] - start_joints[j]) for j in range(6))
                        if drift > 0.05:  # > 3 degrees
                            self.get_logger().warn(
                                f"Robot drifted {math.degrees(drift):.1f}° during planning — "
                                f"re-planning from fresh state"
                            )
                            start_joints = fresh_start
                            self.sim_joint_positions = list(fresh_start)
                            # Re-plan with the fresh start
                            trajectory_data = self._plan_motion_moveit_sync(
                                start_joints, target_joints,
                                velocity_scale=move_speed,
                                acceleration_scale=move_speed,
                            )
                            if trajectory_data is None:
                                self.get_logger().warn(f"Re-plan from fresh state failed")
                                collision_rejected += 1
                                await safe_send_feedback({
                                    'type': 'rpc_feedback',
                                    'request_id': request_id,
                                    'current_pose_index': i,
                                    'total_poses': total_poses,
                                    'current_pose_name': pose_name,
                                    'status': 'collision_rejected',
                                })
                                continue
                    
                    self.get_logger().info("Sending trajectory to real robot (via UR driver)...")
                    real_success, real_duration = await self._execute_real_robot_trajectory(
                        trajectory_data,
                        velocity_scale=move_speed,
                        wait_for_completion=True,  # CRITICAL: Wait for robot to finish
                    )
                    if not real_success:
                        self.get_logger().error(f"Real robot trajectory FAILED for pose {pose_name}")
                        real_failed += 1
                        # Send failure feedback for this pose
                        await safe_send_feedback({
                            'type': 'rpc_feedback',
                            'request_id': request_id,
                            'current_pose_index': i,
                            'total_poses': total_poses,
                            'current_pose_name': pose_name,
                            'status': 'real_robot_failed',
                        })
                        # After a failure (likely protective stop), wait and try to
                        # recover the robot program before the next pose.
                        self.get_logger().info(
                            "Waiting 5s for robot recovery after execution failure..."
                        )
                        await asyncio.sleep(5.0)
                        # Try to re-establish the robot program
                        recovered = await self._ensure_robot_ready(timeout=15.0)
                        if not recovered:
                            self.get_logger().error(
                                "Robot not recoverable after protective stop — "
                                "stopping proto-sim early"
                            )
                            break  # Exit the pose loop entirely
                        continue  # Skip to next pose - don't count as completed
                    else:
                        # Sync simulation to target after real robot completes
                        self.sim_joint_positions = list(target_joints)
                
                # Execute simulation visualization (only in simulation-only mode)
                # In 'both' mode, we already waited for the real robot, so just update sim state
                if mode == 'simulation':
                    await self._execute_trajectory(
                        trajectory_data,
                        publish_rate=50.0,
                    )
                    sim_success = True
                    self.sim_joint_positions = list(target_joints)
                elif mode == 'both':
                    # Already synced above after real robot completed
                    sim_success = True
                
                # Send progress update - at pose
                await safe_send_feedback({
                    'type': 'rpc_feedback',
                    'request_id': request_id,
                    'current_pose_index': i,
                    'total_poses': total_poses,
                    'current_pose_name': pose_name,
                    'status': 'idle',
                })
                
                # Wait at pose
                for _ in range(int(idle_time * 10)):
                    if self._proto_sim_stop_requested:
                        break
                    await asyncio.sleep(0.1)
                    self._publish_joint_state()
                
                completed += 1
        
            # After all poses, smoothly return to home position
            home_position = [0.0, -math.pi/2, 0.0, -math.pi/2, 0.0, 0.0]
            
            if not self._proto_sim_stop_requested:
                self.get_logger().info("Returning to home position...")
                
                # Determine start position for home planning (same logic as pose planning)
                if mode in ('both', 'real') and self.real_robot_joint_positions:
                    home_start_joints = self.real_robot_joint_positions
                    self.sim_joint_positions = list(self.real_robot_joint_positions)
                else:
                    home_start_joints = self.sim_joint_positions
                
                # Plan trajectory to home
                home_trajectory = None
                if self._has_moveit and self.plan_motion_client:
                    try:
                        home_trajectory = self._plan_motion_moveit_sync(
                            home_start_joints, home_position,
                            velocity_scale=min(0.3, move_speed),  # Slower for safety
                            acceleration_scale=min(0.3, move_speed),
                        )
                        
                        # MoveIt with TOTG should provide timing, log for debugging
                        if home_trajectory:
                            waypoints = home_trajectory.get('waypoints', [])
                            time_from_start = home_trajectory.get('time_from_start', [])
                            total_time = time_from_start[-1] if time_from_start else 0.0
                            self.get_logger().info(f"Planned home trajectory: {len(waypoints)} waypoints, {total_time:.2f}s")
                    except Exception as e:
                        self.get_logger().warn(f"MoveIt planning to home failed: {e}")
                
                if home_trajectory:
                    # Execute return home trajectory with wait for completion
                    if mode in ('both', 'real'):
                        _, _ = await self._execute_real_robot_trajectory(
                            home_trajectory,
                            velocity_scale=0.3,  # Slower for safety
                            wait_for_completion=True,
                        )
                        self.sim_joint_positions = list(home_position)
                    
                    # Execute simulation (only in simulation-only mode)
                    if mode == 'simulation':
                        await self._execute_trajectory(
                            home_trajectory,
                            publish_rate=50.0,
                        )
                        self.sim_joint_positions = list(home_position)
                else:
                    # Fallback: direct move
                    if mode in ('simulation', 'both'):
                        await self._execute_sim_movement(
                            home_position,
                            move_speed=move_speed,
                            require_collision_check=False,
                        )
                    if mode in ('real', 'both'):
                        await self._execute_real_robot_movement(
                            home_position,
                            velocity=0.3,
                            acceleration=0.3,
                        )
                
        finally:
            self._proto_sim_running = False
            if mode in ('simulation', 'both'):
                await self._resume_real_robot_joint_publishing()
        
        # Final result
        result = {
            'type': 'rpc_result',
            'request_id': request_id,
            'success': completed > 0 and real_failed == 0 and not self._proto_sim_stop_requested,
            'message': f'Completed {completed}/{total_poses} poses ({collision_rejected} rejected for collision, {failed} IK failed, {real_failed} real robot failed)',
            'completed': completed,
            'collision_rejected': collision_rejected,
            'ik_failed': failed,
            'real_failed': real_failed,
            'total': total_poses,
            'stopped': self._proto_sim_stop_requested,
        }
        
        self.get_logger().info(f"Proto-sim finished: {result['message']}")
        await safe_send_feedback(result)
        
        return result
    
    async def _pause_real_robot_joint_publishing(self) -> None:
        """Pause real robot's joint state publishing for simulation mode."""
        self.get_logger().info("Checking if pause service is ready...")
        
        # Wait a bit for service to become ready
        for _ in range(10):
            if self.pause_joint_pub_client.service_is_ready():
                break
            self.get_logger().info("Waiting for pause service...")
            await asyncio.sleep(0.1)
        
        if not self.pause_joint_pub_client.service_is_ready():
            self.get_logger().warn("Pause joint publishing service not available after waiting")
            return
        
        try:
            self.get_logger().info("Calling pause service...")
            request = Trigger.Request()
            future = self.pause_joint_pub_client.call_async(request)
            # Simple approach - just call and continue, check result briefly
            await asyncio.sleep(0.2)  # Give service time to process
            if future.done():
                result = future.result()
                self.get_logger().info(f"Pause service response: success={result.success}, msg={result.message}")
            else:
                self.get_logger().info("Pause service called (async, not waiting for response)")
        except Exception as e:
            self.get_logger().warn(f"Error calling pause service: {e}")
    
    async def _resume_real_robot_joint_publishing(self) -> None:
        """Resume real robot's joint state publishing after simulation."""
        self.get_logger().info("Checking if resume service is ready...")
        
        if not self.resume_joint_pub_client.service_is_ready():
            self.get_logger().warn("Resume joint publishing service not available")
            return
        
        try:
            self.get_logger().info("Calling resume service...")
            request = Trigger.Request()
            future = self.resume_joint_pub_client.call_async(request)
            await asyncio.sleep(0.2)  # Give service time to process
            if future.done():
                result = future.result()
                self.get_logger().info(f"Resume service response: success={result.success}, msg={result.message}")
            else:
                self.get_logger().info("Resume service called (async, not waiting for response)")
        except Exception as e:
            self.get_logger().warn(f"Failed to resume joint publishing: {e}")

    def _add_collision_objects_to_planning_scene(self) -> None:
        """Add environment collision objects (table, face) to MoveIt planning scene.
        
        From valid8_environment.urdf:
        - Table link at world [0, 0, 1.0], with collision box at offset [0, 0, -0.5]
        - Table collision box size: [2.1, 1.1, 1.04] (extends from world z=-0.02 to z=1.02)
        - Face at world [0.1742, 0, 1.6] with 90° yaw
        - Robot base at world [-0.6758, 0, 1.03] with -90° yaw
        
        So in base_link frame:
        - Table center (of collision box) is approximately at [0.6758, 0, -0.53]
        - Face is at approximately [0, 0.85, 0.57]
        """
        if not self._has_moveit or self._collision_objects_added:
            return
        
        try:
            from moveit_msgs.msg import PlanningScene, CollisionObject
            from shape_msgs.msg import SolidPrimitive
            from geometry_msgs.msg import Pose as GeometryPose
            
            planning_scene = PlanningScene()
            planning_scene.is_diff = True
            
            # Get transforms for table and face to verify positions
            try:
                table_tf = self.tf_buffer.lookup_transform(
                    'base_link', 'table_link', rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=2.0)
                )
                face_tf = self.tf_buffer.lookup_transform(
                    'base_link', 'face_link', rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=2.0)
                )
                
                self.get_logger().info(f"Table in base_link: ({table_tf.transform.translation.x:.3f}, "
                                       f"{table_tf.transform.translation.y:.3f}, {table_tf.transform.translation.z:.3f})")
                self.get_logger().info(f"Face in base_link: ({face_tf.transform.translation.x:.3f}, "
                                       f"{face_tf.transform.translation.y:.3f}, {face_tf.transform.translation.z:.3f})")
            except Exception as e:
                self.get_logger().warn(f"Could not get transforms for collision objects: {e}")
                self.get_logger().warn("Proceeding with hardcoded positions based on URDF")
                # Use fallback positions based on URDF calculations
                table_tf = None
                face_tf = None
            
            # Add table as collision object
            # The table collision box in URDF is size [2.1, 1.1, 1.04] centered at [0, 0, -0.5] relative to table_link
            # table_link is at world [0, 0, 1.0], so collision box center is at world [0, 0, 0.5]
            # Robot base is at world [-0.6758, 0, 1.03]
            table_obj = CollisionObject()
            table_obj.header.frame_id = 'base_link'
            table_obj.header.stamp = self.get_clock().now().to_msg()
            table_obj.id = 'optical_table'
            table_obj.operation = CollisionObject.ADD
            
            # Table collision box (matching URDF exactly)
            table_primitive = SolidPrimitive()
            table_primitive.type = SolidPrimitive.BOX
            table_primitive.dimensions = [2.1, 1.1, 1.04]  # From URDF
            
            table_pose = GeometryPose()
            if table_tf is not None:
                # Use TF-derived position, but offset by -0.5m in Z (collision box offset from table_link)
                # Also need to rotate the offset by the table orientation
                table_pose.position.x = table_tf.transform.translation.x
                table_pose.position.y = table_tf.transform.translation.y
                table_pose.position.z = table_tf.transform.translation.z - 0.5  # Offset for collision box center
                table_pose.orientation = table_tf.transform.rotation
            else:
                # Hardcoded fallback: table collision center in base_link frame
                # World: table_link at [0,0,1], collision at [0,0,0.5]
                # Base at world [-0.6758, 0, 1.03] with -90° yaw
                # In base_link: [0 - (-0.6758), 0 - 0, 0.5 - 1.03] = [0.6758, 0, -0.53]
                # But base has -90° yaw, so X,Y swap: [0, 0.6758, -0.53]
                table_pose.position.x = 0.0
                table_pose.position.y = 0.6758
                table_pose.position.z = -0.53
                table_pose.orientation.w = 1.0
            
            table_obj.primitives.append(table_primitive)
            table_obj.primitive_poses.append(table_pose)
            planning_scene.world.collision_objects.append(table_obj)
            
            # Add face as collision object (sphere approximation for head)
            face_obj = CollisionObject()
            face_obj.header.frame_id = 'base_link'
            face_obj.header.stamp = self.get_clock().now().to_msg()
            face_obj.id = 'face_fixture'
            face_obj.operation = CollisionObject.ADD
            
            face_primitive = SolidPrimitive()
            face_primitive.type = SolidPrimitive.SPHERE
            face_primitive.dimensions = [0.15]  # Head radius ~15cm (slightly larger for safety)
            
            face_pose = GeometryPose()
            if face_tf is not None:
                face_pose.position.x = face_tf.transform.translation.x
                face_pose.position.y = face_tf.transform.translation.y
                face_pose.position.z = face_tf.transform.translation.z
                face_pose.orientation = face_tf.transform.rotation
            else:
                # Hardcoded fallback
                face_pose.position.x = 0.0
                face_pose.position.y = 0.85
                face_pose.position.z = 0.57
                face_pose.orientation.w = 1.0
            
            face_obj.primitives.append(face_primitive)
            face_obj.primitive_poses.append(face_pose)
            planning_scene.world.collision_objects.append(face_obj)
            
            # Publish the planning scene
            self.planning_scene_pub.publish(planning_scene)
            self._collision_objects_added = True
            self.get_logger().info(f"Added collision objects to planning scene:")
            self.get_logger().info(f"  - optical_table: box {table_primitive.dimensions} at "
                                   f"({table_pose.position.x:.3f}, {table_pose.position.y:.3f}, {table_pose.position.z:.3f})")
            self.get_logger().info(f"  - face_fixture: sphere r={face_primitive.dimensions[0]:.2f} at "
                                   f"({face_pose.position.x:.3f}, {face_pose.position.y:.3f}, {face_pose.position.z:.3f})")
            
        except Exception as e:
            self.get_logger().error(f"Failed to add collision objects: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _compute_demo_pose_joints(
        self,
        horiz_mm: float,
        vert_mm: float,
        dist_mm: float,
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
    ) -> Optional[List[float]]:
        """
        Compute joint configuration for demonstration mode.
        
        This generates visually dramatic joint configurations that show
        the robot moving through different orientations. This is used when
        IK solving is disabled (default) for visualization purposes.
        
        For real pose accuracy, enable use_ik=True to use actual inverse
        kinematics solving via motion_planner_node.
        """
        # Convert mm to meters for internal calculations
        dist_m = dist_mm / 1000.0
        horiz_m = horiz_mm / 1000.0
        vert_m = vert_mm / 1000.0
        
        # Generate visually dramatic joint configurations
        # These scaling factors produce ~30-45 degree movements for typical proto-sim params
        
        # Base rotation (j0): yaw creates visible rotation around base
        # Full yaw range maps to ±45 degrees base rotation
        j0 = math.radians(yaw_deg) * 0.5 + horiz_m * 2.0
        
        # Shoulder (j1): pitch tilts the robot arm up/down  
        # Pitch -30 to +30 creates ~30 degree shoulder movement
        j1 = -math.pi/2 + math.radians(pitch_deg) * 0.5 + vert_m * 1.5
        
        # Elbow (j2): distance affects how extended the arm is
        # 250-450mm range creates visible elbow bend
        j2 = 0.5 + (0.35 - dist_m) * 3.0
        
        # Wrist 1 (j3): compensate for shoulder to keep tool oriented
        j3 = -math.pi/2 - math.radians(pitch_deg) * 0.3
        
        # Wrist 2 (j4): roll orientation of tool
        j4 = math.radians(roll_deg + 90) * 0.5
        
        # Wrist 3 (j5): fine roll adjustment
        j5 = math.radians(yaw_deg) * 0.3
        
        # Clamp to safe UR5e joint limits
        joints = [
            float(np.clip(j0, -math.pi, math.pi)),
            float(np.clip(j1, -math.pi, 0)),
            float(np.clip(j2, -math.pi, math.pi)),
            float(np.clip(j3, -math.pi, 0)),
            float(np.clip(j4, -math.pi, math.pi)),
            float(np.clip(j5, -math.pi, math.pi)),
        ]
        
        return joints
    
    def _solve_ik_for_pose(
        self,
        position: Tuple[float, float, float],
        orientation_quat_xyzw: Tuple[float, float, float, float],
        seed_joints: Optional[List[float]] = None,
    ) -> Optional[List[float]]:
        """
        Solve inverse kinematics for a target Cartesian pose.
        
        This method attempts to use MoveIt 2's compute_ik service.
        Falls back to analytical IK if MoveIt service is unavailable.
        
        Args:
            position: Target (x, y, z) in base_link frame (meters)
            orientation_quat_xyzw: Target orientation quaternion (x, y, z, w)
            seed_joints: Optional seed configuration for IK solver (use current joints for continuity)
            
        Returns:
            Joint angles [j0, j1, j2, j3, j4, j5] or None if IK failed
        """
        # Try MoveIt 2 IK first
        if self._has_moveit and self.compute_ik_client:
            try:
                self.get_logger().info(f"Trying MoveIt IK for pos={position}")
                result = self._solve_ik_moveit_sync(position, orientation_quat_xyzw, seed_joints=seed_joints)
                if result is not None:
                    self.get_logger().info(f"MoveIt IK success: {[f'{j:.3f}' for j in result]}")
                    return result
                else:
                    self.get_logger().info("MoveIt IK failed, trying analytical fallback")
            except Exception as e:
                self.get_logger().warn(f"MoveIt IK error: {e}, using analytical fallback")
        else:
            self.get_logger().info(f"MoveIt not available (has_moveit={self._has_moveit}), using analytical")
        
        # Fallback: use simple analytical IK approximation
        self.get_logger().info("Using analytical IK fallback")
        return self._solve_ik_analytical(position, orientation_quat_xyzw)
    
    def _solve_ik_moveit_sync(
        self,
        position: Tuple[float, float, float],
        orientation_quat_xyzw: Tuple[float, float, float, float],
        seed_joints: Optional[List[float]] = None,
    ) -> Optional[List[float]]:
        """
        Call MoveIt 2's compute_ik service synchronously.
        
        Uses seed_joints as the starting configuration for IK search.
        This helps find solutions closer to the current configuration,
        making motion planning more reliable.
        
        This is a blocking call that waits for the IK solution.
        """
        if not self.compute_ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("MoveIt compute_ik service not available")
            return None
        
        try:
            from moveit_msgs.srv import GetPositionIK
            from moveit_msgs.msg import PositionIKRequest, RobotState, MoveItErrorCodes
            from geometry_msgs.msg import PoseStamped
            from sensor_msgs.msg import JointState
            
            # Error code names for better debugging
            error_names = {
                1: "SUCCESS",
                -1: "FAILURE",
                -2: "PLANNING_FAILED",
                -10: "START_STATE_IN_COLLISION",
                -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
                -12: "GOAL_IN_COLLISION",
                -26: "START_STATE_INVALID",
                -31: "NO_IK_SOLUTION",
            }
            
            # Build IK request
            request = GetPositionIK.Request()
            request.ik_request.group_name = "ur_manipulator"
            request.ik_request.avoid_collisions = True  # Enable collision checking in IK
            
            # CRITICAL: Specify which link the target pose is for
            # The client generates poses for tool_tip_link (where the camera/phone is)
            # Without this, MoveIt defaults to tool0 which is offset from the actual tool tip
            request.ik_request.ik_link_name = "tool_tip_link"
            
            # Set the target pose
            target_pose = PoseStamped()
            target_pose.header.frame_id = "base_link"
            target_pose.header.stamp = self.get_clock().now().to_msg()
            target_pose.pose.position.x = position[0]
            target_pose.pose.position.y = position[1]
            target_pose.pose.position.z = position[2]
            target_pose.pose.orientation.x = orientation_quat_xyzw[0]
            target_pose.pose.orientation.y = orientation_quat_xyzw[1]
            target_pose.pose.orientation.z = orientation_quat_xyzw[2]
            target_pose.pose.orientation.w = orientation_quat_xyzw[3]
            request.ik_request.pose_stamped = target_pose
            
            # Use current joint configuration as seed for IK solver
            # This produces solutions closer to current pose, making motion planning easier
            robot_state = RobotState()
            robot_state.joint_state.name = self.sim_joint_names
            # Use provided seed_joints, fall back to current sim state, then home position
            if seed_joints is not None:
                robot_state.joint_state.position = list(seed_joints)
            elif hasattr(self, 'sim_joint_positions') and self.sim_joint_positions:
                robot_state.joint_state.position = list(self.sim_joint_positions)
            else:
                # Fallback to home position if nothing else available
                home_position = [0.0, -math.pi/2, 0.0, -math.pi/2, 0.0, 0.0]
                robot_state.joint_state.position = home_position
            request.ik_request.robot_state = robot_state
            
            # Call the service
            future = self.compute_ik_client.call_async(request)
            
            # Wait for result with timeout
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            
            if future.done():
                response = future.result()
                if response.error_code.val == MoveItErrorCodes.SUCCESS:
                    # Extract joint positions from solution
                    solution = response.solution.joint_state
                    joint_positions = list(solution.position)
                    if len(joint_positions) >= 6:
                        joints = joint_positions[:6]
                        
                        # Validate joints are within UR robot limits
                        # All UR joints have limits of approximately [-2π, 2π]
                        # We use slightly larger tolerance to allow the full range
                        joint_limit = 2.0 * math.pi + 0.1
                        
                        joints_valid = True
                        for i, j in enumerate(joints):
                            if j < -joint_limit or j > joint_limit:
                                self.get_logger().warn(
                                    f"IK solution joint {i} = {j:.3f} outside limits [-{joint_limit:.2f}, {joint_limit:.2f}]"
                                )
                                joints_valid = False
                                break
                        
                        if joints_valid:
                            # Normalize to [-π, π] to avoid huge sweeps that
                            # collide or exceed velocity limits.
                            joints = self._normalize_joint_angles(joints, seed_joints)
                            self.get_logger().info(f"MoveIt IK success: {[f'{j:.3f}' for j in joints]}")
                            return joints
                        else:
                            self.get_logger().warn("IK solution rejected: joints outside limits")
                else:
                    error_name = error_names.get(response.error_code.val, f"UNKNOWN({response.error_code.val})")
                    self.get_logger().debug(f"MoveIt IK failed: {error_name} for pos={position}")
            
            return None
            
        except Exception as e:
            self.get_logger().warn(f"MoveIt IK call failed: {e}")
            return None

    @staticmethod
    def _normalize_joint_angles(
        joints: List[float],
        seed: Optional[List[float]] = None,
    ) -> List[float]:
        """Normalize joint angles to the equivalent value closest to the seed.
        
        UR joints are continuous and IK may return values like 4.53 rad
        which is equivalent to 4.53 - 2π ≈ -1.75 rad.  If the seed
        (current robot state) is -1.57 rad, the -1.75 solution is much
        closer and avoids a huge sweep through collision space.
        
        For each joint we pick the k*2π offset that minimises |j - seed_j|.
        """
        TWO_PI = 2.0 * math.pi
        result = list(joints)
        if seed is None:
            # Just wrap to [-π, π]
            for i in range(len(result)):
                while result[i] > math.pi:
                    result[i] -= TWO_PI
                while result[i] < -math.pi:
                    result[i] += TWO_PI
            return result
        
        for i in range(min(len(result), len(seed))):
            # Pick the 2π-offset of result[i] closest to seed[i]
            diff = result[i] - seed[i]
            # Number of full rotations to remove
            k = round(diff / TWO_PI)
            result[i] -= k * TWO_PI
        return result

    def _solve_ik_analytical(
        self,
        position: Tuple[float, float, float],
        orientation_quat_xyzw: Tuple[float, float, float, float],
    ) -> Optional[List[float]]:
        """
        Solve IK using analytical approach for UR5e.
        
        This is a simplified 6-DOF IK solver as a fallback when MoveIt is unavailable.
        For production, MoveIt 2 should always be used.
        
        The UR5e URDF has these frame conventions:
        - base_link: Robot mounting frame
        - base_link_inertia: Rotated 180° (π rad) from base_link around Z
        - shoulder_pan_joint rotates around base_link_inertia's Z axis
        
        When computing j0 (shoulder pan), we need to account for this 180° offset.
        A target at (0, +Y, Z) in base_link frame requires j0 = -π/2 in shoulder_pan_joint
        because the joint frame is rotated 180° from base_link.
        """
        try:
            x, y, z = position
            qx, qy, qz, qw = orientation_quat_xyzw
            
            # UR5e DH parameters (from URDF)
            d1 = 0.1625   # base to shoulder offset
            a2 = -0.425   # upper arm length
            a3 = -0.3922  # forearm length
            d4 = 0.1333   # wrist 1 offset
            d5 = 0.0997   # wrist 2 offset
            d6 = 0.0996   # wrist 3 to flange
            
            # Tool offset from flange (iphone tool)
            tool_offset = 0.08  # 80mm in -X direction of tool0
            
            # Compute wrist center position (back off from TCP along tool Z-axis)
            # The tool Z-axis direction comes from the quaternion
            # For quaternion (x,y,z,w), the Z-axis of the rotated frame is:
            # z_axis = [2*(xz+wy), 2*(yz-wx), 1-2*(xx+yy)]
            tool_z_x = 2*(qx*qz + qw*qy)
            tool_z_y = 2*(qy*qz - qw*qx)
            tool_z_z = 1 - 2*(qx*qx + qy*qy)
            
            # Wrist center is TCP position minus tool length along tool Z
            wc_x = x - (d6 + tool_offset) * tool_z_x
            wc_y = y - (d6 + tool_offset) * tool_z_y
            wc_z = z - (d6 + tool_offset) * tool_z_z
            
            # J0 (shoulder pan): angle to point arm towards wrist center
            # The shoulder_pan_joint frame (base_link_inertia) is rotated 180° from base_link
            # So we compute the angle in base_link frame and add π
            j0_base = math.atan2(wc_y, wc_x)
            j0 = j0_base + math.pi
            # Normalize to [-π, π]
            while j0 > math.pi:
                j0 -= 2 * math.pi
            while j0 < -math.pi:
                j0 += 2 * math.pi
            
            # Distance from base to wrist center in XY plane
            r_wc = math.sqrt(wc_x**2 + wc_y**2)
            
            # Account for wrist offset (d4) perpendicular to arm plane
            # For elbow-down configuration
            r_arm = r_wc  # Simplified - full IK needs proper handling
            
            # Height of wrist center relative to shoulder
            z_arm = wc_z - d1
            
            # Arm lengths for 2-link planar IK
            L1 = abs(a2)  # Upper arm
            L2 = abs(a3)  # Forearm (simplified, ignoring d4 for now)
            
            # Distance from shoulder to wrist center in arm plane
            d_arm = math.sqrt(r_arm**2 + z_arm**2)
            
            # Check reachability
            if d_arm > L1 + L2:
                self.get_logger().debug(f"Target out of reach: d={d_arm:.3f} > max={L1+L2:.3f}")
                return None
            if d_arm < abs(L1 - L2) + 0.01:  # Small margin
                self.get_logger().debug(f"Target too close: d={d_arm:.3f} < min={abs(L1-L2):.3f}")
                return None
            
            # J2 (elbow): cosine rule for elbow angle
            cos_j2 = (d_arm**2 - L1**2 - L2**2) / (2 * L1 * L2)
            cos_j2 = np.clip(cos_j2, -1.0, 1.0)
            j2 = -math.acos(cos_j2)  # Elbow-down configuration (negative)
            
            # J1 (shoulder lift): angle to reach wrist center
            # Two angles: angle to target + angle from elbow geometry
            alpha = math.atan2(z_arm, r_arm)  # Angle to wrist center
            # Angle from triangle formed by L1, L2, d_arm
            cos_beta = (d_arm**2 + L1**2 - L2**2) / (2 * d_arm * L1)
            cos_beta = np.clip(cos_beta, -1.0, 1.0)
            beta = math.acos(cos_beta)
            # J1 in UR convention (0 is horizontal forward, negative is up)
            j1 = -(alpha + beta)  # Shoulder lift
            
            # Wrist joints (j3, j4, j5) from end-effector orientation
            # This is simplified - proper decomposition requires full FK
            
            # Extract RPY from quaternion for approximate wrist angles
            # Roll (X), Pitch (Y), Yaw (Z) in intrinsic XYZ order
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
            roll = math.atan2(sinr_cosp, cosr_cosp)
            
            sinp = 2 * (qw * qy - qz * qx)
            sinp = np.clip(sinp, -1.0, 1.0)
            pitch = math.asin(sinp)
            
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            # Approximate wrist decomposition
            # J3 compensates for arm pose to achieve desired pitch
            j3 = pitch - j1 - j2
            
            # J4 sets roll around wrist 2 axis
            j4 = roll - math.pi/2
            
            # J5 sets final orientation around wrist 3 axis
            j5 = yaw - j0_base
            
            # Normalize and clamp to UR5e joint limits
            def normalize_angle(a):
                while a > math.pi:
                    a -= 2 * math.pi
                while a < -math.pi:
                    a += 2 * math.pi
                return a
            
            j3 = normalize_angle(j3)
            j4 = normalize_angle(j4)
            j5 = normalize_angle(j5)
            
            # UR5e joint limits (from URDF)
            # All joints: [-2π, 2π] except some have tighter practical limits
            joints = [
                float(np.clip(j0, -2*math.pi, 2*math.pi)),
                float(np.clip(j1, -math.pi, 0)),           # Shoulder lift: [-π, 0]
                float(np.clip(j2, -math.pi, math.pi)),     # Elbow
                float(np.clip(j3, -2*math.pi, 2*math.pi)), # Wrist 1
                float(np.clip(j4, -2*math.pi, 2*math.pi)), # Wrist 2
                float(np.clip(j5, -2*math.pi, 2*math.pi)), # Wrist 3
            ]
            
            self.get_logger().debug(f"Analytical IK: pos=({x:.3f},{y:.3f},{z:.3f}) -> joints={[f'{j:.2f}' for j in joints]}")
            
            return joints
            
        except Exception as e:
            self.get_logger().warn(f"Analytical IK failed: {e}")
            import traceback
            self.get_logger().debug(traceback.format_exc())
            return None

    def _generate_proto_sim_joints(
        self,
        target_pos: np.ndarray,
        distances: List[float],
        horizontal_angles: List[float],
        vertical_angles: List[float],
    ) -> Dict[Tuple[float, int, int], List[float]]:
        """
        Generate joint configurations for proto-sim poses.
        
        For now, generates demonstration poses that sweep through
        the robot's workspace. In production, this would use IK
        to calculate actual poses pointing at the target.
        """
        pose_joints = {}
        
        # Home position
        home = [0.0, -math.pi/2, 0.0, -math.pi/2, 0.0, 0.0]
        
        # Generate varied poses based on spherical coordinates
        for dist in distances:
            for h_angle in horizontal_angles:
                for v_angle in vertical_angles:
                    # Map spherical coords to joint angles for demonstration
                    # This creates a sweeping motion that's visually interesting
                    
                    # Base rotation follows horizontal angle
                    j0 = math.radians(h_angle) * 0.5  # Scale down
                    
                    # Shoulder adjusts with distance and vertical
                    j1 = -math.pi/2 + math.radians(v_angle) * 0.02 + (dist - 0.4) * 0.5
                    
                    # Elbow adjusts with distance
                    j2 = (0.5 - dist) * 2.0
                    
                    # Wrist angles for orientation
                    j3 = -math.pi/2 - math.radians(v_angle) * 0.03
                    j4 = math.radians(h_angle) * 0.3
                    j5 = 0.0
                    
                    # Clamp to safe limits
                    joints = [
                        np.clip(j0, -math.pi, math.pi),
                        np.clip(j1, -math.pi, 0),
                        np.clip(j2, -math.pi, math.pi),
                        np.clip(j3, -math.pi, 0),
                        np.clip(j4, -math.pi, math.pi),
                        np.clip(j5, -math.pi, math.pi),
                    ]
                    
                    pose_joints[(dist, h_angle, v_angle)] = joints
        
        return pose_joints

    async def _execute_sim_movement(
        self, 
        target_joints: List[float],
        move_speed: float = 0.5,
        publish_rate: float = 50.0,
        require_collision_check: bool = True,
    ) -> bool:
        """
        Execute simulated movement by interpolating joint positions
        and publishing JointState messages.
        
        If require_collision_check=True and MoveIt planning fails (collision detected),
        the movement is REJECTED and False is returned. This ensures collision safety.
        
        Args:
            target_joints: Target joint positions [j0, j1, j2, j3, j4, j5]
            move_speed: Movement speed scale (0.1 to 1.0)
            publish_rate: Rate to publish joint states (Hz)
            require_collision_check: If True, REJECT movement if collision-free path cannot be found
            
        Returns:
            True if movement was executed successfully, False if rejected due to collision
        """
        start_joints = np.array(self.sim_joint_positions)
        end_joints = np.array(target_joints)
        
        # Plan with collision checking via MoveIt 2
        trajectory_data = None
        planning_failed = False
        if self._has_moveit and self.plan_motion_client:
            try:
                trajectory_data = self._plan_motion_moveit_sync(start_joints.tolist(), target_joints)
                if trajectory_data:
                    waypoints = trajectory_data.get('waypoints', [])
                    self.get_logger().info(f"MoveIt planned trajectory with {len(waypoints)} waypoints")
                else:
                    planning_failed = True
                    self.get_logger().warn(f"MoveIt planning failed - no collision-free path found")
            except Exception as e:
                planning_failed = True
                self.get_logger().warn(f"MoveIt planning failed: {e}")
        else:
            # MoveIt not available
            if require_collision_check:
                self.get_logger().error("Collision checking required but MoveIt is not available")
                return False
        
        # If collision checking is required and planning failed, REJECT the movement
        if require_collision_check and planning_failed:
            self.get_logger().warn("Movement REJECTED due to collision - no fallback to unchecked movement")
            return False
        
        if trajectory_data and len(trajectory_data.get('waypoints', [])) > 1:
            # Execute planned trajectory (collision-free path) with smooth interpolation
            await self._execute_trajectory(trajectory_data, publish_rate)
        
        # Ensure we end at target
        self.sim_joint_positions = target_joints
        self._publish_joint_state()
        
        self.get_logger().info(f"Sim movement complete, final joints: {[f'{j:.3f}' for j in target_joints]}")
        return True

    def _plan_motion_moveit_sync(
        self,
        start_joints: List[float],
        target_joints: List[float],
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> Optional[Dict[str, Any]]:
        """
        Plan motion using MoveIt 2's planning service.
        
        MoveIt applies TOTG (Time-Optimal Trajectory Generation) to produce
        a complete trajectory with positions, velocities, accelerations, and
        proper timing. The velocity_scale and acceleration_scale parameters
        control how fast the planned trajectory is (0.0-1.0).
        
        Returns a dict with:
        - 'waypoints': List of joint position waypoints (for sim visualization)
        - 'time_from_start': List of time_from_start for each waypoint (seconds)
        - 'joint_trajectory': Raw JointTrajectory ROS msg (for real robot)
        """
        if not self.plan_motion_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("MoveIt motion planning service not available")
            return None
        
        try:
            from moveit_msgs.srv import GetMotionPlan
            from moveit_msgs.msg import RobotState, Constraints, JointConstraint, MoveItErrorCodes, MotionPlanRequest
            from sensor_msgs.msg import JointState as JointStateMsg
            
            # Build motion plan request
            request = GetMotionPlan.Request()
            request.motion_plan_request.group_name = "ur_manipulator"
            # Increased planning attempts and time for better reliability
            request.motion_plan_request.num_planning_attempts = 20
            request.motion_plan_request.allowed_planning_time = 10.0
            request.motion_plan_request.max_velocity_scaling_factor = velocity_scale
            request.motion_plan_request.max_acceleration_scaling_factor = acceleration_scale
            
            # Set start state from current joint positions
            start_state = RobotState()
            start_state.joint_state.header.stamp = self.get_clock().now().to_msg()
            start_state.joint_state.name = self.sim_joint_names
            start_state.joint_state.position = start_joints
            start_state.is_diff = False  # Full state, not a diff
            request.motion_plan_request.start_state = start_state
            
            # Set workspace bounds to help OMPL
            from moveit_msgs.msg import WorkspaceParameters
            request.motion_plan_request.workspace_parameters.header.frame_id = "base_link"
            request.motion_plan_request.workspace_parameters.min_corner.x = -2.0
            request.motion_plan_request.workspace_parameters.min_corner.y = -2.0
            request.motion_plan_request.workspace_parameters.min_corner.z = -0.5
            request.motion_plan_request.workspace_parameters.max_corner.x = 2.0
            request.motion_plan_request.workspace_parameters.max_corner.y = 2.0
            request.motion_plan_request.workspace_parameters.max_corner.z = 3.0
            
            # Set goal constraints
            goal_constraints = Constraints()
            for i, (name, pos) in enumerate(zip(self.sim_joint_names, target_joints)):
                jc = JointConstraint()
                jc.joint_name = name
                jc.position = pos
                jc.tolerance_above = 0.01
                jc.tolerance_below = 0.01
                jc.weight = 1.0
                goal_constraints.joint_constraints.append(jc)
            request.motion_plan_request.goal_constraints.append(goal_constraints)
            
            # Call the service - increase timeout for more planning attempts
            future = self.plan_motion_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
            
            if future.done():
                response = future.result()
                if response.motion_plan_response.error_code.val == MoveItErrorCodes.SUCCESS:
                    # Extract waypoints and timing from trajectory
                    trajectory = response.motion_plan_response.trajectory.joint_trajectory
                    waypoints = []
                    time_from_start = []
                    for point in trajectory.points:
                        waypoints.append(list(point.positions))
                        # Convert ROS duration to seconds
                        t = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
                        time_from_start.append(t)
                    
                    # Log trajectory details including peak velocities for debugging
                    has_velocities = any(len(p.velocities) > 0 for p in trajectory.points)
                    has_accelerations = any(len(p.accelerations) > 0 for p in trajectory.points)
                    
                    # Calculate peak velocities per joint for velocity-limit debugging
                    peak_vel_str = ""
                    if has_velocities:
                        num_joints = len(trajectory.joint_names)
                        peak_vels = [0.0] * num_joints
                        for point in trajectory.points:
                            for j in range(min(num_joints, len(point.velocities))):
                                peak_vels[j] = max(peak_vels[j], abs(point.velocities[j]))
                        peak_vel_str = f", peak_vel=[{', '.join(f'{v:.2f}' for v in peak_vels)}] rad/s"
                    
                    self.get_logger().info(
                        f"MoveIt trajectory: {len(trajectory.points)} points, "
                        f"duration={time_from_start[-1]:.2f}s, "
                        f"vel_scale={velocity_scale}, accel_scale={acceleration_scale}"
                        f"{peak_vel_str}"
                    )
                    
                    return {
                        'waypoints': waypoints,
                        'time_from_start': time_from_start,
                        # Keep the raw JointTrajectory for the real robot controller
                        'joint_trajectory': trajectory,
                        # Keep the full RobotTrajectory for MoveIt's ExecuteTrajectory action
                        'robot_trajectory': response.motion_plan_response.trajectory,
                    }
                else:
                    self.get_logger().debug(f"MoveIt planning error: {response.motion_plan_response.error_code.val}")
            
            return None
            
        except Exception as e:
            self.get_logger().warn(f"MoveIt planning call failed: {e}")
            return None

    async def _execute_trajectory(
        self,
        trajectory_data: Dict[str, Any],
        publish_rate: float = 50.0,
    ) -> None:
        """
        Execute a trajectory by publishing joint states at the specified rate.
        
        Uses MoveIt's computed trajectory timing for smooth, consistent motion.
        Interpolates between waypoints for high-frequency publishing to Foxglove.
        """
        waypoints = trajectory_data.get('waypoints', [])
        time_from_start = trajectory_data.get('time_from_start', [])
        
        if not waypoints:
            self.get_logger().warn("No waypoints in trajectory data")
            return
        
        self.get_logger().info(f"Trajectory execution: {len(waypoints)} waypoints, timing: {time_from_start}")
        
        # If no timing info or timing is near-zero, compute timing from joint velocities
        # MoveIt's GetMotionPlan service doesn't always apply time parametrization
        needs_timing = (
            not time_from_start or 
            len(time_from_start) != len(waypoints) or
            (time_from_start[-1] < 0.1 and len(waypoints) > 1)  # Near-zero timing
        )
        
        if needs_timing:
            # Compute timing based on max joint velocity (conservative: 1.0 rad/s)
            max_joint_velocity = 1.0  # rad/s - conservative for smooth motion
            time_from_start = [0.0]
            for i in range(1, len(waypoints)):
                prev_wp = np.array(waypoints[i - 1])
                curr_wp = np.array(waypoints[i])
                max_joint_diff = np.max(np.abs(curr_wp - prev_wp))
                segment_duration = max(0.02, max_joint_diff / max_joint_velocity)  # At least 20ms per segment
                time_from_start.append(time_from_start[-1] + segment_duration)
            self.get_logger().info(f"Computed timing: {[f'{t:.3f}' for t in time_from_start]}")
        
        # Total trajectory duration from MoveIt or computed
        total_duration = time_from_start[-1] if time_from_start else 2.0
        total_duration = max(0.5, total_duration)  # Minimum 0.5s duration
        
        # Publish at a fixed rate (e.g., 50Hz) for smooth visualization
        dt = 1.0 / publish_rate
        num_samples = max(1, int(total_duration * publish_rate))
        
        self.get_logger().info(f"Trajectory: duration={total_duration:.2f}s, samples={num_samples}, dt={dt:.4f}s")
        
        start_time = asyncio.get_event_loop().time()
        
        for sample_idx in range(num_samples + 1):  # +1 to include final position
            if self._proto_sim_stop_requested:
                break
            
            # Current time in trajectory
            t = min(sample_idx * dt, total_duration)
            
            # Find which segment we're in using binary search approach
            # We want segment_idx such that time_from_start[segment_idx] <= t < time_from_start[segment_idx + 1]
            segment_idx = 0
            for i in range(len(time_from_start) - 1):
                if time_from_start[i + 1] <= t:
                    segment_idx = i + 1
                else:
                    break
            
            # Ensure segment_idx is within valid range for interpolation
            segment_idx = min(segment_idx, len(waypoints) - 2)
            segment_idx = max(segment_idx, 0)
            
            # Interpolate within segment
            if segment_idx < len(waypoints) - 1:
                t0 = time_from_start[segment_idx]
                t1 = time_from_start[segment_idx + 1]
                segment_duration = t1 - t0
                if segment_duration > 0:
                    alpha = min(1.0, max(0.0, (t - t0) / segment_duration))
                else:
                    alpha = 1.0
                
                wp0 = np.array(waypoints[segment_idx])
                wp1 = np.array(waypoints[segment_idx + 1])
                interpolated = wp0 + alpha * (wp1 - wp0)
                self.sim_joint_positions = interpolated.tolist()
            else:
                self.sim_joint_positions = waypoints[-1]
            
            self._publish_joint_state()
            
            # Wait to maintain publish rate
            elapsed = asyncio.get_event_loop().time() - start_time
            target_time = (sample_idx + 1) * dt
            sleep_time = target_time - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def _publish_joint_state(self) -> None:
        """Publish current simulated joint state."""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ''
        msg.name = self.sim_joint_names
        msg.position = self.sim_joint_positions
        msg.velocity = [0.0] * 6
        msg.effort = [0.0] * 6
        
        self.joint_state_pub.publish(msg)
        # Log occasionally to avoid spam
        if not hasattr(self, '_pub_count'):
            self._pub_count = 0
        self._pub_count += 1
        if self._pub_count == 1 or self._pub_count % 50 == 0:
            positions_str = ', '.join([f'{j:.2f}' for j in self.sim_joint_positions])
            self.get_logger().info(f'Publishing joint state #{self._pub_count}: [{positions_str}]')

    def _real_robot_joint_state_callback(self, msg: JointState) -> None:
        """
        Callback for real robot joint states from joint_state_broadcaster.
        """
        if len(msg.position) < 6:
            return
        
        joint_positions = {}
        for name, pos in zip(msg.name, msg.position):
            joint_positions[name] = pos
        
        ordered_positions = []
        for joint_name in self.sim_joint_names:
            if joint_name in joint_positions:
                ordered_positions.append(joint_positions[joint_name])
            else:
                return
        
        self.real_robot_joint_positions = ordered_positions

    def _robot_program_running_callback(self, msg) -> None:
        """Callback for robot_program_running topic."""
        self._robot_program_running = msg.data

    async def _ensure_robot_ready(self, timeout: float = 15.0) -> bool:
        """
        Ensure the UR robot is ready to execute trajectories.
        
        Checks that robot_program_running is True (external control program
        is active on the teach pendant). If not, attempts to resend the
        robot program and waits for it to become active.
        
        Returns:
            True if robot is ready, False if timed out
        """
        if self._robot_program_running:
            return True
        
        self.get_logger().warn("Robot program not running - attempting to resend...")
        
        # Try to resend the robot program
        if self._resend_program_client.wait_for_service(timeout_sec=2.0):
            try:
                future = self._resend_program_client.call_async(Trigger.Request())
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                if future.done() and future.result().success:
                    self.get_logger().info("Robot program resend requested successfully")
                else:
                    self.get_logger().warn("Robot program resend request returned non-success")
            except Exception as e:
                self.get_logger().error(f"Failed to resend robot program: {e}")
        else:
            self.get_logger().warn("Resend robot program service not available")
        
        # Wait for robot_program_running to become True
        start_time = asyncio.get_event_loop().time()
        while not self._robot_program_running:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                self.get_logger().error(
                    f"Timeout ({timeout}s) waiting for robot_program_running to become True"
                )
                return False
            await asyncio.sleep(0.5)
            # Spin to process callbacks
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().info("Robot program is running - ready for trajectory execution")
        # Give a bit more time for controller_stopper to activate controller
        await asyncio.sleep(1.0)
        return True

    async def rpc_stop_proto_sim(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Stop running protocol simulation."""
        if self._proto_sim_running:
            self._proto_sim_stop_requested = True
            return {'success': True, 'message': 'Stop requested'}
        return {'success': False, 'message': 'No protocol simulation running'}

    async def rpc_check_collision(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check collision for given joint configuration."""
        joint_positions = params.get('joint_positions', [0.0] * 6)
        
        # TODO: Use collision_checker.py to check collision
        # For now, return no collision
        return {
            'success': True,
            'in_collision': False,
            'colliding_pairs': [],
            'min_distance': 0.1,
        }

    async def rpc_plan_motion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Plan motion from current to target configuration."""
        start_joints = params.get('start_joints', [0.0] * 6)
        target_joints = params.get('target_joints')
        target_pose = params.get('target_pose')
        motion_type = params.get('motion_type', 'joint')
        velocity_scale = params.get('velocity_scale', 0.5)
        
        # TODO: Use motion_planner.py to generate trajectory
        # For now, return success with empty trajectory
        return {
            'success': True,
            'trajectory': {
                'points': [],
                'duration': 1.0,
            },
            'collision_free': True,
            'planning_time': 0.05,
        }

    async def send_error(
        self,
        client: ConnectedClient,
        request_id: Optional[str],
        message: str
    ):
        """Send error response to client."""
        error = {
            'type': 'error',
            'request_id': request_id,
            'message': message,
            'timestamp': time.time(),
        }
        await client.websocket.send(json.dumps(error))

    async def _execute_real_robot_trajectory(
        self,
        trajectory_data: Dict[str, Any],
        velocity_scale: float = 0.5,
        wait_for_completion: bool = True,
    ) -> Tuple[bool, float]:
        """
        Execute a trajectory on the real robot via MoveIt's ExecuteTrajectory action.
        
        This uses MoveIt's execution pipeline which properly manages the
        scaled_joint_trajectory_controller, validates start state, and handles
        the full trajectory lifecycle. MoveIt sends the complete trajectory
        (positions, velocities, accelerations, timing from TOTG) to the controller.
        
        Falls back to FollowJointTrajectory if ExecuteTrajectory is not available.
        
        Args:
            trajectory_data: Dictionary with 'robot_trajectory' (full MoveIt RobotTrajectory)
            velocity_scale: Not used (scaling is done in MoveIt planning request)
            wait_for_completion: If True, wait for trajectory to complete
            
        Returns:
            Tuple of (success: bool, duration: float)
        """
        # Get the full RobotTrajectory from MoveIt planning
        robot_trajectory = trajectory_data.get('robot_trajectory')
        raw_trajectory = trajectory_data.get('joint_trajectory')
        
        if robot_trajectory is None and raw_trajectory is None:
            self.get_logger().error("No trajectory data available for real robot execution")
            return False, 0.0
        
        # Compute total duration
        jt = raw_trajectory or robot_trajectory.joint_trajectory
        if len(jt.points) < 2:
            self.get_logger().warn(f"Trajectory too short: {len(jt.points)} points")
            return False, 0.0
        
        last_point = jt.points[-1]
        total_duration = last_point.time_from_start.sec + last_point.time_from_start.nanosec * 1e-9
        
        self.get_logger().info(
            f"Executing trajectory on real robot: "
            f"{len(jt.points)} points, duration={total_duration:.2f}s"
        )
        
        # CRITICAL PRE-FLIGHT CHECK: Ensure robot program is running
        # Without the external control program active on the teach pendant,
        # the controller cannot move the robot and will immediately abort.
        robot_ready = await self._ensure_robot_ready(timeout=15.0)
        if not robot_ready:
            self.get_logger().error("Robot not ready - cannot execute trajectory")
            return False, 0.0
        
        # === PRIMARY: Use MoveIt's ExecuteTrajectory action ===
        # This is the standard MoveIt2 way to execute planned trajectories.
        # MoveIt handles: start state validation, controller selection, trajectory forwarding.
        if self.execute_trajectory_client and robot_trajectory:
            if not self.execute_trajectory_client.wait_for_server(timeout_sec=2.0):
                self.get_logger().warn("MoveIt ExecuteTrajectory not available, falling back")
            else:
                try:
                    goal = ExecuteTrajectoryAction.Goal()
                    goal.trajectory = robot_trajectory
                    
                    self.get_logger().info("Sending trajectory via MoveIt ExecuteTrajectory...")
                    
                    send_goal_future = self.execute_trajectory_client.send_goal_async(goal)
                    
                    # Wait for goal acceptance
                    try:
                        await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: rclpy.spin_until_future_complete(
                                    self, send_goal_future, timeout_sec=5.0
                                )
                            ),
                            timeout=6.0
                        )
                    except asyncio.TimeoutError:
                        self.get_logger().error("Timeout waiting for ExecuteTrajectory goal acceptance")
                        return False, 0.0
                    
                    if send_goal_future.done():
                        goal_handle = send_goal_future.result()
                        if goal_handle.accepted:
                            self.get_logger().info(
                                f"MoveIt ExecuteTrajectory accepted (duration={total_duration:.2f}s)"
                            )
                            
                            if wait_for_completion:
                                # Wait for result from MoveIt (proper completion monitoring)
                                result_future = goal_handle.get_result_async()
                                try:
                                    await asyncio.wait_for(
                                        asyncio.get_event_loop().run_in_executor(
                                            None,
                                            lambda: rclpy.spin_until_future_complete(
                                                self, result_future,
                                                timeout_sec=total_duration + 10.0
                                            )
                                        ),
                                        timeout=total_duration + 15.0
                                    )
                                    
                                    if result_future.done():
                                        result = result_future.result()
                                        error_code = result.result.error_code.val
                                        if error_code == 1:  # MoveItErrorCodes.SUCCESS
                                            self.get_logger().info("MoveIt trajectory execution succeeded")
                                            return True, total_duration
                                        else:
                                            self.get_logger().error(
                                                f"MoveIt trajectory execution failed with error code: {error_code}"
                                            )
                                            return False, 0.0
                                except asyncio.TimeoutError:
                                    self.get_logger().error("Timeout waiting for trajectory execution")
                                    return False, 0.0
                            else:
                                return True, total_duration
                        else:
                            self.get_logger().error("MoveIt ExecuteTrajectory goal rejected")
                            return False, 0.0
                
                except Exception as e:
                    self.get_logger().error(f"ExecuteTrajectory error: {e}")
                    # Fall through to FollowJointTrajectory fallback
        
        # === FALLBACK: Direct FollowJointTrajectory ===
        if not self.follow_trajectory_client:
            self.get_logger().error("No trajectory execution method available")
            return False, 0.0
        
        if not self.follow_trajectory_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("FollowJointTrajectory server not available")
            return False, 0.0
        
        try:
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = jt
            
            joint_names = list(jt.joint_names)
            goal.goal_time_tolerance = RosDuration(sec=0, nanosec=500000000)
            goal.goal_tolerance = [
                JointTolerance(position=0.01, velocity=0.01, name=name)
                for name in joint_names
            ]
            
            self.get_logger().info("Sending trajectory via FollowJointTrajectory (fallback)...")
            send_goal_future = self.follow_trajectory_client.send_goal_async(goal)
            
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: rclpy.spin_until_future_complete(
                            self, send_goal_future, timeout_sec=5.0
                        )
                    ),
                    timeout=6.0
                )
            except asyncio.TimeoutError:
                self.get_logger().error("Timeout waiting for FollowJointTrajectory goal acceptance")
                return False, 0.0
            
            if send_goal_future.done():
                goal_handle = send_goal_future.result()
                if goal_handle.accepted:
                    self.get_logger().info(f"FollowJointTrajectory accepted (duration={total_duration:.2f}s)")
                    if wait_for_completion:
                        wait_time = total_duration + 2.0
                        self.get_logger().info(f"Waiting {wait_time:.1f}s for completion...")
                        await asyncio.sleep(wait_time)
                    return True, total_duration
                else:
                    self.get_logger().error("FollowJointTrajectory goal rejected")
                    return False, 0.0
            
            return False, 0.0
            
        except Exception as e:
            self.get_logger().error(f"Error in FollowJointTrajectory: {e}")
            return False, 0.0

    async def _execute_real_robot_movement(
        self,
        target_joints: List[float],
        velocity: float = 0.5,
        acceleration: float = 0.5,
    ) -> bool:
        """
        Execute movement on the real robot via the /robot/move_joints service.
        
        This calls the robot_control_node's MoveJoints service which sends URScript
        commands to the physical UR robot.
        
        Args:
            target_joints: Target joint positions [j0, j1, j2, j3, j4, j5] in radians
            velocity: Joint velocity scale (0.0 to 1.05)
            acceleration: Joint acceleration scale (0.0 to 1.4)
            
        Returns:
            True if motion was initiated successfully, False otherwise
        """
        if not self.move_joints_client:
            self.get_logger().error("MoveJoints service client not available")
            return False
        
        # Wait for service to be available (with timeout)
        if not self.move_joints_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("MoveJoints service not available after 2s timeout")
            return False
        
        # Create request
        request = MoveJoints.Request()
        request.joint_positions = target_joints
        request.velocity = velocity * 1.05  # Scale to UR robot velocity range
        request.acceleration = acceleration * 1.4  # Scale to UR robot accel range
        request.wait_for_completion = False  # Non-blocking, motion runs on robot
        
        self.get_logger().info(f"Sending real robot motion: {[f'{j:.3f}' for j in target_joints]}")
        
        try:
            # Call service asynchronously
            future = self.move_joints_client.call_async(request)
            
            # Wait for response (with timeout)
            try:
                response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, lambda: rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)),
                    timeout=6.0
                )
            except asyncio.TimeoutError:
                self.get_logger().error("MoveJoints service call timed out")
                return False
            
            if future.done():
                result = future.result()
                if result.success:
                    self.get_logger().info(f"Real robot motion initiated: {result.message}")
                    return True
                else:
                    self.get_logger().error(f"Real robot motion failed: {result.message}")
                    return False
            else:
                self.get_logger().error("MoveJoints service call incomplete")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Error calling MoveJoints service: {e}")
            return False

    async def broadcast(self, message: str):
        """Broadcast message to all connected clients."""
        for client in self._connected_clients.values():
            try:
                await client.websocket.send(message)
            except Exception as e:
                self.get_logger().error(
                    f"Broadcast error to {client.client_id}: {e}"
                )


async def run_gateway(node: CommandGatewayNode):
    """Run the command gateway with WebSocket server."""
    await node.start_websocket_server()
    
    # Keep running
    try:
        await asyncio.Future()  # Run forever
    except asyncio.CancelledError:
        pass


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    node = CommandGatewayNode()
    
    # Create executor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    # Run both ROS 2 and WebSocket
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Start WebSocket server
        ws_task = loop.create_task(run_gateway(node))
        
        # Run ROS 2 executor in thread
        import threading
        ros_thread = threading.Thread(target=executor.spin, daemon=True)
        ros_thread.start()
        
        # Run event loop
        loop.run_until_complete(ws_task)
        
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Command Gateway...")
    finally:
        if node.ws_server:
            node.ws_server.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
