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
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState

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
        
        # Proto-sim control flags
        self._proto_sim_running = False
        self._proto_sim_stop_requested = False
        
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
            ping_interval=20,
            ping_timeout=10,
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
            elif method == 'run_proto_sim':
                result = await self.rpc_run_proto_sim(client, request_id, params)
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
        """Get available robots and objects in the environment."""
        # TODO: Get actual environment info from URDF/config
        # For now, return configured values
        return {
            'success': True,
            'robots': [self.default_robot],
            'objects': ['face_link', 'table_link', 'shop_floor'],
            'reference_frames': {
                'world': 'World Origin',
                'base_link': 'Robot Base',
                'tool0': 'Tool Center Point',
                'face_link': 'Face Fixture',
            },
        }

    async def rpc_run_proto_sim(
        self,
        client: ConnectedClient,
        request_id: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run protocol simulation with actual robot movement visualization.
        
        Generates spherical coordinates around target object and moves
        the robot to each pose while publishing joint states for visualization.
        
        Parameters match simforge_new/control/proto_simulation.py:
        - horiz: horizontal shift from object center (mm)
        - vert: vertical shift from object center (mm)
        - distance: distance from object (mm)
        - roll: roll angle (degrees)
        - pitch: pitch angle (degrees) - look up/down
        - yaw: yaw angle (degrees) - look left/right
        """
        if self._proto_sim_running:
            return {
                'success': False,
                'error': 'Protocol simulation already running',
            }
        
        self._proto_sim_running = True
        self._proto_sim_stop_requested = False
        
        robot_name = params.get('robot_name', self.default_robot)
        target_object = params.get('target_object', 'face_link')
        
        # Parameters matching simforge_new structure
        horiz = params.get('horiz', [0])  # mm
        vert = params.get('vert', [0])  # mm
        distance = params.get('distance', [250, 350, 450, 550])  # mm
        roll = params.get('roll', [-90])  # degrees
        pitch = params.get('pitch', [-45, -30, 0, 15])  # degrees
        yaw = params.get('yaw', [-30, 0, 30])  # degrees
        
        idle_time = params.get('idle_time', 2.0)
        randomize = params.get('randomize', False)
        mode = params.get('mode', 'simulation')  # 'simulation', 'real', 'both'
        move_speed = params.get('move_speed', 0.5)  # 0.0-1.0
        
        self.get_logger().info(
            f"Proto-sim: {robot_name} -> {target_object}, mode={mode}"
        )
        self.get_logger().info(
            f"  horiz={horiz}, vert={vert}, distance={distance}"
        )
        self.get_logger().info(
            f"  roll={roll}, pitch={pitch}, yaw={yaw}"
        )
        self.get_logger().info(f"  idle_time={idle_time}, move_speed={move_speed}, randomize={randomize}")
        
        # Pause real robot's joint state publishing during simulation
        if mode in ('simulation', 'both'):
            self.get_logger().info("Attempting to pause real robot joint publishing...")
            try:
                await self._pause_real_robot_joint_publishing()
                self.get_logger().info("Pause request completed")
            except Exception as e:
                self.get_logger().error(f"Failed to pause joint publishing: {e}")
        
        # Generate pose list from all parameter combinations
        # Matching simforge_new order: horiz -> vert -> distance -> roll -> pitch -> yaw
        import itertools
        poses = list(itertools.product(horiz, vert, pitch, yaw, distance, roll))
        total_poses = len(poses)
        
        if randomize:
            import random
            random.shuffle(poses)
        
        self.get_logger().info(f"Proto-sim: {total_poses} poses to execute")
        
        # Execute poses
        completed = 0
        failed = 0
        
        try:
            self.get_logger().info(f"Starting pose execution loop with {total_poses} poses")
            for i, (h, v, p, y, d, r) in enumerate(poses):
                if self._proto_sim_stop_requested:
                    self.get_logger().info("Proto-sim stopped by user request")
                    break
                
                pose_name = f"H{h}_V{v}_D{d}_R{r}_P{p}_Y{y}"
                if i == 0 or (i + 1) % 5 == 0:
                    self.get_logger().info(f"Executing pose {i+1}/{total_poses}: {pose_name}")
                
                # Generate target joint configuration for this pose
                target_joints = self._compute_proto_pose_joints(
                    horiz_mm=h, vert_mm=v, dist_mm=d,
                    roll_deg=r, pitch_deg=p, yaw_deg=y
                )
                
                if target_joints is None:
                    self.get_logger().warn(f"No valid joints for pose {pose_name}")
                    failed += 1
                    continue
                
                # Send progress update - moving to pose
                feedback = {
                    'type': 'rpc_feedback',
                    'request_id': request_id,
                    'current_pose_index': i,
                    'total_poses': total_poses,
                    'current_pose_name': pose_name,
                    'progress_percent': (i / total_poses) * 100,
                    'status': 'moving',
                }
                await client.websocket.send(json.dumps(feedback))
                
                # Execute simulated movement with joint state publishing
                if mode in ('simulation', 'both'):
                    self.get_logger().info(f"Moving to joints: {[f'{j:.2f}' for j in target_joints]}")
                    try:
                        await self._execute_sim_movement(
                            target_joints, 
                            move_speed=move_speed
                        )
                    except Exception as e:
                        self.get_logger().error(f"Movement error: {e}")
                        import traceback
                        self.get_logger().error(traceback.format_exc())
                
                # TODO: If mode is 'real' or 'both', also send to real robot
                # if mode in ('real', 'both'):
                #     await self._execute_real_movement(target_joints)
                
                # Send progress update - at pose (idle)
                feedback['status'] = 'idle'
                await client.websocket.send(json.dumps(feedback))
                
                # Wait at pose (idle time)
                idle_steps = int(idle_time * 10)  # 10 Hz check for stop
                for _ in range(idle_steps):
                    if self._proto_sim_stop_requested:
                        break
                    await asyncio.sleep(0.1)
                    # Keep publishing joint states while idle
                    self._publish_joint_state()
                
                completed += 1
                
        finally:
            self._proto_sim_running = False
            # Resume real robot's joint state publishing
            if mode in ('simulation', 'both'):
                await self._resume_real_robot_joint_publishing()
        
        # Send final result
        result = {
            'type': 'rpc_result',
            'request_id': request_id,
            'success': not self._proto_sim_stop_requested,
            'message': f'Completed {completed}/{total_poses} poses',
            'completed': completed,
            'failed': failed,
            'total': total_poses,
            'stopped': self._proto_sim_stop_requested,
        }
        await client.websocket.send(json.dumps(result))
        
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

    def _compute_proto_pose_joints(
        self,
        horiz_mm: float,
        vert_mm: float,
        dist_mm: float,
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
    ) -> Optional[List[float]]:
        """
        Compute joint configuration for a proto-sim pose.
        
        This generates demonstration poses that create visible robot motion.
        In production, this would use IK to compute actual poses pointing
        at the target object from the specified spherical coordinates.
        """
        # Convert mm to meters for internal calculations
        dist_m = dist_mm / 1000.0
        horiz_m = horiz_mm / 1000.0
        vert_m = vert_mm / 1000.0
        
        # Generate visually interesting joint configurations
        # Base rotation influenced by yaw
        j0 = math.radians(yaw_deg) * 0.02  # Scale down for safety
        
        # Shoulder influenced by pitch and distance
        j1 = -math.pi/2 + math.radians(pitch_deg) * 0.01 + (dist_m - 0.35) * 0.8
        
        # Elbow influenced by distance
        j2 = (0.4 - dist_m) * 2.5
        
        # Wrist 1 influenced by pitch
        j3 = -math.pi/2 - math.radians(pitch_deg) * 0.02
        
        # Wrist 2 influenced by yaw and roll
        j4 = math.radians(yaw_deg) * 0.02 + math.radians(roll_deg) * 0.01
        
        # Wrist 3 influenced by roll
        j5 = math.radians(roll_deg) * 0.01
        
        # Add horizontal and vertical shifts as small offsets
        j0 += horiz_m * 0.5
        j1 += vert_m * 0.3
        
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
    ) -> None:
        """
        Execute simulated movement by interpolating joint positions
        and publishing JointState messages.
        """
        start_joints = np.array(self.sim_joint_positions)
        end_joints = np.array(target_joints)
        
        # Calculate movement duration based on max joint change and speed
        max_delta = np.max(np.abs(end_joints - start_joints))
        # Scale duration: slower speed = longer duration
        base_duration = max_delta / (math.pi * 0.5)  # ~2 seconds for 90 deg at speed=1.0
        duration = base_duration / max(0.1, move_speed)
        duration = max(0.2, min(duration, 5.0))  # Clamp between 0.2s and 5s
        
        self.get_logger().info(
            f"Sim movement: duration={duration:.2f}s, max_delta={max_delta:.3f}rad"
        )
        
        # Interpolate and publish
        dt = 1.0 / publish_rate
        num_steps = int(duration * publish_rate)
        
        for step in range(num_steps + 1):
            if self._proto_sim_stop_requested:
                break
            
            t = step / max(1, num_steps)  # 0 to 1
            # Smooth interpolation (ease in/out)
            t_smooth = (1 - math.cos(t * math.pi)) / 2
            
            # Interpolate joints
            current_joints = start_joints + (end_joints - start_joints) * t_smooth
            self.sim_joint_positions = current_joints.tolist()
            
            # Publish joint state
            self._publish_joint_state()
            
            await asyncio.sleep(dt)
        
        # Ensure we end at target
        self.sim_joint_positions = target_joints
        self._publish_joint_state()
        
        self.get_logger().info(f"Sim movement complete, final joints: {[f'{j:.3f}' for j in target_joints]}")

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
            self.get_logger().info(f\"Publishing joint state #{self._pub_count}: {[f'{j:.2f}' for j in self.sim_joint_positions]}\")

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
