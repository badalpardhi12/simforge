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
from typing import Dict, Optional, Any
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose, Point, Quaternion
from trajectory_msgs.msg import JointTrajectory

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
        
        # === Service Clients ===
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
        """Run protocol simulation."""
        robot_name = params.get('robot_name', self.default_robot)
        target_object = params.get('target_object', 'face_link')
        distances = params.get('distances', [0.3, 0.4, 0.5])
        horizontal_angles = params.get('horizontal_angles', [-30, 0, 30])
        vertical_angles = params.get('vertical_angles', [-15, 0, 15])
        idle_time = params.get('idle_time', 2.0)
        randomize = params.get('randomize', False)
        mode = params.get('mode', 'simulation')  # 'simulation', 'real', 'both'
        
        self.get_logger().info(
            f"Proto-sim: {robot_name} -> {target_object}, mode={mode}"
        )
        
        # Generate pose list
        import itertools
        poses = list(itertools.product(distances, horizontal_angles, vertical_angles))
        total_poses = len(poses)
        
        if randomize:
            import random
            random.shuffle(poses)
        
        self.get_logger().info(f"Proto-sim: {total_poses} poses to execute")
        
        # Execute poses
        completed = 0
        failed = 0
        
        for i, (dist, h_angle, v_angle) in enumerate(poses):
            pose_name = f"D{dist}_H{h_angle}_V{v_angle}"
            
            # Send progress update
            feedback = {
                'type': 'rpc_feedback',
                'request_id': request_id,
                'current_pose_index': i,
                'total_poses': total_poses,
                'current_pose_name': pose_name,
                'progress_percent': (i / total_poses) * 100,
                'status': 'executing',
            }
            await client.websocket.send(json.dumps(feedback))
            
            # TODO: In production:
            # 1. Calculate target pose from object position + spherical coords
            # 2. Plan collision-free path using motion_planner
            # 3. Execute on simulation (always) and/or real robot (if mode allows)
            # 4. Wait for idle_time at pose
            
            # Simulate execution time
            await asyncio.sleep(0.3)  # Reduced for testing
            
            completed += 1
        
        # Send final result
        result = {
            'type': 'rpc_result',
            'request_id': request_id,
            'success': True,
            'message': f'Completed {completed}/{total_poses} poses',
            'completed': completed,
            'failed': failed,
            'total': total_poses,
        }
        await client.websocket.send(json.dumps(result))
        
        return result

    async def rpc_stop_proto_sim(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Stop running protocol simulation."""
        # TODO: Implement stop mechanism
        return {'success': True, 'message': 'Stop requested'}

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
