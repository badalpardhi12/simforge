"""
Simforge Command Client

WebSocket client for controlling robots from macOS (or any platform).
Communicates with the Command Gateway Node on the server via JSON-RPC over WebSocket.

This client handles:
- Connection management with automatic reconnection
- Heartbeat sending for safety watchdog
- Sending movement commands (joint, Cartesian, trajectory)
- Receiving feedback and results
- Emergency stop functionality
"""

import asyncio
import json
import time
import logging
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, asdict
from enum import IntEnum
import ssl

try:
    from websockets.client import connect as ws_connect, WebSocketClientProtocol
    from websockets.exceptions import ConnectionClosed, WebSocketException
except ImportError:
    raise ImportError("websockets package required. Install with: pip install websockets>=12.0")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MotionType(IntEnum):
    """Motion type enumeration."""
    JOINT = 0
    CARTESIAN = 1
    TRAJECTORY = 2


class ConnectionState(IntEnum):
    """Connection state enumeration."""
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2
    RECONNECTING = 3


@dataclass
class Pose:
    """6-DOF pose (position + quaternion orientation)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'Pose':
        return cls(**d)


@dataclass
class MoveResult:
    """Result of a move command."""
    success: bool
    message: str = ""
    final_joint_positions: Optional[List[float]] = None
    final_pose: Optional[Pose] = None
    execution_time_sec: float = 0.0
    planning_time_sec: float = 0.0
    replan_count: int = 0


@dataclass
class MoveFeedback:
    """Feedback during move execution."""
    progress: float  # 0.0 to 1.0
    current_joint_positions: Optional[List[float]] = None
    current_pose: Optional[Pose] = None
    status: str = ""
    distance_to_goal: float = 0.0
    time_remaining: float = 0.0


@dataclass
class ClientConfig:
    """Client configuration."""
    server_ip: str = "localhost"
    command_port: int = 8765
    heartbeat_rate_hz: float = 2.0  # 2Hz - send heartbeat every 500ms
    reconnect_delay_sec: float = 1.0
    max_reconnect_attempts: int = 10
    client_id: str = "mac_client"
    use_ssl: bool = False
    connection_timeout_sec: float = 5.0


class SimforgeClient:
    """
    WebSocket client for controlling robots from macOS.
    
    Usage:
        async with SimforgeClient(server_ip="192.168.1.100") as client:
            result = await client.move_robot(
                "ur20",
                target_joints=[0, -1.57, 1.57, 0, 0, 0],
                velocity_scale=0.3
            )
    
    Or without async context manager:
        client = SimforgeClient(server_ip="192.168.1.100")
        await client.connect()
        result = await client.move_robot(...)
        await client.disconnect()
    """

    def __init__(
        self,
        server_ip: str = "localhost",
        command_port: int = 8765,
        client_id: str = "mac_client",
        **kwargs
    ):
        """
        Initialize the Simforge client.
        
        Args:
            server_ip: IP address of the server (AI Workstation or Jetson Thor)
            command_port: WebSocket port for command gateway (default 8765)
            client_id: Unique identifier for this client
            **kwargs: Additional configuration options (see ClientConfig)
        """
        self.config = ClientConfig(
            server_ip=server_ip,
            command_port=command_port,
            client_id=client_id,
            **kwargs
        )
        
        self._ws: Optional[WebSocketClientProtocol] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._receiver_task: Optional[asyncio.Task] = None
        self._sequence_number: int = 0
        self._state = ConnectionState.DISCONNECTED
        self._reconnect_attempts: int = 0
        
        # Callbacks
        self._feedback_callbacks: Dict[str, Callable[[MoveFeedback], None]] = {}
        self._state_callbacks: List[Callable[[ConnectionState], None]] = []
        
        # Pending requests (for matching responses)
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._request_id: int = 0

    @property
    def server_uri(self) -> str:
        """Get the WebSocket URI."""
        protocol = "wss" if self.config.use_ssl else "ws"
        return f"{protocol}://{self.config.server_ip}:{self.config.command_port}"

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._state == ConnectionState.CONNECTED and self._ws is not None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> bool:
        """
        Connect to the server.
        
        Returns:
            True if connected successfully, False otherwise.
        """
        if self._state == ConnectionState.CONNECTED:
            logger.warning("Already connected")
            return True
        
        self._state = ConnectionState.CONNECTING
        self._notify_state_change()
        
        try:
            ssl_context = ssl.create_default_context() if self.config.use_ssl else None
            
            logger.info(f"Connecting to {self.server_uri}...")
            
            # Build connection kwargs - be compatible with different websockets versions
            connect_kwargs = {}
            
            # Only add ssl context if using wss
            if self.config.use_ssl and ssl_context:
                connect_kwargs["ssl"] = ssl_context
            
            # Try connecting - websockets API varies between versions
            try:
                # Try with ping parameters (websockets 10.x style)
                self._ws = await asyncio.wait_for(
                    ws_connect(self.server_uri, ping_interval=20, ping_timeout=10, **connect_kwargs),
                    timeout=self.config.connection_timeout_sec
                )
            except TypeError:
                # Fall back to basic connection (older or newer API)
                self._ws = await asyncio.wait_for(
                    ws_connect(self.server_uri, **connect_kwargs),
                    timeout=self.config.connection_timeout_sec
                )
            
            self._state = ConnectionState.CONNECTED
            self._reconnect_attempts = 0
            self._notify_state_change()
            
            # Start background tasks
            self._heartbeat_task = asyncio.create_task(self._send_heartbeat())
            self._receiver_task = asyncio.create_task(self._receive_messages())
            
            logger.info(f"Connected to {self.server_uri}")
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"Connection timeout to {self.server_uri}")
            self._state = ConnectionState.DISCONNECTED
            return False
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._state = ConnectionState.DISCONNECTED
            return False

    async def disconnect(self):
        """Disconnect from the server."""
        logger.info("Disconnecting...")
        
        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self._receiver_task:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket
        if self._ws:
            await self._ws.close()
            self._ws = None
        
        self._state = ConnectionState.DISCONNECTED
        self._notify_state_change()
        logger.info("Disconnected")

    async def _send_heartbeat(self):
        """Send heartbeat at configured rate to maintain safety watchdog."""
        interval = 1.0 / self.config.heartbeat_rate_hz
        
        while True:
            try:
                if self._ws and self._state == ConnectionState.CONNECTED:
                    self._sequence_number += 1
                    
                    heartbeat = {
                        "type": "heartbeat",
                        "client_id": self.config.client_id,
                        "timestamp_ns": time.time_ns(),
                        "sequence": self._sequence_number,
                    }
                    
                    await self._ws.send(json.dumps(heartbeat))
                
                await asyncio.sleep(interval)
                
            except ConnectionClosed:
                logger.warning("Connection closed during heartbeat")
                await self._handle_disconnect()
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(interval)

    async def _receive_messages(self):
        """Receive and dispatch messages from server."""
        while True:
            try:
                if not self._ws:
                    await asyncio.sleep(0.1)
                    continue
                
                message = await self._ws.recv()
                msg = json.loads(message)
                
                msg_type = msg.get("type", "")
                
                if msg_type == "feedback":
                    self._handle_feedback(msg)
                elif msg_type == "result":
                    self._handle_result(msg)
                elif msg_type == "rpc_result":
                    self._handle_rpc_result(msg)
                elif msg_type == "rpc_feedback":
                    self._handle_rpc_feedback(msg)
                elif msg_type == "robot_state":
                    self._handle_robot_state(msg)
                elif msg_type == "error":
                    self._handle_error(msg)
                else:
                    logger.debug(f"Unknown message type: {msg_type}")
                    
            except ConnectionClosed:
                logger.warning("Connection closed")
                await self._handle_disconnect()
                break
            except asyncio.CancelledError:
                break
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
            except Exception as e:
                logger.error(f"Receive error: {e}")
                await asyncio.sleep(0.5)  # Prevent tight error loops

    async def _handle_disconnect(self):
        """Handle unexpected disconnection with reconnection logic."""
        if self._state == ConnectionState.DISCONNECTED:
            return
        
        self._state = ConnectionState.RECONNECTING
        self._notify_state_change()
        
        while self._reconnect_attempts < self.config.max_reconnect_attempts:
            self._reconnect_attempts += 1
            logger.info(
                f"Reconnection attempt {self._reconnect_attempts}/"
                f"{self.config.max_reconnect_attempts}"
            )
            
            await asyncio.sleep(self.config.reconnect_delay_sec)
            
            if await self.connect():
                return
        
        logger.error("Max reconnection attempts reached")
        self._state = ConnectionState.DISCONNECTED
        self._notify_state_change()

    def _handle_feedback(self, msg: Dict[str, Any]):
        """Handle movement feedback."""
        request_id = msg.get("request_id")
        if request_id and request_id in self._feedback_callbacks:
            feedback = MoveFeedback(
                progress=msg.get("progress", 0.0),
                status=msg.get("status", ""),
                distance_to_goal=msg.get("distance_to_goal", 0.0),
                time_remaining=msg.get("time_remaining", 0.0),
            )
            if msg.get("current_joint_positions"):
                feedback.current_joint_positions = msg["current_joint_positions"]
            if msg.get("current_pose"):
                feedback.current_pose = Pose.from_dict(msg["current_pose"])
            
            self._feedback_callbacks[request_id](feedback)

    def _handle_result(self, msg: Dict[str, Any]):
        """Handle command result."""
        request_id = msg.get("request_id")
        if request_id and request_id in self._pending_requests:
            future = self._pending_requests.pop(request_id)
            if not future.done():
                future.set_result(msg)
            
            # Clean up feedback callback
            self._feedback_callbacks.pop(request_id, None)

    def _handle_robot_state(self, msg: Dict[str, Any]):
        """Handle robot state update."""
        # Can be extended to track robot state locally
        pass

    def _handle_rpc_feedback(self, msg: Dict[str, Any]):
        """Handle RPC feedback (progress updates during long operations)."""
        # For now, just log - can be extended with callbacks
        logger.debug(f"RPC feedback: {msg}")

    def _handle_error(self, msg: Dict[str, Any]):
        """Handle error message."""
        request_id = msg.get("request_id")
        error_msg = msg.get("message", "Unknown error")
        
        logger.error(f"Server error: {error_msg}")
        
        if request_id and request_id in self._pending_requests:
            future = self._pending_requests.pop(request_id)
            if not future.done():
                future.set_exception(RuntimeError(error_msg))

    def _notify_state_change(self):
        """Notify registered callbacks of state change."""
        for callback in self._state_callbacks:
            try:
                callback(self._state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    def on_state_change(self, callback: Callable[[ConnectionState], None]):
        """Register a callback for connection state changes."""
        self._state_callbacks.append(callback)

    async def _send_request(
        self,
        request: Dict[str, Any],
        feedback_callback: Optional[Callable[[MoveFeedback], None]] = None,
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """Send a request and wait for result."""
        if not self.is_connected:
            raise RuntimeError("Not connected to server")
        
        # Generate request ID
        self._request_id += 1
        request_id = f"{self.config.client_id}_{self._request_id}"
        request["request_id"] = request_id
        
        # Create future for result
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future
        
        # Register feedback callback if provided
        if feedback_callback:
            self._feedback_callbacks[request_id] = feedback_callback
        
        try:
            # Send request
            await self._ws.send(json.dumps(request))
            
            # Wait for result with timeout
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
            
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            self._feedback_callbacks.pop(request_id, None)
            raise TimeoutError(f"Request timed out after {timeout}s")

    async def move_robot(
        self,
        robot_name: str,
        target_pose: Optional[Pose] = None,
        target_joints: Optional[List[float]] = None,
        velocity_scale: float = 0.5,
        acceleration_scale: float = 0.5,
        collision_check: bool = True,
        reference_frame: str = "world",
        allow_replanning: bool = True,
        timeout: float = 60.0,
        feedback_callback: Optional[Callable[[MoveFeedback], None]] = None,
    ) -> MoveResult:
        """
        Send a move command to the robot.
        
        Args:
            robot_name: Name of the robot (e.g., "ur20_1")
            target_pose: Target Cartesian pose (for CARTESIAN motion)
            target_joints: Target joint positions in radians (for JOINT motion)
            velocity_scale: Velocity scaling factor (0.0 to 1.0)
            acceleration_scale: Acceleration scaling factor (0.0 to 1.0)
            collision_check: Enable collision checking via cuMotion
            reference_frame: Reference frame for Cartesian motion
            allow_replanning: Allow replanning if obstacles detected
            timeout: Maximum time to wait for completion
            feedback_callback: Optional callback for progress updates
            
        Returns:
            MoveResult with success status and final state
        """
        # Determine motion type
        if target_joints is not None:
            motion_type = MotionType.JOINT
        elif target_pose is not None:
            motion_type = MotionType.CARTESIAN
        else:
            raise ValueError("Either target_pose or target_joints must be provided")
        
        request = {
            "type": "move_robot",
            "robot_name": robot_name,
            "motion_type": motion_type.value,
            "velocity_scale": velocity_scale,
            "acceleration_scale": acceleration_scale,
            "collision_check_enabled": collision_check,
            "reference_frame": reference_frame,
            "allow_replanning": allow_replanning,
        }
        
        if target_pose:
            request["target_pose"] = target_pose.to_dict()
        if target_joints:
            request["target_joints"] = target_joints
        
        logger.info(f"Moving {robot_name} - motion_type={motion_type.name}")
        
        result = await self._send_request(
            request,
            feedback_callback=feedback_callback,
            timeout=timeout
        )
        
        move_result = MoveResult(
            success=result.get("success", False),
            message=result.get("message", ""),
            execution_time_sec=result.get("execution_time_sec", 0.0),
            planning_time_sec=result.get("planning_time_sec", 0.0),
            replan_count=result.get("replan_count", 0),
        )
        
        if result.get("final_joint_positions"):
            move_result.final_joint_positions = result["final_joint_positions"]
        if result.get("final_pose"):
            move_result.final_pose = Pose.from_dict(result["final_pose"])
        
        return move_result

    async def execute_task(
        self,
        instruction: str,
        robot_name: str = "ur20",
        timeout: float = 120.0,
        max_velocity_scale: float = 0.3,
        feedback_callback: Optional[Callable[[MoveFeedback], None]] = None,
    ) -> MoveResult:
        """
        Execute a VLA-guided manipulation task.
        
        Args:
            instruction: Natural language instruction (e.g., "pick up the red cup")
            robot_name: Name of the robot to use
            timeout: Maximum time for task execution
            max_velocity_scale: Maximum velocity for safety
            feedback_callback: Optional callback for progress updates
            
        Returns:
            MoveResult with task completion status
        """
        request = {
            "type": "execute_task",
            "instruction": instruction,
            "robot_name": robot_name,
            "max_velocity_scale": max_velocity_scale,
        }
        
        logger.info(f"Executing task: {instruction}")
        
        result = await self._send_request(
            request,
            feedback_callback=feedback_callback,
            timeout=timeout
        )
        
        return MoveResult(
            success=result.get("success", False),
            message=result.get("message", ""),
            execution_time_sec=result.get("total_time_sec", 0.0),
        )

    async def emergency_stop(self) -> bool:
        """
        Trigger emergency stop on all robots.
        
        This is a fire-and-forget command that doesn't wait for confirmation.
        
        Returns:
            True if command was sent successfully
        """
        if not self.is_connected:
            logger.error("Cannot send E-STOP: not connected")
            return False
        
        try:
            request = {
                "type": "emergency_stop",
                "client_id": self.config.client_id,
                "timestamp_ns": time.time_ns(),
            }
            
            await self._ws.send(json.dumps(request))
            logger.critical("EMERGENCY STOP SENT")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send E-STOP: {e}")
            return False

    async def protective_stop(self) -> bool:
        """
        Trigger protective stop (controlled deceleration).
        
        Returns:
            True if command was sent successfully
        """
        if not self.is_connected:
            logger.error("Cannot send protective stop: not connected")
            return False
        
        try:
            request = {
                "type": "protective_stop",
                "client_id": self.config.client_id,
                "reason": "Client requested protective stop",
            }
            
            await self._ws.send(json.dumps(request))
            logger.warning("PROTECTIVE STOP SENT")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send protective stop: {e}")
            return False

    async def get_robot_state(self, robot_name: str) -> Optional[Dict[str, Any]]:
        """
        Get current state of a robot.
        
        Args:
            robot_name: Name of the robot
            
        Returns:
            Dictionary with robot state or None if unavailable
        """
        request = {
            "type": "get_robot_state",
            "robot_name": robot_name,
        }
        
        try:
            result = await self._send_request(request, timeout=5.0)
            return result.get("robot_state")
        except Exception as e:
            logger.error(f"Failed to get robot state: {e}")
            return None

    async def ping(self) -> float:
        """
        Ping the server and measure round-trip time.
        
        Returns:
            Round-trip time in milliseconds
        """
        start = time.time()
        
        request = {
            "type": "ping",
            "timestamp_ns": time.time_ns(),
        }
        
        result = await self._send_request(request, timeout=5.0)
        
        rtt = (time.time() - start) * 1000
        logger.debug(f"Ping: {rtt:.1f}ms")
        return rtt

    async def call_rpc(
        self,
        method: str,
        params: Dict[str, Any],
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Call a generic RPC method on the server.
        
        Args:
            method: RPC method name (e.g., 'get_environment_info', 'run_proto_sim')
            params: Parameters to pass to the method
            timeout: Maximum time to wait for result
            
        Returns:
            Dictionary with RPC result
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to server")
        
        # Generate request ID
        self._request_id += 1
        request_id = f"{self.config.client_id}_{self._request_id}"
        
        request = {
            "type": "rpc",
            "request_id": request_id,
            "method": method,
            "params": params,
        }
        
        # Create future for result
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future
        
        try:
            # Send request
            await self._ws.send(json.dumps(request))
            
            # Wait for result with timeout
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
            
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise TimeoutError(f"RPC call '{method}' timed out after {timeout}s")

    def _handle_rpc_result(self, msg: Dict[str, Any]):
        """Handle RPC result message."""
        request_id = msg.get("request_id")
        if request_id and request_id in self._pending_requests:
            future = self._pending_requests.pop(request_id)
            if not future.done():
                future.set_result(msg)


# Convenience function for simple usage
async def connect(server_ip: str, **kwargs) -> SimforgeClient:
    """
    Create and connect a SimforgeClient.
    
    Args:
        server_ip: IP address of the server
        **kwargs: Additional configuration options
        
    Returns:
        Connected SimforgeClient instance
    """
    client = SimforgeClient(server_ip=server_ip, **kwargs)
    await client.connect()
    return client
