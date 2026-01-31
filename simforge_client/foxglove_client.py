"""
Foxglove WebSocket Client for Visualization

This module provides a client for connecting to Foxglove Bridge to receive
ROS 2 visualization data (robot state, meshes, camera feeds, etc.)

Uses the foxglove-websocket package for protocol compliance.
"""

import asyncio
import json
import logging
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass

try:
    from foxglove_websocket import Client
    HAS_FOXGLOVE = True
except ImportError:
    HAS_FOXGLOVE = False
    logging.warning(
        "foxglove-websocket not installed. "
        "Install with: pip install foxglove-websocket"
    )

logger = logging.getLogger(__name__)


@dataclass
class TopicInfo:
    """Information about a ROS 2 topic."""
    name: str
    message_type: str
    encoding: str = "json"


class FoxgloveClient:
    """
    Client for connecting to Foxglove Bridge on the server.
    
    This provides:
    - Real-time robot state visualization
    - 3D mesh visualization from nvblox
    - Camera feeds
    - Trajectory visualization
    
    Usage:
        client = FoxgloveClient(server_ip="192.168.1.100")
        await client.connect()
        
        await client.subscribe("/joint_states", callback=handle_joints)
        await client.subscribe("/nvblox/mesh", callback=handle_mesh)
        
        # Run message loop
        async for message in client:
            process(message)
    """

    # Default topics to subscribe to
    DEFAULT_TOPICS = [
        TopicInfo("/joint_states", "sensor_msgs/JointState"),
        TopicInfo("/robot_state", "simforge_msgs/RobotState"),
        TopicInfo("/safety/status", "simforge_msgs/SafetyStatus"),
        TopicInfo("/camera/color/image_raw", "sensor_msgs/Image"),
        TopicInfo("/nvblox/mesh", "nvblox_msgs/Mesh3D"),
        TopicInfo("/cumotion/planned_path", "nav_msgs/Path"),
    ]

    def __init__(
        self,
        server_ip: str = "localhost",
        port: int = 9090,
    ):
        """
        Initialize the Foxglove client.
        
        Args:
            server_ip: IP address of the Foxglove Bridge server
            port: WebSocket port (default 9090)
        """
        if not HAS_FOXGLOVE:
            raise ImportError(
                "foxglove-websocket required. Install with: pip install foxglove-websocket"
            )
        
        self.server_ip = server_ip
        self.port = port
        self.uri = f"ws://{server_ip}:{port}"
        
        self._client: Optional[Client] = None
        self._subscriptions: Dict[str, Callable] = {}
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if connected to Foxglove Bridge."""
        return self._connected

    async def connect(self) -> bool:
        """
        Connect to Foxglove Bridge.
        
        Returns:
            True if connected successfully
        """
        try:
            logger.info(f"Connecting to Foxglove Bridge at {self.uri}...")
            self._client = Client(self.uri)
            await self._client.__aenter__()
            self._connected = True
            logger.info("Connected to Foxglove Bridge")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Foxglove Bridge: {e}")
            self._connected = False
            return False

    async def disconnect(self):
        """Disconnect from Foxglove Bridge."""
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._connected = False
            logger.info("Disconnected from Foxglove Bridge")

    async def subscribe(
        self,
        topic: str,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Subscribe to a ROS 2 topic.
        
        Args:
            topic: Topic name (e.g., "/joint_states")
            callback: Optional callback function for messages
        """
        if not self._connected:
            raise RuntimeError("Not connected to Foxglove Bridge")
        
        logger.info(f"Subscribing to {topic}")
        await self._client.subscribe(topic)
        
        if callback:
            self._subscriptions[topic] = callback

    async def subscribe_defaults(self, callbacks: Dict[str, Callable] = None):
        """
        Subscribe to default visualization topics.
        
        Args:
            callbacks: Dict mapping topic names to callback functions
        """
        callbacks = callbacks or {}
        
        for topic_info in self.DEFAULT_TOPICS:
            try:
                await self.subscribe(
                    topic_info.name,
                    callback=callbacks.get(topic_info.name)
                )
            except Exception as e:
                logger.warning(f"Could not subscribe to {topic_info.name}: {e}")

    async def __aiter__(self):
        """Async iterator for receiving messages."""
        if not self._connected:
            raise RuntimeError("Not connected to Foxglove Bridge")
        
        async for message in self._client:
            # Dispatch to registered callbacks
            topic = message.get("topic", "")
            if topic in self._subscriptions:
                try:
                    self._subscriptions[topic](message)
                except Exception as e:
                    logger.error(f"Callback error for {topic}: {e}")
            
            yield message

    async def get_topics(self) -> List[TopicInfo]:
        """
        Get list of available topics from server.
        
        Returns:
            List of TopicInfo objects
        """
        # This would require additional Foxglove protocol support
        # For now, return default topics
        return self.DEFAULT_TOPICS


async def demo():
    """Demo of Foxglove client functionality."""
    
    def handle_joint_states(msg):
        data = msg.get("data", {})
        positions = data.get("position", [])
        print(f"Joint positions: {[f'{p:.2f}' for p in positions[:6]]}")
    
    def handle_robot_state(msg):
        data = msg.get("data", {})
        mode = data.get("robot_mode", 0)
        print(f"Robot mode: {mode}")
    
    client = FoxgloveClient(server_ip="localhost")
    
    if await client.connect():
        await client.subscribe("/joint_states", handle_joint_states)
        await client.subscribe("/robot_state", handle_robot_state)
        
        print("Receiving messages... (Ctrl+C to stop)")
        
        try:
            async for message in client:
                pass  # Callbacks handle the messages
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            await client.disconnect()
    else:
        print("Could not connect to Foxglove Bridge")


if __name__ == "__main__":
    asyncio.run(demo())
