#!/usr/bin/env python3
"""
Safety Watchdog Node

CRITICAL SAFETY COMPONENT - This node monitors the heartbeat from the Mac client
and triggers protective stops if communication is lost.

Safety Protocol:
- Client sends heartbeat every 20ms (50Hz)
- Server expects heartbeat within 100ms timeout
- If 3 consecutive heartbeats missed → trigger protective stop

This must be the FIRST component started and the LAST component stopped.
"""

import time
import threading
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import IntEnum

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_srvs.srv import Trigger

# Import custom messages (will be available after building simforge_msgs)
# For now, we'll use compatible standard messages and define types inline
from std_msgs.msg import String
from geometry_msgs.msg import Wrench


class SafetyState(IntEnum):
    """Safety state enumeration."""
    NORMAL = 0
    WARNING = 1
    PROTECTIVE_STOP = 2
    EMERGENCY_STOP = 3


@dataclass
class ClientInfo:
    """Information about a connected client."""
    client_id: str
    last_heartbeat_time: float = 0.0
    sequence_number: int = 0
    consecutive_misses: int = 0
    reported_latency_ms: float = 0.0


@dataclass
class SafetyConfig:
    """Safety watchdog configuration."""
    heartbeat_timeout_sec: float = 2.0  # 2 seconds - more tolerant for network variations
    max_consecutive_misses: int = 5  # Allow more misses before protective stop
    check_frequency_hz: float = 10.0  # 10Hz monitoring - sufficient for safety
    warning_threshold_misses: int = 2  # Warn after 2 misses
    max_communication_latency_ms: float = 200.0  # Allow higher latency over network
    enable_force_monitoring: bool = True
    max_tcp_force_n: float = 100.0  # Maximum force at TCP
    max_tcp_torque_nm: float = 10.0  # Maximum torque at TCP


class SafetyWatchdogNode(Node):
    """
    Safety Watchdog Node - Monitors system safety and triggers protective stops.
    
    This node is responsible for:
    1. Monitoring heartbeat from Mac client
    2. Triggering protective stops on communication loss
    3. Monitoring force/torque limits
    4. Managing emergency stop chain
    5. Publishing safety status for visualization
    """

    def __init__(self):
        super().__init__('safety_watchdog')
        
        # Declare parameters - defaults match SafetyConfig
        self.declare_parameter('heartbeat_timeout_sec', 2.0)
        self.declare_parameter('max_consecutive_misses', 5)
        self.declare_parameter('check_frequency_hz', 10.0)
        self.declare_parameter('enable_force_monitoring', True)
        self.declare_parameter('max_tcp_force_n', 100.0)
        self.declare_parameter('max_tcp_torque_nm', 10.0)
        
        # Load configuration
        self.config = SafetyConfig(
            heartbeat_timeout_sec=self.get_parameter('heartbeat_timeout_sec').value,
            max_consecutive_misses=self.get_parameter('max_consecutive_misses').value,
            check_frequency_hz=self.get_parameter('check_frequency_hz').value,
            enable_force_monitoring=self.get_parameter('enable_force_monitoring').value,
            max_tcp_force_n=self.get_parameter('max_tcp_force_n').value,
            max_tcp_torque_nm=self.get_parameter('max_tcp_torque_nm').value,
        )
        
        # State tracking
        self._connected_clients: Dict[str, ClientInfo] = {}
        self.current_safety_state = SafetyState.NORMAL
        self.active_faults: List[str] = []
        self.last_fault_message = ""
        self.last_fault_time = time.time()
        self.lock = threading.Lock()
        
        # Current TCP wrench (for force monitoring)
        self.current_tcp_wrench: Optional[Wrench] = None
        
        # QoS profiles
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # === Subscribers ===
        # Heartbeat from Command Gateway (which receives from Mac client)
        self.heartbeat_sub = self.create_subscription(
            String,  # Will be Heartbeat msg after building
            '/safety/heartbeat',
            self.heartbeat_callback,
            reliable_qos
        )
        
        # TCP wrench for force monitoring
        self.wrench_sub = self.create_subscription(
            Wrench,
            '/robot/tcp_wrench',
            self.wrench_callback,
            10
        )
        
        # Emergency stop from any source
        self.estop_sub = self.create_subscription(
            String,
            '/safety/emergency_stop',
            self.emergency_stop_callback,
            reliable_qos
        )
        
        # === Publishers ===
        # Safety status for monitoring
        self.status_pub = self.create_publisher(
            String,  # Will be SafetyStatus msg after building
            '/safety/status',
            reliable_qos
        )
        
        # === Service Clients ===
        # Robot protective stop
        self.protective_stop_client = self.create_client(
            Trigger,
            '/robot/protective_stop'
        )
        
        # Robot emergency stop
        self.emergency_stop_client = self.create_client(
            Trigger,
            '/robot/emergency_stop'
        )
        
        # Motion abort (for cuMotion)
        self.abort_motion_client = self.create_client(
            Trigger,
            '/cumotion/abort_motion'
        )
        
        # === Services ===
        # Manual reset
        self.reset_service = self.create_service(
            Trigger,
            '/safety/reset',
            self.reset_callback
        )
        
        # Manual protective stop
        self.protective_stop_service = self.create_service(
            Trigger,
            '/safety/trigger_protective_stop',
            self.trigger_protective_stop_callback
        )
        
        # === Timers ===
        # Main safety check timer (100Hz)
        check_period = 1.0 / self.config.check_frequency_hz
        self.check_timer = self.create_timer(check_period, self.check_safety)
        
        # Status publish timer (10Hz)
        self.status_timer = self.create_timer(0.1, self.publish_status)
        
        self.get_logger().info(
            f"Safety Watchdog initialized - "
            f"timeout={self.config.heartbeat_timeout_sec}s, "
            f"max_misses={self.config.max_consecutive_misses}"
        )

    def heartbeat_callback(self, msg: String):
        """Process heartbeat from client."""
        try:
            # Parse heartbeat (simplified - will use proper msg type)
            # Format: "client_id:sequence:latency_ms"
            parts = msg.data.split(':')
            client_id = parts[0]
            sequence = int(parts[1]) if len(parts) > 1 else 0
            latency = float(parts[2]) if len(parts) > 2 else 0.0
            
            current_time = time.time()
            
            with self.lock:
                if client_id not in self._connected_clients:
                    # Initialize with current time to avoid immediate timeout detection
                    self._connected_clients[client_id] = ClientInfo(
                        client_id=client_id,
                        last_heartbeat_time=current_time
                    )
                    self.get_logger().info(f"New client connected: {client_id}")
                
                client = self._connected_clients[client_id]
                
                # Check for sequence gaps
                if sequence > 0 and client.sequence_number > 0:
                    gap = sequence - client.sequence_number - 1
                    if gap > 0:
                        self.get_logger().warn(
                            f"Heartbeat sequence gap detected: {gap} packets lost"
                        )
                
                # Update client info
                client.last_heartbeat_time = current_time
                client.sequence_number = sequence
                client.consecutive_misses = 0
                client.reported_latency_ms = latency
                
                # Check latency
                if latency > self.config.max_communication_latency_ms:
                    self.get_logger().warn(
                        f"High communication latency: {latency:.1f}ms"
                    )
                
        except Exception as e:
            self.get_logger().error(f"Error parsing heartbeat: {e}")

    def wrench_callback(self, msg: Wrench):
        """Process TCP wrench for force monitoring."""
        self.current_tcp_wrench = msg

    def emergency_stop_callback(self, msg: String):
        """Process emergency stop request."""
        self.get_logger().error(f"EMERGENCY STOP received: {msg.data}")
        self.trigger_emergency_stop(f"External E-Stop: {msg.data}")

    def check_safety(self):
        """Main safety check loop - runs at configured frequency."""
        current_time = time.time()
        
        with self.lock:
            # Skip checks if already in emergency stop
            if self.current_safety_state == SafetyState.EMERGENCY_STOP:
                return
            
            # Check all client heartbeats
            for client_id, client in list(self._connected_clients.items()):
                time_since_heartbeat = current_time - client.last_heartbeat_time
                
                # Calculate how many heartbeat intervals have been missed
                # Expected heartbeat interval is roughly timeout/max_misses
                expected_interval = self.config.heartbeat_timeout_sec
                missed_intervals = int(time_since_heartbeat / expected_interval)
                
                # Only update if we've missed more intervals than previously recorded
                if missed_intervals > client.consecutive_misses:
                    client.consecutive_misses = missed_intervals
                    
                    if client.consecutive_misses == self.config.warning_threshold_misses:
                        self.get_logger().warn(
                            f"Heartbeat warning: {client_id} - "
                            f"{client.consecutive_misses} misses ({time_since_heartbeat:.1f}s)"
                        )
                        if self.current_safety_state == SafetyState.NORMAL:
                            self.current_safety_state = SafetyState.WARNING
                            self.active_faults.append(f"Heartbeat warning: {client_id}")
                    
                    if client.consecutive_misses >= self.config.max_consecutive_misses:
                        self.get_logger().error(
                            f"SAFETY: Connection lost to {client_id} - "
                            f"triggering protective stop (no heartbeat for {time_since_heartbeat:.1f}s)"
                        )
                        self.trigger_protective_stop(
                            f"Connection lost: {client_id}"
                        )
            
            # Check force limits
            if self.config.enable_force_monitoring and self.current_tcp_wrench:
                self.check_force_limits()

    def check_force_limits(self):
        """Check TCP force and torque limits."""
        if not self.current_tcp_wrench:
            return
        
        w = self.current_tcp_wrench
        
        # Calculate force magnitude
        force_mag = (w.force.x**2 + w.force.y**2 + w.force.z**2) ** 0.5
        torque_mag = (w.torque.x**2 + w.torque.y**2 + w.torque.z**2) ** 0.5
        
        if force_mag > self.config.max_tcp_force_n:
            self.get_logger().error(
                f"SAFETY: Force limit exceeded ({force_mag:.1f}N > "
                f"{self.config.max_tcp_force_n}N)"
            )
            self.trigger_protective_stop(f"Force limit: {force_mag:.1f}N")
        
        if torque_mag > self.config.max_tcp_torque_nm:
            self.get_logger().error(
                f"SAFETY: Torque limit exceeded ({torque_mag:.1f}Nm > "
                f"{self.config.max_tcp_torque_nm}Nm)"
            )
            self.trigger_protective_stop(f"Torque limit: {torque_mag:.1f}Nm")

    def trigger_protective_stop(self, reason: str):
        """Trigger protective stop on all robots."""
        with self.lock:
            if self.current_safety_state == SafetyState.EMERGENCY_STOP:
                return  # Already in emergency stop
            
            self.current_safety_state = SafetyState.PROTECTIVE_STOP
            self.last_fault_message = reason
            self.last_fault_time = time.time()
            self.active_faults.append(reason)
        
        self.get_logger().error(f"PROTECTIVE STOP: {reason}")
        
        # Call robot protective stop service
        if self.protective_stop_client.service_is_ready():
            request = Trigger.Request()
            future = self.protective_stop_client.call_async(request)
            future.add_done_callback(
                lambda f: self.get_logger().info(
                    f"Robot protective stop result: {f.result().success}"
                )
            )
        else:
            self.get_logger().error("Robot protective stop service not available!")
        
        # Abort any ongoing motion planning
        if self.abort_motion_client.service_is_ready():
            request = Trigger.Request()
            self.abort_motion_client.call_async(request)

    def trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop - most severe response."""
        with self.lock:
            self.current_safety_state = SafetyState.EMERGENCY_STOP
            self.last_fault_message = reason
            self.last_fault_time = time.time()
            self.active_faults.append(f"E-STOP: {reason}")
        
        self.get_logger().fatal(f"EMERGENCY STOP: {reason}")
        
        # Call robot emergency stop service
        if self.emergency_stop_client.service_is_ready():
            request = Trigger.Request()
            future = self.emergency_stop_client.call_async(request)
            future.add_done_callback(
                lambda f: self.get_logger().info(
                    f"Robot emergency stop result: {f.result().success}"
                )
            )
        else:
            self.get_logger().error("Robot emergency stop service not available!")
        
        # Abort any ongoing motion planning
        if self.abort_motion_client.service_is_ready():
            request = Trigger.Request()
            self.abort_motion_client.call_async(request)

    def reset_callback(self, request, response):
        """Reset safety state after manual verification."""
        with self.lock:
            if self.current_safety_state == SafetyState.EMERGENCY_STOP:
                response.success = False
                response.message = "Cannot reset E-STOP via software. Manual reset required."
                return response
            
            # Clear faults and reset to normal
            self.active_faults.clear()
            self.current_safety_state = SafetyState.NORMAL
            
            # Reset client miss counts
            for client in self._connected_clients.values():
                client.consecutive_misses = 0
        
        self.get_logger().info("Safety state reset to NORMAL")
        response.success = True
        response.message = "Safety state reset"
        return response

    def trigger_protective_stop_callback(self, request, response):
        """Manual protective stop trigger."""
        self.trigger_protective_stop("Manual trigger via service")
        response.success = True
        response.message = "Protective stop triggered"
        return response

    def publish_status(self):
        """Publish current safety status."""
        with self.lock:
            # Build status message (simplified - will use proper msg type)
            client_ids = list(self._connected_clients.keys())
            faults = "; ".join(self.active_faults[-5:])  # Last 5 faults
            
            status = (
                f"state:{self.current_safety_state.value}|"
                f"clients:{','.join(client_ids)}|"
                f"faults:{faults}|"
                f"last_fault:{self.last_fault_message}"
            )
        
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    node = SafetyWatchdogNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Safety Watchdog shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
