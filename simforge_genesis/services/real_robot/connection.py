"""Low-level RTDE connection management for UR robots.

This module provides robust connection handling for Universal Robots
using the ur_rtde library for Real-Time Data Exchange protocol.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import rtde_control
import rtde_receive


class ConnectionState(Enum):
    """Connection state machine states."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    ERROR = auto()
    RECONNECTING = auto()


@dataclass
class ConnectionConfig:
    """Configuration for UR robot RTDE connection."""
    
    robot_ip: str
    """IP address of the UR robot controller."""
    
    rtde_frequency: float = 500.0
    """RTDE update frequency in Hz (max 500Hz for e-Series, 125Hz for CB3)."""
    
    use_external_control_cap: bool = False
    """Whether to use the ExternalControl URCap (requires URCap installation)."""
    
    connection_timeout: float = 10.0
    """Timeout for initial connection in seconds."""
    
    reconnect_attempts: int = 3
    """Number of reconnection attempts before giving up."""
    
    reconnect_delay: float = 1.0
    """Delay between reconnection attempts in seconds."""
    
    custom_script: bool = False
    """Whether using a custom control script on the robot."""


@dataclass
class RobotState:
    """Current state snapshot from the robot."""
    
    timestamp: float = 0.0
    """Timestamp of this state sample."""
    
    actual_q: Tuple[float, ...] = field(default_factory=lambda: (0.0,) * 6)
    """Actual joint positions in radians."""
    
    actual_qd: Tuple[float, ...] = field(default_factory=lambda: (0.0,) * 6)
    """Actual joint velocities in rad/s."""
    
    target_q: Tuple[float, ...] = field(default_factory=lambda: (0.0,) * 6)
    """Target joint positions in radians."""
    
    target_qd: Tuple[float, ...] = field(default_factory=lambda: (0.0,) * 6)
    """Target joint velocities in rad/s."""
    
    actual_tcp_pose: Tuple[float, ...] = field(default_factory=lambda: (0.0,) * 6)
    """Actual TCP pose [x, y, z, rx, ry, rz] in meters and radians."""
    
    actual_tcp_speed: Tuple[float, ...] = field(default_factory=lambda: (0.0,) * 6)
    """Actual TCP speed [vx, vy, vz, wx, wy, wz]."""
    
    robot_mode: int = 0
    """Current robot mode (see UR documentation)."""
    
    safety_status: int = 0
    """Current safety status."""
    
    program_running: bool = False
    """Whether a program is currently running on the controller."""
    
    protective_stopped: bool = False
    """Whether robot is in protective stop."""
    
    emergency_stopped: bool = False
    """Whether robot is in emergency stop."""


class URRobotConnection:
    """Manages RTDE connection to a Universal Robot.
    
    This class handles:
    - Connection establishment and teardown
    - Automatic reconnection on connection loss
    - Thread-safe state reading
    - Robot mode/safety monitoring
    
    Example:
        >>> config = ConnectionConfig(robot_ip="192.168.1.9")
        >>> conn = URRobotConnection(config)
        >>> conn.connect()
        >>> state = conn.get_state()
        >>> print(f"Joint positions: {state.actual_q}")
        >>> conn.disconnect()
    """
    
    def __init__(
        self,
        config: ConnectionConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config
        self._logger = logger or logging.getLogger(f"ur_connection.{config.robot_ip}")
        
        self._state = ConnectionState.DISCONNECTED
        self._lock = threading.RLock()
        
        self._rtde_c: Optional[rtde_control.RTDEControlInterface] = None
        self._rtde_r: Optional[rtde_receive.RTDEReceiveInterface] = None
        
        self._state_callbacks: List[Callable[[RobotState], None]] = []
        self._connection_callbacks: List[Callable[[ConnectionState], None]] = []
        
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._latest_state: Optional[RobotState] = None
    
    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        with self._lock:
            return self._state
    
    @property
    def config(self) -> ConnectionConfig:
        """Connection configuration."""
        return self._config
    
    @property
    def is_connected(self) -> bool:
        """Whether connection is active and healthy."""
        return self.state == ConnectionState.CONNECTED
    
    @property
    def control_interface(self) -> Optional[rtde_control.RTDEControlInterface]:
        """Direct access to RTDE control interface (use with caution)."""
        return self._rtde_c
    
    @property
    def receive_interface(self) -> Optional[rtde_receive.RTDEReceiveInterface]:
        """Direct access to RTDE receive interface."""
        return self._rtde_r
    
    def connect(self) -> bool:
        """Establish connection to the robot.
        
        Returns:
            True if connection successful, False otherwise.
        """
        with self._lock:
            if self._state == ConnectionState.CONNECTED:
                self._logger.info("Already connected to %s", self._config.robot_ip)
                return True
            
            self._set_state(ConnectionState.CONNECTING)
        
        try:
            self._logger.info(
                "Connecting to UR robot at %s (freq=%.0fHz)",
                self._config.robot_ip,
                self._config.rtde_frequency,
            )
            
            # Build control interface flags
            flags = 0
            if self._config.use_external_control_cap:
                flags |= rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP
            if self._config.custom_script:
                flags |= rtde_control.RTDEControlInterface.FLAG_CUSTOM_SCRIPT
            
            # Connect receive interface first (faster, validates IP)
            self._rtde_r = rtde_receive.RTDEReceiveInterface(
                self._config.robot_ip,
                self._config.rtde_frequency,
            )
            
            # Connect control interface
            if flags:
                self._rtde_c = rtde_control.RTDEControlInterface(
                    self._config.robot_ip,
                    self._config.rtde_frequency,
                    flags,
                )
            else:
                self._rtde_c = rtde_control.RTDEControlInterface(
                    self._config.robot_ip,
                    self._config.rtde_frequency,
                )
            
            with self._lock:
                self._set_state(ConnectionState.CONNECTED)
            
            self._logger.info("Connected to UR robot at %s", self._config.robot_ip)
            
            # Start state monitoring thread
            self._start_monitoring()
            
            return True
            
        except Exception as exc:
            self._logger.error("Connection failed: %s", exc)
            self._cleanup_interfaces()
            with self._lock:
                self._set_state(ConnectionState.ERROR)
            return False
    
    def disconnect(self) -> None:
        """Gracefully disconnect from the robot."""
        self._logger.info("Disconnecting from %s", self._config.robot_ip)
        
        self._stop_monitoring.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        
        self._cleanup_interfaces()
        
        with self._lock:
            self._set_state(ConnectionState.DISCONNECTED)
    
    def get_state(self) -> RobotState:
        """Get current robot state snapshot.
        
        Returns:
            Current robot state. Returns cached state if available,
            otherwise fetches fresh state.
        """
        if self._latest_state is not None:
            return self._latest_state
        
        return self._fetch_state()
    
    def _fetch_state(self) -> RobotState:
        """Fetch fresh state from robot."""
        if not self._rtde_r:
            return RobotState()
        
        try:
            state = RobotState(
                timestamp=time.time(),
                actual_q=tuple(self._rtde_r.getActualQ()),
                actual_qd=tuple(self._rtde_r.getActualQd()),
                target_q=tuple(self._rtde_r.getTargetQ()),
                target_qd=tuple(self._rtde_r.getTargetQd()),
                actual_tcp_pose=tuple(self._rtde_r.getActualTCPPose()),
                actual_tcp_speed=tuple(self._rtde_r.getActualTCPSpeed()),
                robot_mode=self._rtde_r.getRobotMode(),
                safety_status=self._rtde_r.getSafetyStatusBits(),
                program_running=self._rtde_r.isConnected(),
                protective_stopped=self._rtde_r.isProtectiveStopped(),
                emergency_stopped=self._rtde_r.isEmergencyStopped(),
            )
            return state
        except Exception as exc:
            self._logger.warning("Failed to fetch state: %s", exc)
            return RobotState()
    
    def register_state_callback(self, callback: Callable[[RobotState], None]) -> None:
        """Register callback for state updates."""
        self._state_callbacks.append(callback)
    
    def register_connection_callback(self, callback: Callable[[ConnectionState], None]) -> None:
        """Register callback for connection state changes."""
        self._connection_callbacks.append(callback)
    
    def _set_state(self, new_state: ConnectionState) -> None:
        """Update connection state and notify callbacks."""
        old_state = self._state
        self._state = new_state
        if old_state != new_state:
            self._logger.debug("Connection state: %s -> %s", old_state.name, new_state.name)
            for cb in self._connection_callbacks:
                try:
                    cb(new_state)
                except Exception as exc:
                    self._logger.warning("Connection callback failed: %s", exc)
    
    def _cleanup_interfaces(self) -> None:
        """Clean up RTDE interfaces."""
        if self._rtde_c:
            try:
                self._rtde_c.stopScript()
            except Exception:
                pass
            try:
                self._rtde_c.disconnect()
            except Exception:
                pass
            self._rtde_c = None
        
        if self._rtde_r:
            try:
                self._rtde_r.disconnect()
            except Exception:
                pass
            self._rtde_r = None
    
    def _start_monitoring(self) -> None:
        """Start background state monitoring thread."""
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name=f"ur_monitor_{self._config.robot_ip}",
        )
        self._monitor_thread.start()
    
    def _monitoring_loop(self) -> None:
        """Background thread for continuous state monitoring."""
        interval = 1.0 / min(self._config.rtde_frequency, 125.0)  # Cap monitoring rate
        
        while not self._stop_monitoring.is_set():
            try:
                state = self._fetch_state()
                self._latest_state = state
                
                for cb in self._state_callbacks:
                    try:
                        cb(state)
                    except Exception as exc:
                        self._logger.debug("State callback error: %s", exc)
                
            except Exception as exc:
                self._logger.warning("Monitor loop error: %s", exc)
                if not self._check_connection_health():
                    self._attempt_reconnect()
            
            time.sleep(interval)
    
    def _check_connection_health(self) -> bool:
        """Check if connection is still healthy."""
        try:
            if self._rtde_r and self._rtde_r.isConnected():
                return True
        except Exception:
            pass
        return False
    
    def _attempt_reconnect(self) -> bool:
        """Attempt to reconnect to the robot."""
        with self._lock:
            if self._state == ConnectionState.RECONNECTING:
                return False
            self._set_state(ConnectionState.RECONNECTING)
        
        self._cleanup_interfaces()
        
        for attempt in range(self._config.reconnect_attempts):
            self._logger.info(
                "Reconnection attempt %d/%d",
                attempt + 1,
                self._config.reconnect_attempts,
            )
            time.sleep(self._config.reconnect_delay)
            
            if self.connect():
                return True
        
        with self._lock:
            self._set_state(ConnectionState.ERROR)
        return False
    
    def __enter__(self) -> "URRobotConnection":
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()


__all__ = ["URRobotConnection", "ConnectionConfig", "ConnectionState", "RobotState"]
