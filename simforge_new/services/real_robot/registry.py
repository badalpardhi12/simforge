"""Multi-robot registry and coordination for UR robots.

This module provides management of multiple UR robot connections,
enabling coordinated control of robot cells with multiple arms.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from .driver import URRobotDriver, RobotDriverConfig
from .connection import ConnectionState


@dataclass
class RobotRegistration:
    """Registration entry for a robot in the registry."""
    
    name: str
    """Robot instance name."""
    
    config: RobotDriverConfig
    """Robot configuration."""
    
    driver: Optional[URRobotDriver] = None
    """Active driver instance (None if not connected)."""
    
    enabled: bool = True
    """Whether robot is enabled for operations."""


class URRobotRegistry:
    """Registry for managing multiple UR robot connections.
    
    This class provides:
    - Registration and discovery of robots from config
    - Coordinated connection/disconnection
    - Robot lookup by name
    - Health monitoring across all robots
    
    Example:
        >>> registry = URRobotRegistry()
        >>> registry.register_from_env_preset(preset_config)
        >>> registry.connect_all()
        >>> for name, driver in registry.connected_robots():
        ...     print(f"{name}: {driver.get_joint_positions(degrees=True)}")
        >>> registry.disconnect_all()
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger("ur_registry")
        self._robots: Dict[str, RobotRegistration] = {}
        self._lock = threading.RLock()
    
    @property
    def robot_names(self) -> List[str]:
        """List of all registered robot names."""
        with self._lock:
            return list(self._robots.keys())
    
    @property
    def connected_count(self) -> int:
        """Number of currently connected robots."""
        with self._lock:
            return sum(
                1 for reg in self._robots.values()
                if reg.driver and reg.driver.is_connected
            )
    
    def register(
        self,
        name: str,
        config: RobotDriverConfig,
        *,
        enabled: bool = True,
    ) -> None:
        """Register a robot with the registry.
        
        Args:
            name: Unique name for this robot instance.
            config: Robot driver configuration.
            enabled: Whether robot is enabled for operations.
        """
        with self._lock:
            if name in self._robots:
                self._logger.warning("Robot '%s' already registered, updating", name)
            
            self._robots[name] = RobotRegistration(
                name=name,
                config=config,
                enabled=enabled,
            )
            
            self._logger.info(
                "Registered robot '%s' at %s (model=%s, enabled=%s)",
                name,
                config.robot_ip,
                config.robot_model,
                enabled,
            )
    
    def register_from_preset(
        self,
        preset_data: Dict,
        *,
        ip_mapping: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Register robots from an environment preset configuration.
        
        Args:
            preset_data: Parsed preset YAML data.
            ip_mapping: Optional mapping of robot names to IP addresses.
                       If not provided, looks for 'real_robot_ip' in robot metadata.
        
        Returns:
            List of registered robot names.
        """
        registered = []
        robots = preset_data.get("robots", [])
        
        for robot_spec in robots:
            name = robot_spec.get("name", "unnamed_robot")
            
            # Get IP address
            ip = None
            if ip_mapping:
                ip = ip_mapping.get(name)
            
            # Check metadata for IP
            if not ip:
                metadata = robot_spec.get("metadata", {})
                ip = metadata.get("real_robot_ip")
            
            if not ip:
                self._logger.info("Robot '%s' has no IP configured, skipping", name)
                continue
            
            # Determine robot model from URDF path
            urdf_path = robot_spec.get("urdf", "")
            model = self._infer_robot_model(urdf_path)
            
            # Get velocity/acceleration from control settings
            control = robot_spec.get("control", {})
            limits = control.get("limits", {})
            
            config = RobotDriverConfig(
                robot_ip=ip,
                robot_name=name,
                robot_model=model,
                max_joint_velocity=limits.get("joint_velocity", 1.05),
                max_joint_acceleration=limits.get("joint_acceleration", 1.4),
                velocity_scaling=limits.get("cartesian_velocity", 0.1) * 10,  # Rough scaling
            )
            
            # Check for tool settings
            tool = robot_spec.get("tool", {})
            if tool:
                attach_pose = tool.get("attach_pose")
                if attach_pose and len(attach_pose) >= 6:
                    # Convert attach pose to TCP offset
                    config.tcp_offset = tuple(attach_pose[:6])
            
            self.register(name, config)
            registered.append(name)
        
        return registered
    
    def unregister(self, name: str) -> bool:
        """Unregister a robot from the registry.
        
        Args:
            name: Robot name to unregister.
        
        Returns:
            True if robot was found and unregistered.
        """
        with self._lock:
            if name not in self._robots:
                return False
            
            reg = self._robots[name]
            if reg.driver and reg.driver.is_connected:
                reg.driver.disconnect()
            
            del self._robots[name]
            self._logger.info("Unregistered robot '%s'", name)
            return True
    
    def get(self, name: str) -> Optional[URRobotDriver]:
        """Get a robot driver by name.
        
        Args:
            name: Robot name.
        
        Returns:
            Driver instance if connected, None otherwise.
        """
        with self._lock:
            reg = self._robots.get(name)
            if reg and reg.driver:
                return reg.driver
            return None
    
    def connect(self, name: str) -> bool:
        """Connect to a specific robot.
        
        Args:
            name: Robot name.
        
        Returns:
            True if connection successful.
        """
        with self._lock:
            reg = self._robots.get(name)
            if not reg:
                self._logger.error("Robot '%s' not registered", name)
                return False
            
            if not reg.enabled:
                self._logger.info("Robot '%s' is disabled, skipping", name)
                return False
            
            if reg.driver and reg.driver.is_connected:
                self._logger.info("Robot '%s' already connected", name)
                return True
            
            driver = URRobotDriver(
                reg.config,
                self._logger.getChild(name),
            )
            
            if driver.connect():
                reg.driver = driver
                return True
            
            return False
    
    def disconnect(self, name: str) -> None:
        """Disconnect from a specific robot.
        
        Args:
            name: Robot name.
        """
        with self._lock:
            reg = self._robots.get(name)
            if reg and reg.driver:
                reg.driver.disconnect()
                reg.driver = None
    
    def connect_all(self, parallel: bool = False) -> Dict[str, bool]:
        """Connect to all registered and enabled robots.
        
        Args:
            parallel: If True, attempt connections in parallel.
        
        Returns:
            Dict mapping robot names to connection success.
        """
        results = {}
        
        if parallel:
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self.connect, name): name
                    for name in self.robot_names
                }
                for future in concurrent.futures.as_completed(futures):
                    name = futures[future]
                    try:
                        results[name] = future.result()
                    except Exception as exc:
                        self._logger.error("Connect failed for %s: %s", name, exc)
                        results[name] = False
        else:
            for name in self.robot_names:
                results[name] = self.connect(name)
        
        return results
    
    def disconnect_all(self) -> None:
        """Disconnect from all robots."""
        for name in self.robot_names:
            self.disconnect(name)
    
    def connected_robots(self) -> Iterator[Tuple[str, URRobotDriver]]:
        """Iterate over connected robots.
        
        Yields:
            Tuples of (name, driver) for connected robots.
        """
        with self._lock:
            for name, reg in self._robots.items():
                if reg.driver and reg.driver.is_connected:
                    yield name, reg.driver
    
    def get_status(self) -> Dict[str, Dict]:
        """Get status of all registered robots.
        
        Returns:
            Dict mapping robot names to status info.
        """
        status = {}
        
        with self._lock:
            for name, reg in self._robots.items():
                robot_status = {
                    "ip": reg.config.robot_ip,
                    "model": reg.config.robot_model,
                    "enabled": reg.enabled,
                    "connected": False,
                    "ready": False,
                }
                
                if reg.driver:
                    robot_status["connected"] = reg.driver.is_connected
                    if reg.driver.is_connected:
                        robot_status["ready"] = reg.driver.is_ready()
                        state = reg.driver.robot_state
                        robot_status["joint_positions_deg"] = [
                            round(q * 180 / 3.14159, 1) for q in state.actual_q
                        ]
                
                status[name] = robot_status
        
        return status
    
    def _infer_robot_model(self, urdf_path: str) -> str:
        """Infer robot model from URDF path."""
        path_lower = urdf_path.lower()
        
        if "ur3" in path_lower:
            return "UR3e"
        elif "ur5" in path_lower:
            return "UR5e"
        elif "ur10" in path_lower:
            return "UR10e"
        elif "ur16" in path_lower:
            return "UR16e"
        elif "ur20" in path_lower:
            return "UR20"
        
        return "UR5e"  # Default
    
    def __enter__(self) -> "URRobotRegistry":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect_all()


__all__ = ["URRobotRegistry", "RobotRegistration"]
