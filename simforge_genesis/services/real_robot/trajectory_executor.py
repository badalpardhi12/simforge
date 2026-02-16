"""Time-parameterized trajectory execution for UR robots.

This module provides precise trajectory following using the RTDE servoJ command
for time-synchronized motion control. It supports:

- Dense waypoint trajectories from simulation planning
- Time-parameterized execution with configurable velocity scaling
- Velocity-limited trajectory interpolation
- Real-time feedback and monitoring
"""

from __future__ import annotations

import logging
import math
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


class ExecutionState(Enum):
    """Trajectory execution state."""
    IDLE = auto()
    EXECUTING = auto()
    PAUSED = auto()
    STOPPING = auto()
    COMPLETED = auto()
    ERROR = auto()


@dataclass
class TrajectoryExecutionConfig:
    """Configuration for trajectory execution."""
    
    servo_frequency: float = 500.0
    """Control loop frequency in Hz (matches RTDE frequency)."""
    
    lookahead_time: float = 0.1
    """ServoJ lookahead time in seconds [0.03, 0.2]."""
    
    servo_gain: float = 300.0
    """ServoJ proportional gain [100, 2000]."""
    
    velocity_scaling: float = 1.0
    """Velocity scaling factor (0.0 to 1.0)."""
    
    acceleration_scaling: float = 1.0
    """Acceleration scaling factor (0.0 to 1.0)."""
    
    max_joint_velocity: float = 3.14
    """Maximum joint velocity in rad/s (default π rad/s)."""
    
    max_joint_acceleration: float = 6.28
    """Maximum joint acceleration in rad/s² (default 2π rad/s²)."""
    
    blend_radius: float = 0.03
    """Blend radius for moveJ path execution in radians."""
    
    position_tolerance: float = 0.002
    """Joint position tolerance for completion in radians."""
    
    settling_time: float = 0.1
    """Time to wait for robot to settle after trajectory completion."""
    
    interpolation_mode: str = "time"
    """Interpolation mode: 'time' (use waypoint times) or 'linear' (constant velocity)."""


@dataclass
class TrajectoryPoint:
    """Single point in a trajectory."""
    
    positions: Tuple[float, ...]
    """Joint positions in radians."""
    
    time_from_start: float = 0.0
    """Time from trajectory start in seconds."""
    
    velocities: Optional[Tuple[float, ...]] = None
    """Optional: Joint velocities in rad/s."""
    
    accelerations: Optional[Tuple[float, ...]] = None
    """Optional: Joint accelerations in rad/s²."""


@dataclass
class Trajectory:
    """Complete trajectory for execution."""
    
    points: List[TrajectoryPoint]
    """Ordered list of trajectory points."""
    
    total_duration: float = 0.0
    """Total trajectory duration in seconds."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional trajectory metadata."""
    
    @classmethod
    def from_waypoints(
        cls,
        waypoints: Sequence[Sequence[float]],
        times_s: Optional[Sequence[float]] = None,
        duration: Optional[float] = None,
    ) -> "Trajectory":
        """Create trajectory from waypoint list.
        
        Args:
            waypoints: List of joint position arrays (in radians).
            times_s: Optional list of times from start for each waypoint.
            duration: Total duration (used to compute times if times_s not provided).
        
        Returns:
            Constructed Trajectory object.
        """
        if not waypoints:
            return cls(points=[], total_duration=0.0)
        
        n_points = len(waypoints)
        
        # Compute times if not provided
        if times_s is not None:
            times = list(times_s)
        elif duration is not None and n_points > 1:
            times = [i * duration / (n_points - 1) for i in range(n_points)]
        else:
            # Estimate duration based on joint deltas
            times = [0.0]
            for i in range(1, n_points):
                delta = max(abs(a - b) for a, b in zip(waypoints[i-1], waypoints[i]))
                dt = delta / 1.0  # Assume 1 rad/s nominal velocity
                times.append(times[-1] + max(dt, 0.008))  # Minimum 8ms per segment
        
        points = []
        for i, wp in enumerate(waypoints):
            point = TrajectoryPoint(
                positions=tuple(float(x) for x in wp),
                time_from_start=times[i] if i < len(times) else times[-1],
            )
            points.append(point)
        
        total_duration = times[-1] if times else 0.0
        
        return cls(points=points, total_duration=total_duration)
    
    @classmethod
    def from_plan_json(cls, plan_data: Dict[str, Any]) -> "Trajectory":
        """Create trajectory from simforge_genesis plan JSON format.
        
        The plan format contains:
        - waypoints: List of joint position arrays
        - times_s: Optional timing information
        - duration: Total duration
        """
        waypoints = plan_data.get("waypoints", [])
        times_s = plan_data.get("times_s")
        duration = plan_data.get("duration")
        
        return cls.from_waypoints(waypoints, times_s, duration)


class URTrajectoryExecutor:
    """Executes trajectories on UR robots using RTDE.
    
    This executor supports two execution modes:
    
    1. ServoJ mode (default): For dense trajectories with precise timing.
       Uses servoJ for real-time joint position streaming at high frequency.
       Best for: Pre-planned trajectories from simulation, smooth continuous motion.
    
    2. MoveJ mode: For sparse waypoint trajectories.
       Uses moveJ/movej path commands with blending.
       Best for: Simple point-to-point motion, when timing is less critical.
    
    Example:
        >>> from simforge_genesis.services.real_robot import URRobotConnection, URTrajectoryExecutor
        >>> conn = URRobotConnection(ConnectionConfig(robot_ip="192.168.1.9"))
        >>> conn.connect()
        >>> executor = URTrajectoryExecutor(conn, TrajectoryExecutionConfig())
        >>> traj = Trajectory.from_waypoints(waypoints, duration=5.0)
        >>> executor.execute(traj)
        >>> executor.wait_for_completion()
    """
    
    def __init__(
        self,
        connection: Any,  # URRobotConnection
        config: TrajectoryExecutionConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._connection = connection
        self._config = config
        self._logger = logger or logging.getLogger("ur_trajectory")
        
        self._state = ExecutionState.IDLE
        self._lock = threading.RLock()
        
        self._current_trajectory: Optional[Trajectory] = None
        self._current_index: int = 0
        self._start_time: float = 0.0
        
        self._execution_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        
        self._callbacks: List[Callable[[ExecutionState, float], None]] = []
    
    @property
    def state(self) -> ExecutionState:
        """Current execution state."""
        with self._lock:
            return self._state
    
    @property
    def config(self) -> TrajectoryExecutionConfig:
        """Execution configuration."""
        return self._config
    
    @property
    def progress(self) -> float:
        """Current execution progress (0.0 to 1.0)."""
        with self._lock:
            if self._current_trajectory is None:
                return 0.0
            if not self._current_trajectory.points:
                return 1.0
            total = len(self._current_trajectory.points)
            return min(1.0, self._current_index / total)
    
    def execute(
        self,
        trajectory: Trajectory,
        *,
        blocking: bool = False,
        use_servo: bool = True,
    ) -> bool:
        """Start trajectory execution.
        
        Args:
            trajectory: The trajectory to execute.
            blocking: If True, wait for completion before returning.
            use_servo: If True, use servoJ; otherwise use moveJ path.
        
        Returns:
            True if execution started successfully.
        """
        with self._lock:
            if self._state == ExecutionState.EXECUTING:
                self._logger.warning("Already executing a trajectory")
                return False
            
            if not self._connection.is_connected:
                self._logger.error("Robot not connected")
                return False
            
            if not trajectory.points:
                self._logger.warning("Empty trajectory")
                return True
            
            self._current_trajectory = trajectory
            self._current_index = 0
            self._state = ExecutionState.EXECUTING
        
        self._stop_event.clear()
        self._pause_event.set()  # Not paused
        
        if use_servo:
            self._execution_thread = threading.Thread(
                target=self._execute_servo_loop,
                daemon=True,
                name="ur_servo_exec",
            )
        else:
            self._execution_thread = threading.Thread(
                target=self._execute_movej_path,
                daemon=True,
                name="ur_movej_exec",
            )
        
        self._execution_thread.start()
        
        if blocking:
            self.wait_for_completion()
        
        return True
    
    def execute_waypoints(
        self,
        waypoints: Sequence[Sequence[float]],
        *,
        duration: Optional[float] = None,
        times_s: Optional[Sequence[float]] = None,
        blocking: bool = True,
        use_servo: bool = True,
    ) -> bool:
        """Execute trajectory from raw waypoints.
        
        Args:
            waypoints: List of joint positions in radians.
            duration: Total execution duration in seconds.
            times_s: Optional specific times for each waypoint.
            blocking: Wait for completion.
            use_servo: Use servoJ vs moveJ.
        
        Returns:
            True if execution successful.
        """
        trajectory = Trajectory.from_waypoints(waypoints, times_s, duration)
        return self.execute(trajectory, blocking=blocking, use_servo=use_servo)
    
    def stop(self, deceleration: float = 2.0) -> None:
        """Stop current trajectory execution.
        
        Args:
            deceleration: Deceleration rate in rad/s².
        """
        self._stop_event.set()
        
        with self._lock:
            if self._state == ExecutionState.EXECUTING:
                self._state = ExecutionState.STOPPING
        
        # Command robot to stop
        if self._connection.is_connected and self._connection.control_interface:
            try:
                self._connection.control_interface.servoStop(deceleration)
            except Exception as exc:
                self._logger.warning("Stop command failed: %s", exc)
    
    def pause(self) -> None:
        """Pause trajectory execution."""
        self._pause_event.clear()
        with self._lock:
            if self._state == ExecutionState.EXECUTING:
                self._state = ExecutionState.PAUSED
        
        # Stop current motion
        if self._connection.is_connected and self._connection.control_interface:
            try:
                self._connection.control_interface.servoStop(2.0)
            except Exception:
                pass
    
    def resume(self) -> None:
        """Resume paused trajectory execution."""
        self._pause_event.set()
        with self._lock:
            if self._state == ExecutionState.PAUSED:
                self._state = ExecutionState.EXECUTING
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for trajectory execution to complete.
        
        Args:
            timeout: Maximum time to wait in seconds.
        
        Returns:
            True if completed successfully, False if timed out or error.
        """
        if self._execution_thread is None:
            return True
        
        self._execution_thread.join(timeout=timeout)
        
        return self._state == ExecutionState.COMPLETED
    
    def register_callback(self, callback: Callable[[ExecutionState, float], None]) -> None:
        """Register callback for execution state changes.
        
        Callback receives (state, progress) arguments.
        """
        self._callbacks.append(callback)
    
    def _execute_servo_loop(self) -> None:
        """ServoJ-based trajectory execution loop.
        
        This provides precise time-synchronized trajectory following
        by streaming joint positions at the control frequency.
        """
        rtde_c = self._connection.control_interface
        rtde_r = self._connection.receive_interface
        
        if rtde_c is None or rtde_r is None:
            self._set_state(ExecutionState.ERROR)
            return
        
        trajectory = self._current_trajectory
        if trajectory is None or not trajectory.points:
            self._set_state(ExecutionState.COMPLETED)
            return
        
        dt = 1.0 / self._config.servo_frequency
        velocity_scale = max(0.01, self._config.velocity_scaling)  # Ensure non-zero
        
        # Calculate actual execution duration (slower = longer duration)
        execution_duration = trajectory.total_duration / velocity_scale
        
        self._start_time = time.time()
        
        self._logger.info(
            "Starting servo execution: %d points, %.2fs trajectory, %.2fs execution time",
            len(trajectory.points),
            trajectory.total_duration,
            execution_duration,
        )
        
        loop_count = 0
        max_loops = int(execution_duration * self._config.servo_frequency * 1.5)  # Safety limit
        
        try:
            while not self._stop_event.is_set():
                loop_count += 1
                
                # Safety check to prevent infinite loop
                if loop_count > max_loops:
                    self._logger.warning("Max loop count reached, ending trajectory")
                    break
                
                # Handle pause
                if not self._pause_event.is_set():
                    time.sleep(0.05)
                    continue
                
                t_start = rtde_c.initPeriod()
                
                # Calculate elapsed real time
                elapsed_real = time.time() - self._start_time
                
                # Check if we've exceeded execution time
                if elapsed_real >= execution_duration:
                    self._logger.debug("Execution time reached: %.2fs", elapsed_real)
                    break
                
                # Map real elapsed time to trajectory time
                # If velocity_scale=0.5, we go through trajectory at half speed
                # So trajectory_time = elapsed_real * velocity_scale
                trajectory_time = elapsed_real * velocity_scale
                
                # Find current target position by interpolating trajectory
                target_q = self._interpolate_trajectory(trajectory_time)
                
                if target_q is None:
                    self._logger.warning("Interpolation returned None at t=%.3f", trajectory_time)
                    break
                
                # Execute servoJ command
                try:
                    rtde_c.servoJ(
                        list(target_q),
                        0.0,  # velocity (not used)
                        0.0,  # acceleration (not used)
                        dt,
                        self._config.lookahead_time,
                        self._config.servo_gain,
                    )
                except Exception as servo_exc:
                    self._logger.error("servoJ failed: %s", servo_exc)
                    break
                
                # Update progress
                progress = min(1.0, elapsed_real / execution_duration)
                self._notify_callbacks(progress)
                
                # Log progress periodically
                if loop_count % 500 == 0:  # Every ~1 second at 500Hz
                    self._logger.debug(
                        "Servo progress: %.1f%% (loop %d, elapsed %.2fs)",
                        progress * 100,
                        loop_count,
                        elapsed_real,
                    )
                
                rtde_c.waitPeriod(t_start)
            
            self._logger.info("Servo loop completed after %d iterations", loop_count)
            
            # Stop servo mode
            try:
                rtde_c.servoStop(2.0)
            except Exception as stop_exc:
                self._logger.warning("servoStop failed: %s", stop_exc)
            
            # Wait for settling
            time.sleep(self._config.settling_time)
            
            # Verify final position
            if trajectory.points:
                final_target = trajectory.points[-1].positions
                try:
                    actual_q = tuple(rtde_r.getActualQ())
                    max_error = max(abs(a - t) for a, t in zip(actual_q, final_target))
                    
                    if max_error > self._config.position_tolerance:
                        self._logger.warning(
                            "Final position error: %.4f rad (tolerance: %.4f)",
                            max_error,
                            self._config.position_tolerance,
                        )
                        # Do a final correction move
                        try:
                            rtde_c.moveJ(list(final_target), 0.5, 0.5)
                        except Exception:
                            pass
                except Exception as verify_exc:
                    self._logger.warning("Could not verify final position: %s", verify_exc)
            
            self._set_state(ExecutionState.COMPLETED)
            
        except Exception as exc:
            self._logger.error("Servo execution error: %s", exc)
            try:
                rtde_c.servoStop(5.0)
            except Exception:
                pass
            self._set_state(ExecutionState.ERROR)
    
    def _execute_movej_path(self) -> None:
        """MoveJ path-based trajectory execution.
        
        For dense simulation trajectories (many waypoints), this method:
        1. Decimates the trajectory to keep only key waypoints
        2. Uses appropriate blend radius for smooth motion
        3. Executes asynchronously with progress monitoring to avoid connection timeout
        
        This approach provides smooth motion while maintaining trajectory accuracy.
        """
        rtde_c = self._connection.control_interface
        
        if rtde_c is None:
            self._set_state(ExecutionState.ERROR)
            return
        
        trajectory = self._current_trajectory
        if trajectory is None or not trajectory.points:
            self._set_state(ExecutionState.COMPLETED)
            return
        
        # Build optimized path
        vel = self._config.max_joint_velocity * self._config.velocity_scaling
        acc = self._config.max_joint_acceleration * self._config.acceleration_scaling
        
        # Decimate dense trajectories for smooth moveJ execution
        # Keep only significant waypoints based on angular change threshold
        decimated_points = self._decimate_trajectory(trajectory.points)
        
        # Determine blend radius: 
        # - Use configured blend for intermediate points (smooth motion)
        # - Zero blend for final point (precise stopping)
        blend = self._config.blend_radius if self._config.blend_radius > 0 else 0.05  # Default 0.05 rad (~3°) for smoothness
        
        path = []
        for i, point in enumerate(decimated_points):
            # Last point has no blend to ensure precise final position
            r = 0.0 if i == len(decimated_points) - 1 else blend
            
            # Path format: [q1, q2, q3, q4, q5, q6, vel, acc, blend]
            path_entry = list(point.positions) + [vel, acc, r]
            path.append(path_entry)
        
        self._logger.info(
            "Starting moveJ path: %d points (decimated from %d), vel=%.2f, acc=%.2f, blend=%.3f",
            len(path),
            len(trajectory.points),
            vel,
            acc,
            blend,
        )
        
        try:
            # Execute asynchronously to avoid blocking and connection timeout
            success = rtde_c.moveJ(path, True)  # asynchronous=True
            
            if not success:
                self._logger.error("moveJ path failed to start")
                self._set_state(ExecutionState.ERROR)
                return
            
            # Monitor progress until completion
            success = self._wait_for_async_completion(rtde_c)
            
            if success:
                # Wait for settling
                time.sleep(self._config.settling_time)
                self._set_state(ExecutionState.COMPLETED)
            else:
                self._set_state(ExecutionState.ERROR)
            
        except Exception as exc:
            self._logger.error("moveJ path error: %s", exc)
            self._set_state(ExecutionState.ERROR)

    def _decimate_trajectory(
        self,
        points: List[TrajectoryPoint],
        angle_threshold: float = 0.02,  # ~1.1° minimum change to keep point
        max_points: int = 50,  # Maximum points for moveJ path
    ) -> List[TrajectoryPoint]:
        """Decimate trajectory to key waypoints for smooth moveJ execution.
        
        Dense trajectories (e.g., 400 points from simulation) cause jittery motion
        with moveJ because the robot tries to stop at each point. This method
        keeps only significant waypoints based on angular change, ensuring smooth
        blended motion.
        
        Args:
            points: Original trajectory points
            angle_threshold: Minimum angular change (rad) to keep a point
            max_points: Maximum number of output points
            
        Returns:
            Decimated list of key waypoints
        """
        if len(points) <= max_points:
            # If already sparse enough, keep all points
            return points
        
        if len(points) <= 2:
            return points
        
        # Always keep first point
        result = [points[0]]
        
        # Calculate cumulative distance for uniform sampling
        cumulative_dist = [0.0]
        for i in range(1, len(points)):
            delta = sum(
                abs(a - b) for a, b in 
                zip(points[i].positions, points[i-1].positions)
            )
            cumulative_dist.append(cumulative_dist[-1] + delta)
        
        total_dist = cumulative_dist[-1]
        if total_dist < 1e-6:
            # No movement, just return first and last
            return [points[0], points[-1]]
        
        # Target spacing: aim for max_points - 2 intermediate points
        # (excluding first and last which are always included)
        target_points = min(max_points, max(10, len(points) // 8))
        target_spacing = total_dist / (target_points - 1)
        
        last_dist = 0.0
        last_point = points[0]
        
        for i in range(1, len(points) - 1):
            dist_from_last = cumulative_dist[i] - last_dist
            
            # Check if this point is significant
            angular_change = max(
                abs(a - b) for a, b in 
                zip(points[i].positions, last_point.positions)
            )
            
            # Keep point if:
            # 1. Distance exceeds target spacing, OR
            # 2. Angular change exceeds threshold
            if dist_from_last >= target_spacing or angular_change >= angle_threshold:
                result.append(points[i])
                last_dist = cumulative_dist[i]
                last_point = points[i]
        
        # Always keep last point for precise final position
        result.append(points[-1])
        
        self._logger.debug(
            "Decimated trajectory: %d -> %d points (threshold=%.3f rad, target=%d)",
            len(points),
            len(result),
            angle_threshold,
            target_points,
        )
        
        return result

    def _wait_for_async_completion(
        self,
        rtde_c,
        timeout: float = 120.0,
        poll_interval: float = 0.1,
    ) -> bool:
        """Wait for async moveJ operation to complete.
        
        Monitors the robot state to detect when motion has completed.
        This avoids the connection timeout issues with blocking moveJ calls.
        
        Args:
            rtde_c: RTDE control interface
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks
            
        Returns:
            True if completed successfully, False on error/timeout
        """
        rtde_r = self._connection.receive_interface
        start_time = time.time()
        
        # Wait a short time for motion to start
        time.sleep(0.05)
        
        while time.time() - start_time < timeout:
            # Check for stop request
            if self._stop_event.is_set():
                self._logger.info("Stop requested during async move")
                try:
                    rtde_c.stopJ(2.0)
                except Exception:
                    pass
                return False
            
            try:
                # Check if robot is steady (motion complete)
                if rtde_c.isSteady():
                    return True
                
                # Also check async operation progress
                progress = rtde_c.getAsyncOperationProgress()
                if progress < 0:
                    # Negative progress means operation completed
                    return True
                    
            except Exception as exc:
                self._logger.warning("Error checking motion status: %s", exc)
                # Try to reconnect if connection lost
                if not rtde_c.isConnected():
                    self._logger.warning("Connection lost during async move, attempting reconnect")
                    try:
                        rtde_c.reconnect()
                        time.sleep(0.5)
                    except Exception as reconn_exc:
                        self._logger.error("Reconnection failed: %s", reconn_exc)
                        return False
            
            time.sleep(poll_interval)
        
        self._logger.error("Async move timed out after %.1fs", timeout)
        try:
            rtde_c.stopJ(2.0)
        except Exception:
            pass
        return False
    
    def _interpolate_trajectory(self, elapsed: float) -> Optional[Tuple[float, ...]]:
        """Interpolate trajectory at given time.
        
        Args:
            elapsed: Time elapsed since trajectory start.
        
        Returns:
            Interpolated joint positions, or None if past trajectory end.
        """
        trajectory = self._current_trajectory
        if trajectory is None or not trajectory.points:
            return None
        
        # Check if past end
        if elapsed >= trajectory.total_duration:
            return trajectory.points[-1].positions
        
        # Find surrounding points
        prev_point = trajectory.points[0]
        next_point = trajectory.points[0]
        
        for i, point in enumerate(trajectory.points):
            if point.time_from_start >= elapsed:
                next_point = point
                if i > 0:
                    prev_point = trajectory.points[i - 1]
                break
            prev_point = point
            next_point = point
        
        # Linear interpolation between points
        t0 = prev_point.time_from_start
        t1 = next_point.time_from_start
        
        if t1 <= t0:
            return next_point.positions
        
        alpha = (elapsed - t0) / (t1 - t0)
        alpha = max(0.0, min(1.0, alpha))
        
        interpolated = tuple(
            (1.0 - alpha) * p0 + alpha * p1
            for p0, p1 in zip(prev_point.positions, next_point.positions)
        )
        
        return interpolated
    
    def _set_state(self, new_state: ExecutionState) -> None:
        """Update execution state."""
        with self._lock:
            old_state = self._state
            self._state = new_state
            if old_state != new_state:
                self._logger.debug("Execution state: %s -> %s", old_state.name, new_state.name)
    
    def _notify_callbacks(self, progress: float) -> None:
        """Notify callbacks of progress."""
        state = self._state
        for cb in self._callbacks:
            try:
                cb(state, progress)
            except Exception:
                pass


__all__ = [
    "URTrajectoryExecutor",
    "TrajectoryExecutionConfig",
    "Trajectory",
    "TrajectoryPoint",
    "ExecutionState",
]
