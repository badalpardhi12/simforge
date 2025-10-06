"""Synchronized trajectory execution that integrates with simulation loop timing."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..core.trajectories import JointTrajectory


@dataclass
class ActiveTrajectory:
    """A trajectory with timing information for synchronous execution."""
    waypoints: np.ndarray  # [N, dof] array of joint positions
    times: np.ndarray      # [N] array of time values
    start_time: float      # When trajectory execution started (time.time())
    
    def sample(self, current_time: float) -> Dict[str, Any]:
        """Sample trajectory at current time (compatible with legacy ActiveTrajectory)."""
        elapsed = current_time - self.start_time
        
        if elapsed >= self.times[-1]:
            return {"q": self.waypoints[-1], "done": True}
        
        if elapsed <= self.times[0]:
            return {"q": self.waypoints[0], "done": False}
        
        # Find the segment we're in
        idx = np.searchsorted(self.times, elapsed) - 1
        if idx < 0:
            return {"q": self.waypoints[0], "done": False}
        if idx >= len(self.times) - 1:
            return {"q": self.waypoints[-1], "done": True}
        
        # Linear interpolation between waypoints (matching legacy behavior)
        t0, t1 = self.times[idx], self.times[idx + 1]
        if t1 <= t0:
            return {"q": self.waypoints[idx + 1], "done": False}
        a = (elapsed - t0) / (t1 - t0)
        q = (1.0 - a) * self.waypoints[idx] + a * self.waypoints[idx + 1]
        return {"q": q, "done": False}


class SynchronizedTrajectoryManager:
    """Manages active trajectories that are executed synchronously with the simulation loop."""
    
    def __init__(self, logger=None):
        self._active_trajectories: Dict[str, ActiveTrajectory] = {}
        self._final_positions: Dict[str, np.ndarray] = {}  # Position holding after completion
        self._recent_trajectories: Dict[str, deque[Dict[str, Any]]] = defaultdict(deque)
        self._logger = logger
    
    def start_trajectory(self, robot_name: str, trajectory: JointTrajectory) -> None:
        """Start a new trajectory for the given robot.
        
        Converts the new JointTrajectory format to legacy waypoints/times format.
        """
        start_time = time.time()  # Use time.time() like legacy system
        
        # Clear any final position holding when starting new trajectory
        if robot_name in self._final_positions:
            del self._final_positions[robot_name]
        
        # Convert JointTrajectory to legacy format (waypoints + times arrays)
        waypoints = np.array([list(pos) for pos in trajectory.positions], dtype=np.float64)
        times = np.array(list(trajectory.times_s), dtype=np.float64)
        
        self._active_trajectories[robot_name] = ActiveTrajectory(
            waypoints=waypoints,
            times=times,
            start_time=start_time
        )

        trajectory_record = {
            "waypoints": waypoints.tolist(),
            "times_s": times.tolist(),
            "duration": float(trajectory.duration),
        }
        self._recent_trajectories[robot_name].append(trajectory_record)
        if self._logger:
            self._logger.info(
                f"Started synchronized trajectory for {robot_name}: "
                f"duration={trajectory.duration:.3f}s, waypoints={len(trajectory.positions)}"
            )
    
    def stop_trajectory(self, robot_name: str) -> None:
        """Stop an active trajectory for the given robot."""
        if robot_name in self._active_trajectories:
            del self._active_trajectories[robot_name]
        if robot_name in self._final_positions:
            del self._final_positions[robot_name]
        if self._logger:
            self._logger.info(f"Stopped trajectory for {robot_name}")

    def hold_position(self, robot_name: str, position: np.ndarray) -> None:
        """Record a final position without an active trajectory (legacy-style joint hold)."""
        if robot_name in self._active_trajectories:
            del self._active_trajectories[robot_name]
        self._final_positions[robot_name] = np.array(position, dtype=np.float64)
    
    def get_current_positions(self, robot_name: str, current_time: float) -> Optional[np.ndarray]:
        """Get current joint positions for a robot (either from active trajectory or final position)."""
        # Check for active trajectory first (like legacy system)
        if robot_name in self._active_trajectories:
            active_traj = self._active_trajectories[robot_name]
            sample = active_traj.sample(current_time)
            return sample["q"]
        
        # Check for final position holding (like legacy joint_targets after trajectory completion)
        if robot_name in self._final_positions:
            return self._final_positions[robot_name]
        
        return None
    
    def is_trajectory_done(self, robot_name: str, current_time: float) -> bool:
        """Check if a trajectory has completed."""
        if robot_name not in self._active_trajectories:
            return True
        
        active_traj = self._active_trajectories[robot_name]
        if not active_traj:
            return True
        
        sample = active_traj.sample(current_time)
        is_done = sample["done"]
        
        if is_done:
            # Get final position before cleanup for position holding
            final_position = sample["q"]
            
            # Store final position for position holding (like legacy joint_targets)
            self._final_positions[robot_name] = final_position
            
            # Clean up completed trajectory
            del self._active_trajectories[robot_name]
            # Note: Logging moved to session level to avoid repeated messages
        
        return is_done
    
    def has_active_trajectory(self, robot_name: str) -> bool:
        """Check if the robot has an active trajectory."""
        return robot_name in self._active_trajectories

    def get_active_robots(self) -> list[str]:
        """Get list of robots with active trajectories."""
        return list(self._active_trajectories.keys())

    def pop_recent_trajectory(self, robot_name: str) -> Optional[Dict[str, Any]]:
        """Return the most recent planned trajectory for ``robot_name``.

        The record includes ``waypoints`` (list of joint vectors), ``times_s`` and
        ``duration``. Returns ``None`` when no trajectory has been recorded.
        """
        queue = self._recent_trajectories.get(robot_name)
        if not queue:
            return None
        return queue.popleft()

    def clear_recent_trajectories(self, robot_name: str) -> None:
        """Forget any buffered trajectory data for the given robot."""
        if robot_name in self._recent_trajectories:
            self._recent_trajectories[robot_name].clear()


__all__ = ["ActiveTrajectory", "SynchronizedTrajectoryManager"]
