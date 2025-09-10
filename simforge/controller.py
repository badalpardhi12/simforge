from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import commands
from .config import SimforgeConfig, RobotConfig
from .simulator import Simulator


class ControlMode:
    JOINT = "joint"
    CARTESIAN = "cartesian"


class Controller:
    """
    Manages high-level robot control, bridging between the GUI/user input and
    the simulator back-end. Handles queueing of commands (joint moves, Cartesian plans)
    and coordinates thread-safe interactions.

    This class provides APIs for:
    - Joint position control (UI sliders)
    - Cartesian motion planning and execution
    - Mode switching (joint vs Cartesian)
    - Real-time status updates and EE pose feedback

    Args:
        config: Configuration object
        debug: Enable debug logging
    """
    def __init__(self, config: SimforgeConfig, debug: bool = False) -> None:
        self.config = config
        self.sim = Simulator(config, debug=debug)
        # Per-robot control mode (default to JOINT)
        self._mode_by_robot: Dict[str, str] = {r.name: ControlMode.JOINT for r in self.config.robots}
        self._cmd_q: "queue.Queue[commands.Command]" = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._running = False

        # Cache per-robot target state for joint control
        self._targets_deg: Dict[str, List[float]] = {}
        for r in self.config.robots:
            self._targets_deg[r.name] = list(r.initial_joint_positions or [])
        # Per-robot last Cartesian target (to preserve UI state)
        self._last_cartesian: Dict[str, tuple] = {}
        self._last_evt: Dict[str, str] = {}
        # connect event sink
        self.sim.set_event_sink(self._on_sim_event)

    def start(self):
        # Build and use Genesis only on the sim thread
        self.sim.start()
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        self.sim.stop()

    def _worker(self):
        # Relay commands to the sim thread by setting target arrays
        while self._running:
            try:
                cmd = self._cmd_q.get(timeout=0.05)
            except queue.Empty:
                continue

            if isinstance(cmd, commands.SetJointDeg):
                arr = self._targets_deg.get(cmd.robot)
                if arr is None or cmd.idx >= len(arr):
                    continue
                arr[cmd.idx] = float(cmd.val_deg)
                self.sim.set_joint_targets_deg(cmd.robot, arr)
            elif isinstance(cmd, commands.SetJointTargetsDeg):
                self._targets_deg[cmd.robot] = list(cmd.vals_deg)
                self.sim.set_joint_targets_deg(cmd.robot, list(cmd.vals_deg))
            elif isinstance(cmd, commands.SwitchMode):
                if cmd.robot not in self._mode_by_robot:
                    continue
                self._mode_by_robot[cmd.robot] = cmd.mode
                # Optionally reset only this robot to its home position when switching mode
                init = next(
                    (
                        r.initial_joint_positions
                        for r in self.config.robots
                        if r.name == cmd.robot
                    ),
                    None,
                )
                if init:
                    self._targets_deg[cmd.robot] = list(init)
                    self.sim.set_joint_targets_deg(cmd.robot, list(init))
                # Log EE link choice for debugging when entering Cartesian mode for this robot
                if cmd.mode == ControlMode.CARTESIAN:
                    self.sim.log_end_effector_choices()
            elif isinstance(cmd, commands.CartesianMove):
                try:
                    # Enqueue to sim thread to avoid nested kernel calls
                    self.sim.enqueue_cartesian_move(
                        cmd.robot, cmd.pose_xyzrpy, cmd.frame
                    )
                except Exception:
                    # Avoid crashing GUI thread; errors will be logged by simulator
                    continue

    # --- Public API for GUI ---
    def set_joint_deg(self, robot: str, idx: int, val_deg: float):
        self._cmd_q.put(commands.SetJointDeg(robot, idx, val_deg))

    def set_joint_targets_deg(self, robot: str, vals_deg: List[float]):
        self._cmd_q.put(commands.SetJointTargetsDeg(robot, vals_deg))

    def switch_mode(self, robot: str, mode: str):
        assert mode in (ControlMode.JOINT, ControlMode.CARTESIAN)
        self._cmd_q.put(commands.SwitchMode(robot, mode))

    def set_cartesian_target(
        self, robot: str, pose_xyzrpy: tuple, frame: str = "world"
    ):
        # Basic de-bounce: avoid flooding queue with nearly-identical commands
        last = self._last_cartesian.get(robot)
        if last:
            dx = sum(abs(a - b) for a, b in zip(last[:3], pose_xyzrpy[:3]))
            da = sum(abs(a - b) for a, b in zip(last[3:], pose_xyzrpy[3:]))
            if dx < 1e-3 and da < 1.0:  # <1mm and <1deg aggregate
                return
        self._last_cartesian[robot] = pose_xyzrpy
        self._cmd_q.put(commands.CartesianMove(robot, pose_xyzrpy, frame))
        # Explicit debug print so we can see GUI events arriving
        print(
            f"[CTRL] Enqueued cartesian_move for {robot}: {pose_xyzrpy} in {frame}"
        )

    def cancel_cartesian(self, robot: str):
        self._cmd_q.put(commands.CartesianCancel(robot))
        print(f"[CTRL] Cancel cartesian for {robot}")

    # --- Events and queries for GUI ---
    def _on_sim_event(self, kind: str, robot: str, info: str):
        if kind.startswith("cartesian_"):
            self._last_evt[robot] = f"{kind}:{info}"
        elif kind == "shutdown_request":
            # Forward shutdown request to GUI thread
            if hasattr(self, '_shutdown_callback'):
                self._shutdown_callback()
        elif kind == "scene_built":
            # Forward scene built event to GUI thread to show window
            if hasattr(self, '_scene_callback'):
                self._scene_callback()

    def get_last_cartesian_status(self, robot: str):
        return self._last_evt.get(robot)

    def clear_last_cartesian_status(self, robot: str):
        if robot in self._last_evt:
            self._last_evt[robot] = ""

    def get_ee_pose_xyzrpy(self, robot: str):
        return self.sim.get_ee_pose_xyzrpy(robot)

    # --- Additional helpers for GUI state preservation ---
    def get_mode(self, robot: str) -> str:
        return self._mode_by_robot.get(robot, ControlMode.JOINT)

    def get_joint_targets_deg(self, robot: str) -> List[float]:
        return list(self._targets_deg.get(robot, []))

    def get_last_cartesian_target(self, robot: str) -> Optional[tuple]:
        return self._last_cartesian.get(robot)
