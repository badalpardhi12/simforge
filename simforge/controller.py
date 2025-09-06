from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import SimforgeConfig, RobotConfig
from .simulator import Simulator


class ControlMode:
    JOINT = "joint"
    CARTESIAN = "cartesian"


@dataclass
class Command:
    kind: str
    payload: tuple


class Controller:
    def __init__(self, config: SimforgeConfig, debug: bool = False) -> None:
        self.config = config
        self.sim = Simulator(config, debug=debug)
        self.mode = ControlMode.JOINT
        self._cmd_q: "queue.Queue[Command]" = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._running = False

        # Cache per-robot target state for joint control
        self._targets_deg: Dict[str, List[float]] = {}
        for r in self.config.robots:
            self._targets_deg[r.name] = list(r.initial_joint_positions or [])
        self._last_cartesian: Dict[str, tuple] = {}
        self._last_evt: Dict[str, str] = {}
        # connect event sink
        self.sim.set_event_sink(self._on_sim_event)

    def start(self):
        self.sim.build_scene()
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

            if cmd.kind == "set_joint_deg":
                robot, idx, val_deg = cmd.payload
                arr = self._targets_deg.get(robot)
                if arr is None or idx >= len(arr):
                    continue
                arr[idx] = float(val_deg)
                self.sim.set_joint_targets_deg(robot, arr)
            elif cmd.kind == "set_joint_targets_deg":
                robot, arr = cmd.payload
                self._targets_deg[robot] = list(arr)
                self.sim.set_joint_targets_deg(robot, list(arr))
            elif cmd.kind == "switch_mode":
                mode, = cmd.payload
                self.mode = mode
                # Reset robot(s) to home on mode switch
                for r in self.config.robots:
                    init = r.initial_joint_positions or []
                    if init:
                        self._targets_deg[r.name] = list(init)
                        self.sim.set_joint_targets_deg(r.name, list(init))
                # Log EE link choice for debugging when entering Cartesian mode
                if self.mode == ControlMode.CARTESIAN:
                    self.sim.log_end_effector_choices()
            elif cmd.kind == "cartesian_move":
                robot, pose, frame = cmd.payload
                try:
                    # Enqueue to sim thread to avoid nested kernel calls
                    self.sim.enqueue_cartesian_move(robot, pose, frame)
                except Exception:
                    # Avoid crashing GUI thread; errors will be logged by simulator
                    continue

    # --- Public API for GUI ---
    def set_joint_deg(self, robot: str, idx: int, val_deg: float):
        self._cmd_q.put(Command("set_joint_deg", (robot, idx, val_deg)))

    def set_joint_targets_deg(self, robot: str, vals_deg: List[float]):
        self._cmd_q.put(Command("set_joint_targets_deg", (robot, vals_deg)))

    def switch_mode(self, mode: str):
        assert mode in (ControlMode.JOINT, ControlMode.CARTESIAN)
        self._cmd_q.put(Command("switch_mode", (mode,)))

    def set_cartesian_target(self, robot: str, pose_xyzrpy: tuple, frame: str = 'world'):
        # Basic de-bounce: avoid flooding queue with nearly-identical commands
        last = self._last_cartesian.get(robot)
        if last:
            dx = sum(abs(a-b) for a,b in zip(last[:3], pose_xyzrpy[:3]))
            da = sum(abs(a-b) for a,b in zip(last[3:], pose_xyzrpy[3:]))
            if dx < 1e-3 and da < 1.0:  # <1mm and <1deg aggregate
                return
        self._last_cartesian[robot] = pose_xyzrpy
        self._cmd_q.put(Command("cartesian_move", (robot, pose_xyzrpy, frame)))
        # Explicit debug print so we can see GUI events arriving
        print(f"[CTRL] Enqueued cartesian_move for {robot}: {pose_xyzrpy} in {frame}")

    def cancel_cartesian(self, robot: str):
        self._cmd_q.put(Command("cartesian_cancel", (robot,)))
        print(f"[CTRL] Cancel cartesian for {robot}")

    # --- Events and queries for GUI ---
    def _on_sim_event(self, kind: str, robot: str, info: str):
        if kind.startswith("cartesian_"):
            self._last_evt[robot] = f"{kind}:{info}"

    def get_last_cartesian_status(self, robot: str):
        return self._last_evt.get(robot)

    def clear_last_cartesian_status(self, robot: str):
        if robot in self._last_evt:
            self._last_evt[robot] = ""

    def get_ee_pose_xyzrpy(self, robot: str):
        return self.sim.get_ee_pose_xyzrpy(robot)
