from __future__ import annotations

import threading
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import SimforgeConfig, RobotConfig
from .logging_utils import setup_logging
from .urdf_utils import parse_joint_limits, select_end_effector_link


class Simulator:
    def __init__(self, config: SimforgeConfig, debug: bool = False) -> None:
        self.config = config
        self.logger = setup_logging(debug)
        self._scene = None
        self._robots: Dict[str, object] = {}
        self._robot_dofs_idx: Dict[str, List[int]] = {}
        self._initial_q_deg: Dict[str, List[float]] = {}
        self._pd_gains: Dict[str, dict] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._built = False
        # Hold last commanded joint targets (degrees) to enforce kinematic pose each step
        self._targets_deg: Dict[str, List[float]] = {}
        # Active Cartesian trajectories (per robot)
        self._active_traj: Dict[str, Dict[str, object]] = {}
        # Map robot name -> ee link name (URDF)
        self._ee_link_name: Dict[str, str] = {}
        # Event sink for GUI notifications
        self._evt_cb = None
        # Optional UI-thread executor for Genesis ops that must run on main thread
        self._ui_exec = None
        # Planning guard to avoid stepping during heavy kernels on UI thread
        self._planning = False
        # Command queue to run IK/planning on the sim thread
        self._cmd_q: "queue.Queue[tuple]" = queue.Queue()
        self._last_enq_time: float = 0.0

    # --- Genesis helpers ---
    def _import_genesis(self):
        try:
            import genesis as gs  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Genesis is not installed. Please `pip install genesis-world`"
            ) from e
        return gs

    def _init_genesis(self):
        gs = self._import_genesis()
        backend_cfg = (self.config.scene.backend or "gpu").lower()
        if backend_cfg == "gpu":
            gs.init(backend=gs.gpu)
        elif backend_cfg == "cuda":
            gs.init(backend=gs.cuda)
        else:
            gs.init(backend=gs.cpu)
        # Ensure Genesis logger does not double-print alongside our root handler
        import logging as _logging
        g = _logging.getLogger("genesis")
        for h in list(g.handlers):
            g.removeHandler(h)
        g.propagate = True
        g.setLevel(_logging.DEBUG if self.logger.isEnabledFor(_logging.DEBUG) else _logging.INFO)

        self.logger.info("Genesis initialized (backend=%s)", backend_cfg)
        return gs

    def build_scene(self):
        if self._built:
            return
        gs = self._init_genesis()

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.config.scene.dt, gravity=tuple(self.config.scene.gravity)
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, 0.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                max_FPS=self.config.scene.max_fps,
            ),
            show_viewer=self.config.scene.show_viewer
        )
        scene.profiling_options.show_FPS = False

        # Objects
        for obj in self.config.objects:
            if obj.type == "plane":
                # Genesis Plane does not accept a 'size' attribute in 0.3.3; it's an infinite plane.
                scene.add_entity(
                    gs.morphs.Plane(
                        pos=tuple(obj.position),
                    )
                )
            
        # Robots
        for r in self.config.robots:
            self._add_robot(scene, r)

        scene.build()
        # Apply initial joint positions after build (Genesis requires built entities)
        for r in self.config.robots:
            init = self._initial_q_deg.get(r.name)
            if not init:
                continue
            entity = self._robots[r.name]
            dofs_idx = self._robot_dofs_idx[r.name]
            q_rad = np.array([np.deg2rad(v) for v in init], dtype=np.float32)
            try:
                entity.set_dofs_position(q_rad, dofs_idx_local=dofs_idx)
            except Exception:
                try:
                    entity.set_dofs_position(q_rad, dofs_idx)
                except Exception as e:
                    self.logger.warning("Failed to set initial joint positions: %s", e)
        # Kinematic control mode: no PD/force tweaking
        # Resolve EE link names for Cartesian planning (prefer config override)
        for r in self.config.robots:
            ee_name = r.end_effector_link if hasattr(r, 'end_effector_link') else None
            if not ee_name:
                try:
                    ee_name = select_end_effector_link(r.urdf)
                except Exception as e:
                    self.logger.debug("EE select failed for %s: %s", r.name, e)
                    ee_name = None
            if ee_name:
                self._ee_link_name[r.name] = ee_name
                self.logger.info("%s: Using EE link '%s' for Cartesian planning", r.name, ee_name)
            else:
                self.logger.warning("%s: No EE link resolved; Cartesian IK will fail", r.name)
        self._scene = scene
        self._built = True
        self.logger.info("Scene built (viewer=%s)", self.config.scene.show_viewer)

    def _add_robot(self, scene, robot_cfg: RobotConfig):
        gs = self._import_genesis()
        self.logger.info(
            "Loading robot %s from %s", robot_cfg.name, robot_cfg.urdf
        )
        entity = scene.add_entity(
            gs.morphs.URDF(
                file=robot_cfg.urdf,
                pos=tuple(robot_cfg.base_position),
                euler=tuple(robot_cfg.base_orientation),
                fixed=robot_cfg.fixed_base,
            )
        )

        # Cache dof indices
        try:
            dofs_idx = [j.dof_idx_local for j in entity.get_joints()]
        except Exception:
            # Fallback: assume 6DOF
            dofs_idx = list(range(6))
        self._robots[robot_cfg.name] = entity
        self._robot_dofs_idx[robot_cfg.name] = dofs_idx

        # Stash initial joint positions to apply after scene.build()
        if robot_cfg.initial_joint_positions:
            init = list(robot_cfg.initial_joint_positions)
            self._initial_q_deg[robot_cfg.name] = init
            self._targets_deg[robot_cfg.name] = init
        else:
            # Default to zeros matching detected DOFs length
            self._targets_deg[robot_cfg.name] = [0.0] * len(dofs_idx)

        # Kinematic control mode: ignore PD gain configuration

    # --- Control API ---
    def set_joint_targets_rad(self, robot: str, q_rad: List[float]):
        entity = self._robots[robot]
        dofs_idx = self._robot_dofs_idx[robot]
        q = np.array(q_rad, dtype=np.float32)
        # Kinematic control: directly set joint positions (no PD dynamics)
        try:
            entity.set_dofs_position(q, dofs_idx_local=dofs_idx)
        except Exception:
            entity.set_dofs_position(q, dofs_idx)

    def set_joint_targets_deg(self, robot: str, q_deg: List[float]):
        self._targets_deg[robot] = list(q_deg)
        q_rad = [np.deg2rad(v) for v in q_deg]
        self.set_joint_targets_rad(robot, q_rad)

    def get_robot_joint_limits(self, robot_cfg: RobotConfig):
        joints = parse_joint_limits(robot_cfg.urdf)
        return joints

    def get_joint_positions_rad(self, robot: str) -> List[float]:
        entity = self._robots[robot]
        try:
            # Estimate readback via joint objects
            q = [j.qpos for j in entity.get_joints()]
            return [float(v) for v in q]
        except Exception:
            return []

    # --- Lifecycle ---
    def start(self):
        if not self._built:
            self.build_scene()
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, name="simforge-sim", daemon=True)
        self._thread.start()
        self.logger.info("Simulation thread started")

    def enqueue_cartesian_move(self, robot: str, pose_xyzrpy: tuple, frame: str = "base"):
        """Public API to request a Cartesian move; runs on the sim thread."""
        try:
            self._last_enq_time = time.perf_counter()
            self._cmd_q.put(("cartesian_move", (robot, pose_xyzrpy, frame)), block=False)
            self.logger.debug("Enqueued cartesian_move for %s: %s in %s", robot, pose_xyzrpy, frame)
        except Exception as e:
            self.logger.error("Failed to enqueue cartesian_move: %s", e)

    def _loop(self):
        assert self._scene is not None
        dt = self.config.scene.dt
        last = time.perf_counter()
        while not self._stop_evt.is_set():
            # Handle queued commands from controller (ensure Genesis calls happen on this thread)
            try:
                last_cmd = None
                cancels: set[str] = set()
                while True:
                    kind, payload = self._cmd_q.get_nowait()
                    if kind == "cartesian_move":
                        # skip moves for robots that were canceled in this drain
                        r, pose, frame = payload
                        if r in cancels:
                            continue
                        last_cmd = payload  # keep only the latest move
                    elif kind == "cartesian_cancel":
                        (r,) = payload
                        cancels.add(r)
                        # stop active trajectory and drop pending moves for this robot
                        self._active_traj.pop(r, None)
                # no break; rely on exception
            except queue.Empty:
                pass
            if last_cmd is not None:
                robot, pose, frame = last_cmd
                try:
                    self.plan_and_execute_cartesian(robot, pose, frame)
                except Exception as e:
                    self.logger.error("plan_and_execute_cartesian error: %s", e)
            # Apply kinematic holds or follow active trajectories
            if self._planning:
                # Avoid stepping while a main-thread kernel is running
                time.sleep(dt)
                continue
            try:
                for name, entity in self._robots.items():
                    dofs_idx = self._robot_dofs_idx[name]
                    traj = self._active_traj.get(name)
                    if traj:
                        # Follow current waypoint kinematically (no PD; reliable without actuators)
                        target: np.ndarray = traj["current"]  # radians
                        try:
                            entity.set_dofs_position(target, dofs_idx_local=dofs_idx)
                        except Exception:
                            entity.set_dofs_position(target, dofs_idx)
                    else:
                        tdeg = self._targets_deg.get(name)
                        if not tdeg:
                            continue
                        q = np.array([np.deg2rad(v) for v in tdeg], dtype=np.float32)
                        try:
                            entity.set_dofs_position(q, dofs_idx_local=dofs_idx)
                        except Exception:
                            entity.set_dofs_position(q, dofs_idx)
            except Exception as e:
                self.logger.debug("Command application failed: %s", e)
            # Step physics
            try:
                self._scene.step()
            except Exception as e:
                self.logger.error("Scene step failed: %s", e)
                time.sleep(dt)
                continue

            # Progress active trajectories and check for collisions (force spikes)
            try:
                for name, entity in list(self._active_traj.items()):
                    waypoints: List[np.ndarray] = entity.get("waypoints")  # type: ignore
                    idx: int = entity.get("idx")  # type: ignore
                    robot_entity = self._robots[name]
                    dofs_idx = self._robot_dofs_idx[name]
                    # Advance if close enough
                    try:
                        q_actual = np.array([j.qpos for j in robot_entity.get_joints()], dtype=np.float32)
                    except Exception:
                        q_actual = waypoints[idx]
                    err = np.linalg.norm((waypoints[idx] - q_actual))
                    if err < 0.01:
                        idx += 1
                        if idx >= len(waypoints):
                            # Done
                            self._active_traj.pop(name, None)
                            # Set hold target to final pose (deg)
                            q_deg = [float(np.rad2deg(v)) for v in waypoints[-1]]
                            self._targets_deg[name] = q_deg
                            continue
                        entity["idx"] = idx
                        entity["current"] = waypoints[idx]

                    # Simple collision monitor via joint forces (if available)
                    try:
                        jf = robot_entity.get_dofs_force()
                        if np.max(np.abs(jf)) > 200.0:  # threshold
                            self.logger.warning("Collision/force spike detected; stopping trajectory for %s", name)
                            self._active_traj.pop(name, None)
                            self._targets_deg[name] = [float(np.rad2deg(v)) for v in q_actual]
                    except Exception:
                        pass
            except Exception as e:
                self.logger.debug("Trajectory update failed: %s", e)

            # Throttle roughly to real-time viewer FPS
            now = time.perf_counter()
            elapsed = now - last
            sleep_s = max(0.0, dt - elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)
            last = now

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop_evt.set()
            self._thread.join(timeout=2.0)
            self.logger.info("Simulation thread stopped")

    # Debug helper to print EE choices
    def log_end_effector_choices(self):
        for r in self.config.robots:
            name = self._ee_link_name.get(r.name)
            if not name:
                self.logger.info("%s: EE link not resolved yet", r.name)
            else:
                self.logger.info("%s: Using EE link '%s'", r.name, name)

    # Event sink & helpers
    def set_event_sink(self, cb):
        self._evt_cb = cb

    def set_ui_executor(self, call_sync):
        """Install a callable that executes a function on the UI/main thread and returns its result.
        Expected signature: result = call_sync(fn: Callable[[], Any])
        """
        self._ui_exec = call_sync

    def _emit(self, kind: str, robot: str, info: str = ""):
        try:
            if self._evt_cb:
                self._evt_cb(kind, robot, info)
        except Exception:
            pass

    # Quaternion helpers (WXYZ)
    @staticmethod
    def _rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        return np.array([
            cy*cp*cr + sy*sp*sr,
            cy*cp*sr - sy*sp*cr,
            sy*cp*sr + cy*sp*cr,
            sy*cp*cr - cy*sp*sr,
        ], dtype=np.float32)

    @staticmethod
    def _quat_wxyz_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=np.float32)

    @staticmethod
    def _quat_wxyz_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        # rotate v by quaternion q (wxyz)
        w,x,y,z = q
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
        ], dtype=np.float32)
        return R @ v

    def _euler_deg_to_wxyz(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        return self._rpy_to_quat_wxyz(np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw))

    def _base_pose(self, robot_cfg: RobotConfig):
        t = np.array(robot_cfg.base_position, dtype=np.float32)
        q = self._euler_deg_to_wxyz(*robot_cfg.base_orientation)
        return t, q

    def _apply_frame(self, robot_cfg: RobotConfig, target_pos_base, target_quat_base_wxyz, frame: str):
        if frame == 'base':
            t_base, q_base = self._base_pose(robot_cfg)
            pos_local = np.array(target_pos_base, dtype=np.float32)
            pw = t_base + self._quat_wxyz_rotate_vec(q_base, pos_local)
            qw = self._quat_wxyz_multiply(q_base, np.array(target_quat_base_wxyz, dtype=np.float32))
            return tuple(pw.tolist()), tuple(qw.tolist())
        return target_pos_base, target_quat_base_wxyz

    # Pose helpers
    def _get_ee_pose_w(self, entity, ee_name: str):
        link = entity.get_link(ee_name)
        pos = link.get_pos()
        quat = link.get_quat()  # wxyz
        pos = self._to_np(pos)
        quat = self._to_np(quat)
        return pos.astype(np.float32), quat.astype(np.float32)

    @staticmethod
    def _quat_wxyz_conj(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

    @staticmethod
    def _quat_wxyz_normalize(q: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(q)
        return q / (n + 1e-9)

    def _quat_wxyz_to_rotvec(self, q: np.ndarray) -> np.ndarray:
        qn = self._quat_wxyz_normalize(q)
        w, x, y, z = qn
        s = np.linalg.norm([x, y, z])
        if s < 1e-9:
            return np.zeros(3, dtype=np.float32)
        angle = 2.0 * np.arctan2(s, w)
        axis = np.array([x, y, z], dtype=np.float32) / s
        return axis * angle

    # Numeric IK fallback (damped least-squares)
    def _numeric_ik(self, entity, ee_name: str, q0: np.ndarray, pos_w: tuple, quat_w_wxyz: tuple,
                    iters: int = 60, step_scale: float = 1.0, damping: float = 1e-2) -> np.ndarray | None:
        try:
            dofs_idx = [j.dof_idx_local for j in entity.get_joints()]
        except Exception:
            dofs_idx = list(range(len(q0)))
        q = q0.copy().astype(np.float32)
        target_p = np.array(pos_w, dtype=np.float32)
        target_q = np.array(quat_w_wxyz, dtype=np.float32)
        eps = 1e-3  # rad perturbation

        # Cache original pose to restore
        try:
            entity.set_dofs_position(q, dofs_idx_local=dofs_idx)
        except Exception:
            entity.set_dofs_position(q, dofs_idx)

        for _ in range(iters):
            # Current pose
            p_cur, q_cur = self._get_ee_pose_w(entity, ee_name)
            # Error: position + orientation (rotation vector)
            e_pos = target_p - p_cur
            q_err = self._quat_wxyz_multiply(target_q, self._quat_wxyz_conj(q_cur))
            e_rot = self._quat_wxyz_to_rotvec(q_err)
            e = np.concatenate([e_pos, e_rot]).astype(np.float32)
            if np.linalg.norm(e_pos) < 1e-3 and np.linalg.norm(e_rot) < 1e-2:
                break

            # Numerical Jacobian J (6 x n)
            n = len(dofs_idx)
            J = np.zeros((6, n), dtype=np.float32)
            for j in range(n):
                q_pert = q.copy()
                q_pert[j] += eps
                try:
                    entity.set_dofs_position(q_pert, dofs_idx_local=dofs_idx)
                except Exception:
                    entity.set_dofs_position(q_pert, dofs_idx)
                p_p, q_p = self._get_ee_pose_w(entity, ee_name)
                dp = (p_p - p_cur) / eps
                dq = self._quat_wxyz_multiply(q_p, self._quat_wxyz_conj(q_cur))
                drot = self._quat_wxyz_to_rotvec(dq) / eps
                J[:, j] = np.concatenate([dp, drot])

            # Restore to q after perturbations
            try:
                entity.set_dofs_position(q, dofs_idx_local=dofs_idx)
            except Exception:
                entity.set_dofs_position(q, dofs_idx)

            # Damped least squares step
            JT = J.T
            H = JT @ J + (damping * np.eye(J.shape[1], dtype=np.float32))
            dq = step_scale * (np.linalg.solve(H, JT @ e))
            q += dq
            try:
                entity.set_dofs_position(q, dofs_idx_local=dofs_idx)
            except Exception:
                entity.set_dofs_position(q, dofs_idx)

        # Final pose check
        p_cur, q_cur = self._get_ee_pose_w(entity, ee_name)
        e_pos = np.linalg.norm(target_p - p_cur)
        q_err = self._quat_wxyz_multiply(target_q, self._quat_wxyz_conj(q_cur))
        e_rot = np.linalg.norm(self._quat_wxyz_to_rotvec(q_err))

        # Restore original q0; we'll return solution separately
        try:
            entity.set_dofs_position(q0, dofs_idx_local=dofs_idx)
        except Exception:
            entity.set_dofs_position(q0, dofs_idx)

        if e_pos < 5e-3 and e_rot < 5e-2:  # 5 mm and ~3 deg
            return q
        return None

    def get_ee_pose_xyzrpy(self, robot: str):
        ee_name = self._ee_link_name.get(robot)
        if not ee_name:
            return None
        entity = self._robots[robot]
        try:
            link = entity.get_link(ee_name)
            pos = self._to_np(link.get_pos())
            quat = self._to_np(link.get_quat())  # wxyz
            w,x,y,z = quat
            t0 = 2*(w*x + y*z)
            t1 = 1 - 2*(x*x + y*y)
            roll = np.arctan2(t0, t1)
            t2 = 2*(w*y - z*x)
            t2 = np.clip(t2, -1.0, 1.0)
            pitch = np.arcsin(t2)
            t3 = 2*(w*z + x*y)
            t4 = 1 - 2*(y*y + z*z)
            yaw = np.arctan2(t3, t4)
            return (
                float(pos[0]), float(pos[1]), float(pos[2]),
                float(np.rad2deg(roll)), float(np.rad2deg(pitch)), float(np.rad2deg(yaw)),
            )
        except Exception:
            return None

    # Generic helpers to convert tensors/arrays to numpy (CPU)
    @staticmethod
    def _to_np(x):
        try:
            # torch.Tensor on GPU/CPU
            import torch  # type: ignore
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        try:
            # Objects exposing .numpy()
            if hasattr(x, 'numpy') and callable(getattr(x, 'numpy')):
                return x.numpy()
        except Exception:
            pass
        # As a fallback, wrap via numpy.array
        return np.array(x)

    # --- Cartesian planning/execution ---
    def plan_and_execute_cartesian(self, robot: str, pose_xyzrpy: tuple, frame: str = 'base'):
        """
        Compute IK for target pose and execute a smooth joint-space trajectory.
        Falls back to linear joint interpolation if planning backend is unavailable.
        """
        gs = self._import_genesis()
        entity = self._robots[robot]
        dofs_idx = self._robot_dofs_idx[robot]
        # Cancel any existing trajectory for this robot, we will replace it
        self._active_traj.pop(robot, None)
        # Trace request for debugging
        try:
            self.logger.debug("Cartesian request %s: pos=%s rpy=%s frame=%s", robot, pose_xyzrpy[:3], pose_xyzrpy[3:], frame)
        except Exception:
            pass

        # Current joints as starting state
        try:
            q_start = np.array([j.qpos for j in entity.get_joints()], dtype=np.float32)
        except Exception:
            q_start = np.zeros((len(dofs_idx),), dtype=np.float32)

        # Build base-frame target and convert to world if needed
        x, y, z, roll, pitch, yaw = pose_xyzrpy
        q_local_wxyz = self._euler_deg_to_wxyz(roll, pitch, yaw) if hasattr(self, '_euler_deg_to_wxyz') else None
        if q_local_wxyz is None:
            # local fallback if helpers not yet defined in this class
            r, p, yw = np.deg2rad([roll, pitch, yaw])
            cr, sr = np.cos(r/2), np.sin(r/2)
            cp, sp = np.cos(p/2), np.sin(p/2)
            cy, sy = np.cos(yw/2), np.sin(yw/2)
            q_local_wxyz = np.array([cy*cp*cr + sy*sp*sr,
                                     cy*cp*sr - sy*sp*cr,
                                     sy*cp*sr + cy*sp*cr,
                                     sy*cp*cr - cy*sp*sr], dtype=np.float32)
        robot_cfg = next(r for r in self.config.robots if r.name == robot)
        if hasattr(self, '_apply_frame'):
            pos_w, quat_w_wxyz = self._apply_frame(robot_cfg, (x, y, z), q_local_wxyz, frame)
        else:
            pos_w = (x, y, z)
            quat_w_wxyz = q_local_wxyz
        target_pos = (float(pos_w[0]), float(pos_w[1]), float(pos_w[2]))
        # Many APIs expect XYZW; also prepare WXYZ
        target_quat_xyzw = (float(quat_w_wxyz[1]), float(quat_w_wxyz[2]), float(quat_w_wxyz[3]), float(quat_w_wxyz[0]))
        target_quat_wxyz = (float(quat_w_wxyz[0]), float(quat_w_wxyz[1]), float(quat_w_wxyz[2]), float(quat_w_wxyz[3]))

        # Determine end-effector link name and object if possible
        ee_name = self._ee_link_name.get(robot)
        end_eff_link = None
        try:
            end_eff_link = entity.get_link(ee_name) if ee_name else None
        except Exception:
            end_eff_link = None

        started_at = time.perf_counter()
        # Attempt Genesis IK
        q_goal = None
        used_solver = None
        # Preferred: official inverse_kinematics with a link handle
        try:
            if end_eff_link is not None and hasattr(entity, 'inverse_kinematics'):
                try:
                    q_goal = entity.inverse_kinematics(
                        link=end_eff_link,
                        pos=np.array(target_pos, dtype=np.float32),
                        quat=np.array(target_quat_xyzw, dtype=np.float32),
                    )
                    used_solver = 'entity.inverse_kinematics(xyzw)'
                except Exception:
                    q_goal = entity.inverse_kinematics(
                        link=end_eff_link,
                        pos=np.array(target_pos, dtype=np.float32),
                        quat=np.array(target_quat_wxyz, dtype=np.float32),
                    )
                    used_solver = 'entity.inverse_kinematics(wxyz)'
        except Exception as e:
            self.logger.debug("inverse_kinematics raised: %s", e)

        # Try multiple legacy IK signatures/orders if still not solved
        try:
            if hasattr(entity, 'solve_ik'):
                kwargs = {'pos': target_pos, 'initial': q_start}
                # Try passing EE spec via name with different common parameter names
                if ee_name is not None:
                    for key in ('link_name', 'ee', 'tip', 'end_effector', 'tool'):
                        kwargs[key] = ee_name
                # try XYZW then WXYZ; try euler as fallback
                try:
                    q_goal = entity.solve_ik(quat=target_quat_xyzw, **kwargs)
                    used_solver = 'entity.solve_ik(xyzw)'
                except Exception:
                    try:
                        q_goal = entity.solve_ik(quat=target_quat_wxyz, **kwargs)
                        used_solver = 'entity.solve_ik(wxyz)'
                    except Exception:
                        q_goal = entity.solve_ik(euler=(np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)), **kwargs)
                        used_solver = 'entity.solve_ik(euler)'
            elif hasattr(gs, 'motion') and hasattr(gs.motion, 'InverseKinematics'):
                ik = gs.motion.InverseKinematics(entity)
                kw = {'target_pos': target_pos, 'q0': q_start}
                if ee_name is not None:
                    for key in ('link_name', 'ee', 'tip', 'end_effector', 'tool'):
                        kw[key] = ee_name
                try:
                    sol = ik.solve(target_pos=target_pos, target_quat=target_quat_xyzw, q0=q_start)
                    q_goal = getattr(sol, 'q', None)
                    used_solver = 'gs.motion.IK(xyzw)'
                except Exception:
                    try:
                        sol = ik.solve(target_pos=target_pos, target_quat=target_quat_wxyz, q0=q_start)
                        q_goal = getattr(sol, 'q', None)
                        used_solver = 'gs.motion.IK(wxyz)'
                    except Exception:
                        sol = ik.solve(target_pos=target_pos, target_euler=(np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)), q0=q_start)
                        q_goal = getattr(sol, 'q', None)
                        used_solver = 'gs.motion.IK(euler)'
        except Exception as e:
            self.logger.warning("Genesis IK failed: %s", e)

        if q_goal is None:
            # Fallback: numeric IK
            if ee_name:
                q_goal = self._numeric_ik(entity, ee_name, q_start, target_pos, target_quat_wxyz)
                if q_goal is not None:
                    used_solver = 'numeric_ik'
            if q_goal is None:
                self.logger.warning("IK failed (robot=%s); rejecting Cartesian command", robot)
                self._emit("cartesian_failed", robot, "ik_failed")
                return False

        q_goal = self._to_np(q_goal).astype(np.float32)
        # Strict Cartesian: require a planned, collision-checked path if available
        waypoints = None
        if getattr(self.config.control, 'strict_cartesian', True):
            if not hasattr(entity, 'plan_path'):
                self.logger.warning("OMPL/plan_path not available; set control.strict_cartesian=false to test without planning")
                self._emit("cartesian_failed", robot, "planner_unavailable")
                return False
            try:
                self._planning = True
                if self._ui_exec is not None:
                    waypoints = self._ui_exec(lambda: entity.plan_path(qpos_goal=q_goal, num_waypoints=200))
                else:
                    waypoints = entity.plan_path(qpos_goal=q_goal, num_waypoints=200)
            except Exception:
                waypoints = None
            finally:
                self._planning = False
            # Convert and validate waypoints without ambiguous truthiness
            ok = True
            if waypoints is None:
                ok = False
                wp_list = []
            else:
                try:
                    import torch  # type: ignore
                    if hasattr(waypoints, 'numel'):
                        ok = waypoints.numel() > 0
                        wp_np = waypoints.detach().cpu().numpy()
                        wp_list = [wp_np[i].astype(np.float32) for i in range(wp_np.shape[0])]
                    else:
                        wp_np = self._to_np(waypoints)
                        if isinstance(wp_np, np.ndarray) and wp_np.ndim == 2:
                            ok = wp_np.size > 0
                            wp_list = [wp_np[i].astype(np.float32) for i in range(wp_np.shape[0])]
                        else:
                            wp_list = [self._to_np(w).astype(np.float32) for w in waypoints]
                            ok = len(wp_list) > 0
                except Exception:
                    try:
                        wp_np = self._to_np(waypoints)
                        if isinstance(wp_np, np.ndarray) and wp_np.ndim == 2:
                            ok = wp_np.size > 0
                            wp_list = [wp_np[i].astype(np.float32) for i in range(wp_np.shape[0])]
                        else:
                            wp_list = [self._to_np(w).astype(np.float32) for w in waypoints]
                            ok = len(wp_list) > 0
                    except Exception:
                        ok = False
                        wp_list = []

            # Drop stale plan if newer command arrived during planning
            if started_at < self._last_enq_time:
                self.logger.debug("Discarding stale plan for %s due to newer command", robot)
                return False

            if not ok:
                self.logger.warning("Planning failed/collision (robot=%s); rejecting Cartesian command", robot)
                self._emit("cartesian_failed", robot, "plan_failed")
                return False
            waypoints = wp_list
        else:
            N = 60
            waypoints = [q_start + (q_goal - q_start) * (i / (N - 1)) for i in range(N)]

        self._active_traj[robot] = {"waypoints": waypoints, "idx": 0, "current": waypoints[0]}
        self.logger.info("Executing Cartesian move for %s with %d waypoints (solver=%s)", robot, len(waypoints), used_solver)
        self._emit("cartesian_executing", robot, "")
        return True
