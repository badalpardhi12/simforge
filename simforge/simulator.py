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
from .collision import CollisionChecker, CollisionWorld


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
        self._colliders: Dict[str, CollisionChecker] = {}
        self._world: Optional[CollisionWorld] = None  # world-aware collision checker
        # Cache for EE poses (updated on sim thread, read from UI thread)
        self._ee_pose_cache: Dict[str, tuple] = {}
        # Built event for startup synchronization
        self._built_evt = threading.Event()
        # Shutdown synchronization event for GUI thread
        self._shutdown_evt = threading.Event()

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
        # Suppress TensorFlow warnings (keep minimal to avoid conflicts)
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

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
        g.setLevel(_logging.WARNING)  # Only show warnings and errors

        self.logger.info("Genesis initialized (backend=%s)", backend_cfg)
        return gs

    def build_scene(self):
        if self._built:
            return
        gs = self._init_genesis()

        # Patch Genesis cleanup immediately after initialization
        try:
            from .main import _patch_genesis_cleanup
            _patch_genesis_cleanup()
            self.logger.debug("Genesis cleanup patching applied")
        except Exception as e:
            self.logger.warning(f"Could not patch Genesis cleanup in simulator: {e}")

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

        # Store viewer reference for shutdown detection
        self._viewer_open = self.config.scene.show_viewer
        if self.config.scene.show_viewer and hasattr(scene, 'viewer_manager'):
            self._viewer_manager = scene.viewer_manager
            self.logger.debug("Genesis viewer manager found")
        else:
            self._viewer_manager = None

        # Objects
        for obj in self.config.objects:
            if obj.type == "plane":
                # Genesis Plane does not accept a 'size' attribute in 0.3.3; it's an infinite plane.
                scene.add_entity(
                    gs.morphs.Plane(
                        pos=tuple(obj.position),
                    )
                )
            elif obj.type == "box" and obj.size is not None:
                scene.add_entity(gs.morphs.Box(
                    pos=tuple(obj.position),
                    size=tuple(obj.size),
                    euler=tuple(getattr(obj, "orientation_rpy", (0.0,0.0,0.0)))
                ))
            
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
            ee_name = getattr(r, 'end_effector_link', None)
            if not ee_name:
                # Enforce manual EE selection via YAML as requested
                self.logger.error("%s: end_effector_link is not set in the YAML; Cartesian IK will be disabled", r.name)
            else:
                self._ee_link_name[r.name] = ee_name
                self.logger.info("%s: Using EE link '%s' for Cartesian planning", r.name, ee_name)

        # Build collision checkers (per robot) using URDF collision meshes
        for r in self.config.robots:
            try:
                ent = self._robots[r.name]
                ctrl = self._ctrl_for(r.name)
                self._colliders[r.name] = CollisionChecker(
                    robot_name=r.name,
                    urdf_path=r.urdf,
                    entity=ent,
                    logger=self.logger,
                    plane_z=getattr(ctrl, "ground_plane_z", 0.0),
                    allowed_pairs=getattr(ctrl, "allowed_collision_links", []),
                    mesh_shrink=getattr(ctrl, "collision_mesh_shrink", 1.0),
                )
            except Exception as e:
                self.logger.warning("Failed to build CollisionChecker for %s: %s", r.name, e)

        # Optionally auto-allow any self-collision pairs present at the home pose (URDF artifacts)
        for r in self.config.robots:
            ctrl = self._ctrl_for(r.name)
            if getattr(ctrl, "collision_check", True) and getattr(ctrl, "auto_allow_home_collisions", True):
                coll = self._colliders.get(r.name)
                if not coll:
                    continue
                try:
                    hits = coll.list_self_collisions_now()
                except Exception as e:
                    hits = []
                    self.logger.debug("list_self_collisions_now failed for %s: %s", r.name, e)
                if hits:
                    coll.add_allowed_pairs(hits)
                    pairs = ", ".join([f"{a}-{b}" for (a, b) in hits])
                    self.logger.warning(
                        "%s: Auto-allowed %d self-collision pair(s) at home pose: %s",
                        r.name, len(hits), pairs
                    )

        # --- Build world-aware collision manager (robots + objects)
        try:
            gctrl = self._ctrl_for(self.config.robots[0].name) if self.config.robots else self.config.control
        except Exception:
            gctrl = getattr(self.config, "control", None)

        self._world = CollisionWorld(logger=self.logger, min_clearance=getattr(gctrl, "min_clearance_m", 0.0))

        # Register robots
        for r in self.config.robots:
            chk = self._colliders.get(r.name)
            if chk:
                self._world.add_robot(r.name, chk)

        # Add static objects
        for obj in self.config.objects:
            if not getattr(obj, "collision_enabled", True):
                continue
            if obj.type == "box" and obj.size is not None:
                name = obj.name or f"box_{len(self._world._static_objs)}"
                self._world.add_box(
                    name=name,
                    size_xyz=tuple(obj.size),
                    pos_w=tuple(obj.position),
                    rpy_deg_w=tuple(getattr(obj, "orientation_rpy", (0.0, 0.0, 0.0))),
                )
            # 'plane' is already handled by per-robot ground_plane_z
            # (future: add meshes here via trimesh->BVH if needed)

        # Allowed pairs that involve robots and/or objects, e.g. ("ur5e_1/wrist_3_link", "obj:table1")
        try:
            self._world.allow(getattr(gctrl, "world_allowed_pairs", []))
        except Exception:
            pass

        # Apply initial joint targets (degrees from config) and step a few frames
        try:
            for r in self.config.robots:
                q_deg = self._targets_deg.get(r.name)
                if not q_deg:
                    q_deg = r.initial_joint_positions if r.initial_joint_positions else []
                    self._targets_deg[r.name] = q_deg
                if q_deg:
                    entity = self._robots[r.name]
                    dofs_idx = self._robot_dofs_idx[r.name]
                    q_rad = np.array([np.deg2rad(v) for v in q_deg], dtype=np.float32)
                    try:
                        entity.set_dofs_position(q_rad, dofs_idx_local=dofs_idx)
                    except Exception:
                        entity.set_dofs_position(q_rad, dofs_idx)
            # let Genesis propagate state to links before user commands
            for _ in range(3):
                scene.step()
        except Exception as e:
            self.logger.debug("Failed to apply & settle home pose: %s", e)

        self._scene = scene
        self._built = True
        self.logger.info("Scene built (viewer=%s)", self.config.scene.show_viewer)
        # publish "scene ready" and let GUI unblock
        self._built_evt.set()
        if self._evt_cb:
            self._evt_cb("scene_built", "", "")

    def _add_robot(self, scene, robot_cfg: RobotConfig):
        gs = self._import_genesis()
        self.logger.info(
            "Loading robot %s from %s", robot_cfg.name, robot_cfg.urdf
        )
        ctrl = self._ctrl_for(robot_cfg.name)
        # Optionally render collision meshes for debugging if supported by backend
        urdf_kwargs = dict(
            file=robot_cfg.urdf,
            pos=tuple(robot_cfg.base_position),
            euler=tuple(robot_cfg.base_orientation),
            fixed=robot_cfg.fixed_base,
        )
        if getattr(ctrl, "visualize_collision_meshes", False):
            urdf_kwargs["vis_mode"] = "collision"
        try:
            entity = scene.add_entity(gs.morphs.URDF(**urdf_kwargs))
        except TypeError:
            # Older Genesis may not support 'vis_mode'; retry without it
            urdf_kwargs.pop("vis_mode", None)
            entity = scene.add_entity(gs.morphs.URDF(**urdf_kwargs))

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
    def set_joint_targets_deg(self, robot: str, q_deg: List[float]):
        # Just cache targets; the sim thread applies them each loop.
        self._targets_deg[robot] = list(q_deg)

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
        # Build scene on this (sim) thread to keep all Genesis calls on one thread
        if not self._built:
            self.build_scene()
        assert self._scene is not None
        dt = self.config.scene.dt
        last = time.perf_counter()
        viewer_check_counter = 0
        while not self._stop_evt.is_set() and not self._shutdown_evt.is_set():
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
                        # Direct set like old codebase
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
            # Step physics - direct call like old codebase
            try:
                self._scene.step()
            except Exception as e:
                error_msg = str(e).lower()
                if "viewer" in error_msg and ("closed" in error_msg or "shut down" in error_msg):
                    self.logger.info("Genesis viewer closed - initiating graceful shutdown")
                    try:
                        self.trigger_shutdown()
                    except Exception as e:
                        self.logger.warning(f"Error triggering shutdown: {e}")
                        # Still try to exit gracefully even if event sending fails
                        break
                    break  # Exit the simulation loop cleanly
                else:
                    self.logger.error("Scene step failed: %s", e)
                    time.sleep(dt)
                    continue

            # Check for viewer window closure (every ~30 iterations or ~1.5 seconds at 60fps)
            viewer_check_counter += 1
            if viewer_check_counter >= 30:
                viewer_check_counter = 0
                if self._check_viewer_closed():
                    self.logger.info("Genesis viewer closed, initiating graceful shutdown")
                    try:
                        self.trigger_shutdown()
                    except Exception as e:
                        self.logger.warning(f"Error triggering shutdown: {e}")
                        break
                    break

            # Update EE pose cache on sim thread after scene step
            try:
                for name, entity in self._robots.items():
                    ee = self._ee_link_name.get(name)
                    if not ee:
                        continue
                    link = entity.get_link(ee)
                    pos = self._to_np(link.get_pos()).astype(np.float32)
                    quat = self._to_np(link.get_quat()).astype(np.float32)  # wxyz
                    w, x, y, z = quat
                    t0 = 2*(w*x + y*z); t1 = 1 - 2*(x*x + y*y)
                    roll = np.arctan2(t0, t1)
                    t2 = np.clip(2*(w*y - z*x), -1.0, 1.0)
                    pitch = np.arcsin(t2)
                    t3 = 2*(w*z + x*y); t4 = 1 - 2*(y*y + z*z)
                    yaw = np.arctan2(t3, t4)
                    self._ee_pose_cache[name] = (
                        float(pos[0]), float(pos[1]), float(pos[2]),
                        float(np.rad2deg(roll)), float(np.rad2deg(pitch)), float(np.rad2deg(yaw)),
                    )
            except Exception:
                pass

            # Progress active trajectories - execute one waypoint per simulation step
            try:
                for name, traj_data in list(self._active_traj.items()):
                    waypoints = traj_data.get("waypoints")  # numpy array [N, dof]
                    i: int = traj_data.get("i", 0)  # current waypoint index
                    robot_entity = self._robots[name]
                    
                    # Advance to next waypoint every simulation step
                    if i + 1 < waypoints.shape[0]:
                        traj_data["i"] = i + 1
                        traj_data["current"] = waypoints[i + 1]
                    else:
                        # Trajectory complete
                        self._active_traj.pop(name, None)
                        # Set hold target to final pose (deg)
                        q_deg = [float(np.rad2deg(v)) for v in waypoints[-1]]
                        self._targets_deg[name] = q_deg
                        self.logger.info("Trajectory complete for %s", name)

                    # Simple collision monitor via joint forces (if available)
                    try:
                        jf = robot_entity.get_dofs_force()
                        if np.max(np.abs(jf)) > 200.0:  # threshold
                            self.logger.warning("Collision/force spike detected; stopping trajectory for %s", name)
                            self._active_traj.pop(name, None)
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

    def _check_viewer_closed(self):
        """Check if the Genesis viewer window has been closed."""
        if not self._viewer_open or not self._viewer_manager:
            return False

        try:
            # Try to check if the viewer window is still open
            # This is a heuristics-based approach since Genesis doesn't provide direct viewer close detection
            # We can check if scene stepping still works or if certain viewer properties exist
            # For now, we'll use a simple periodic check
            if hasattr(self._viewer_manager, '_closed') and self._viewer_manager._closed:
                self.logger.info("Genesis viewer window detected as closed")
                return True
        except Exception:
            # If we can't access the viewer, assume it's closed
            if self._viewer_open:
                self.logger.info("Genesis viewer may have been closed (detection failed)")
                return True

        return False

    def trigger_shutdown(self):
        """Trigger graceful shutdown from simulator thread."""
        self.logger.info("Triggering graceful shutdown from simulator thread")
        self._shutdown_evt.set()
        # Also notify event sink for GUI thread
        try:
            if self._evt_cb:
                self._evt_cb("shutdown_request", "", "")
        except Exception as e:
            self.logger.warning(f"Error notifying GUI thread of shutdown: {e}")
            # Continue with shutdown even if GUI notification fails

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
        # Never call Genesis from the UI thread; return cache
        return self._ee_pose_cache.get(robot)

    # Thread safety helper
    def _assert_sim_thread(self):
        if threading.current_thread().name != "simforge-sim":
            raise RuntimeError("Genesis access from non-sim thread")

    def wait_until_built(self, timeout=None) -> bool:
        return self._built_evt.wait(timeout)

    # Per-robot effective control (fallback to global)
    def _ctrl_for(self, robot_name: str):
        try:
            return self.config.control_for(robot_name)
        except Exception:
            # Fallback for older configs without control_for
            return getattr(self.config, "control", None)

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

    @staticmethod
    def _path_to_numpy(path):
        """
        Normalize Genesis path output to np.ndarray [K, dof].
        Accepts:
          - torch.Tensor [K, dof]
          - list[torch.Tensor[dof]] / tuple[…]
          - list[list/np.ndarray]
          - None / []
        """
        if path is None:
            return np.zeros((0, 0), dtype=np.float32)

        # Flatten tuple -> list to simplify handling
        if isinstance(path, tuple):
            path = list(path)

        try:
            import torch  # type: ignore
            if isinstance(path, torch.Tensor):
                arr = path.detach().cpu().numpy()
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                return arr.astype(np.float32, copy=False)

            if isinstance(path, (list, tuple)) and len(path) > 0:
                # If it's a list/tuple of tensors of identical shape, stack them
                if isinstance(path[0], torch.Tensor):
                    # Filter any empty items defensively
                    items = [t for t in path if isinstance(t, torch.Tensor) and t.numel() > 0]
                    if len(items) == 0:
                        return np.zeros((0, 0), dtype=np.float32)
                    arr = torch.stack(items).detach().cpu().numpy()
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    return arr.astype(np.float32, copy=False)
        except Exception:
            pass

        # Numpy/list fallback
        arr = np.asarray(path, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _densify_joint_path(self, q_path: np.ndarray, max_joint_delta: float = 0.04) -> np.ndarray:
        """Insert intermediate points so adjacent joints don't change more than max_joint_delta (radians)."""
        if q_path.shape[0] < 2:
            return q_path
        out = [q_path[0]]
        for a, b in zip(q_path[:-1], q_path[1:]):
            seg = b - a
            steps = max(1, int(np.max(np.abs(seg)) / max_joint_delta))
            for s in range(1, steps + 1):
                out.append(a + seg * (s / steps))
        result = np.asarray(out, dtype=np.float32)
        # Clamp to at least 2 waypoints to prevent degenerate paths
        if result.shape[0] < 2:
            result = np.linspace(q_path[0], q_path[-1], 2, dtype=np.float32)
        return result


    # --- Cartesian planning/execution ---
    def plan_and_execute_cartesian(self, robot: str, pose_xyzrpy: tuple, frame: str = 'base'):
        """
        Compute IK for target pose and execute a smooth joint-space trajectory.
        Falls back to linear joint interpolation if planning backend is unavailable.
        """
        self._assert_sim_thread()
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

        # Current joints as starting state - match old codebase simple approach
        try:
            q_start = np.array([j.qpos for j in entity.get_joints()], dtype=np.float32)
        except Exception:
            q_start = np.zeros((len(dofs_idx),), dtype=np.float32)
        # If Genesis hasn't yet populated joint readings, use last hold target
        if not np.any(np.abs(q_start) > 1e-6):
            hold_deg = self._targets_deg.get(robot)
            if hold_deg:
                q_start = np.array([np.deg2rad(v) for v in hold_deg], dtype=np.float32)

        # Build base-frame target and convert to world if needed
        x, y, z, roll, pitch, yaw = pose_xyzrpy
        q_local_wxyz = self._euler_deg_to_wxyz(roll, pitch, yaw)
        robot_cfg = next(r for r in self.config.robots if r.name == robot)
        pos_w, quat_w_wxyz = self._apply_frame(robot_cfg, (x, y, z), q_local_wxyz, frame)
        target_pos = (float(pos_w[0]), float(pos_w[1]), float(pos_w[2]))
        # Many APIs expect XYZW; also prepare WXYZ
        target_quat_xyzw = (float(quat_w_wxyz[1]), float(quat_w_wxyz[2]), float(quat_w_wxyz[3]), float(quat_w_wxyz[0]))
        target_quat_wxyz = (float(quat_w_wxyz[0]), float(quat_w_wxyz[1]), float(quat_w_wxyz[2]), float(quat_w_wxyz[3]))

        # Determine end-effector link name and get IK solution
        ee_name = self._ee_link_name.get(robot)
        end_eff_link = entity.get_link(ee_name) if ee_name else None

        started_at = time.perf_counter()
        q_goal = None
        used_solver = None

        # Try Genesis IK first
        if end_eff_link and hasattr(entity, 'inverse_kinematics'):
            try:
                q_goal = entity.inverse_kinematics(
                    link=end_eff_link,
                    pos=np.array(target_pos, dtype=np.float32),
                    quat=np.array(target_quat_xyzw, dtype=np.float32),
                )
                used_solver = 'entity.inverse_kinematics'
            except Exception:
                pass

        # Fallback: numeric IK
        if q_goal is None and ee_name:
            q_goal = self._numeric_ik(entity, ee_name, q_start, target_pos, target_quat_wxyz)
            if q_goal is not None:
                used_solver = 'numeric_ik'

        if q_goal is None:
            self.logger.warning("IK failed (robot=%s); rejecting Cartesian command", robot)
            self._emit("cartesian_failed", robot, "ik_failed")
            return False

        # Normalize IK result to CPU numpy for downstream consumption (avoid CUDA tensor -> numpy errors)
        try:
            q_goal_np = self._to_np(q_goal).astype(np.float32)
        except Exception:
            try:
                import torch  # type: ignore
                if isinstance(q_goal, torch.Tensor):
                    q_goal_np = q_goal.detach().cpu().numpy().astype(np.float32)
                else:
                    q_goal_np = np.array(q_goal, dtype=np.float32)
            except Exception:
                q_goal_np = np.array(q_goal, dtype=np.float32)

        # Optional IK logging for debugging
        self.logger.debug("IK solver=%s; q_goal dtype=%s shape=%s", used_solver, getattr(q_goal_np, 'dtype', None), getattr(q_goal_np, 'shape', None))
        self.logger.debug("IK solver=%s, q_start=%s", used_solver, q_start)

        ctrl = self._ctrl_for(robot)
        if getattr(ctrl, 'collision_check', True):
            if self._world is not None:
                ok, reason = self._world.check_state(robot, q_goal_np, dofs_idx)
                if not ok:
                    self.logger.warning("IK pose in collision (%s); rejecting Cartesian command for %s", reason, robot)
                    self._emit("cartesian_failed", robot, "ik_in_collision")
                    return False
            elif self._colliders.get(robot) is not None:
                # fallback to per-robot if world not available
                coll = self._colliders[robot]
                ok, reason = coll.check_state(q_goal_np, dofs_idx)
                if not ok:
                    self.logger.warning("IK pose in collision (%s); rejecting Cartesian command for %s", reason, robot)
                    self._emit("cartesian_failed", robot, "ik_in_collision")
                    return False

        # Ensure planner uses q_start as the current state
        try:
            entity.set_dofs_position(q_start, dofs_idx_local=dofs_idx)
        except Exception:
            entity.set_dofs_position(q_start, dofs_idx)
        # one light step to flush transforms
        try:
            self._scene.step()
        except Exception:
            pass

        # --- OMPL planning with enforced timeout/retry limits ---
        waypoints = None
        if getattr(ctrl, 'strict_cartesian', True):
            if not hasattr(entity, 'plan_path'):
                self.logger.warning("OMPL/plan_path not available; set control.strict_cartesian=false to test without planning")
                self._emit("cartesian_failed", robot, "planner_unavailable")
                return False

            # Get configuration parameters with strict type checking
            res = float(getattr(ctrl, 'planner_resolution', 0.03))
            planner_timeout = float(getattr(ctrl, 'planner_timeout', 1.5))
            planner_max_retry = int(getattr(ctrl, 'planner_max_retry', 3))
            num_wp = max(2, int(getattr(ctrl, 'cartesian_waypoints', 100)))
            planner_name = str(getattr(ctrl, 'planner', 'RRTConnect'))

            self.logger.debug("Planning config: timeout=%.1fs, max_retry=%d, waypoints=%d, planner=%s",
                             planner_timeout, planner_max_retry, num_wp, planner_name)

            # --- Prepare inputs (strict NumPy types expected by Genesis) ---
            q_start_np1d = np.asarray(q_start, dtype=np.float32).ravel()
            q_goal_np1d  = np.asarray(q_goal_np, dtype=np.float32).ravel()

            # Prepare plan_path call parameters
            plan_params = {
                'qpos_goal': q_goal_np1d,
                'qpos_start': q_start_np1d,
                'resolution': res,
                'timeout': planner_timeout,
                'max_retry': planner_max_retry,
                'smooth_path': True,
                'num_waypoints': num_wp,
                'ignore_collision': False,
                'planner': planner_name,
            }

            # Use Genesis's built-in timeout and retry capabilities (no manual wrapping needed)
            self._planning = True
            t0 = time.perf_counter()

            try:
                # Genesis handles timeout and retries internally via plan_params
                path = entity.plan_path(**plan_params)
                plan_dt = time.perf_counter() - t0
                self.logger.debug("Planning completed in %.3fs", plan_dt)
            except Exception as e:
                plan_dt = time.perf_counter() - t0
                self.logger.debug("Planning failed after %.3fs: %s", plan_dt, str(e)[:100])
                path = None
            finally:
                self._planning = False

            # Validate the result
            if path is not None:
                p_np = self._path_to_numpy(path)

                self.logger.debug("Planned path: type=%s dtype=%s shape=%s",
                                type(path), getattr(p_np, 'dtype', None), getattr(p_np, 'shape', None))
                self.logger.debug("Path waypoints: first=%s last=%s", p_np[0] if len(p_np) > 0 else "none", p_np[-1] if len(p_np) > 0 else "none")

                # Validate shape and quality
                if (p_np.size > 0 and p_np.shape[0] >= 2 and
                    p_np.shape[1] == q_start_np1d.shape[0] and
                    not np.allclose(p_np[0], p_np[-1], atol=1e-6)):
                    # Path validated successfully - densify for smooth execution
                    waypoints = self._densify_joint_path(p_np, max_joint_delta=0.04)
                    self.logger.debug("Planning successful: %d waypoints generated", len(waypoints))
                else:
                    self.logger.error("Path planner failed: IK succeeded but planner returned invalid path (all waypoints at same position)")
                    self._emit("cartesian_failed", robot, "planner_invalid_path")
                    return False
            else:
                self.logger.error("Path planner failed: All planning attempts failed within timeout")
                self._emit("cartesian_failed", robot, "planner_failed")
                return False
        else:
            # Planning disabled - reject command entirely
            self.logger.error("Path planner failed: Cartesan planning disabled (strict_cartesian=false); cannot execute movement without planning")
            self._emit("cartesian_failed", robot, "planning_disabled")
            return False

        if getattr(ctrl, 'collision_check', True):
            if self._world is not None:
                t_verify = time.perf_counter()
                # keep it fast: sample ~30 segments, 3 substeps each, <0.2s budget
                W = max(2, int(waypoints.shape[0]))
                waypoints_list = [waypoints[i] for i in range(W)]
                ok, reason, seg = self._world.check_path(
                    robot,
                    waypoints_list,
                    dofs_idx,
                    substeps=getattr(ctrl, 'ccd_substeps', 3),
                    stride=max(1, W // 30),
                    max_time_s=getattr(ctrl, 'postcheck_time_s', 0.2),
                )
                verify_dt = time.perf_counter() - t_verify
                self.logger.debug("World postcheck time: %.3f", verify_dt)
                if not ok:
                    self.logger.warning("Planned path collides at segment %d (%s); rejecting", seg, reason)
                    self._emit("cartesian_failed", robot, "path_in_collision")
                    return False
            elif self._colliders.get(robot) is not None:
                # fallback to per-robot if world not available
                coll = self._colliders[robot]
                t_verify = time.perf_counter()
                # keep it fast: sample ~30 segments, 3 substeps each, <0.2s budget
                W = max(2, int(waypoints.shape[0]))
                waypoints_list = [waypoints[i] for i in range(W)]
                ok, reason, seg = coll.check_path(
                    waypoints_list,
                    dofs_idx,
                    substeps=getattr(ctrl, 'ccd_substeps', 3),
                    stride=max(1, W // 30),
                    max_time_s=getattr(ctrl, 'postcheck_time_s', 0.2),
                )
                verify_dt = time.perf_counter() - t_verify
                self.logger.debug("Post: verify time: %.3f", verify_dt)
                if not ok:
                    self.logger.warning("Planned path collides at segment %d (%s); rejecting", seg, reason)
                    self._emit("cartesian_failed", robot, "path_in_collision")
                    return False

        # Debug: Check path quality
        if waypoints.shape[0] > 1:
            d = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
            self.logger.debug("First 10 waypoint distances: avg=%.4f min=%.4f max=%.4f",
                            float(np.mean(d[:10])), float(np.min(d[:10])), float(np.max(d[:10])))

        self._active_traj[robot] = {"waypoints": waypoints, "i": 0, "current": waypoints[0]}
        self.logger.info("Executing Cartesian move for %s with %d waypoints (mode=joints)", robot, int(waypoints.shape[0]))
        self._emit("cartesian_executing", robot, "")
        return True
