from __future__ import annotations

import threading
import queue
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import commands
from .config import SimforgeConfig, RobotConfig
from .logging_utils import setup_logging
from .urdf_utils import parse_joint_limits, select_end_effector_link, get_transform_to_link
import pinocchio as pin
import pybullet as p
from .pybullet_wrapper import PybulletPlanner
from .genesis_wrapper import GenesisWrapper
from .utils import (
    rpy_to_quat_wxyz,
    quat_wxyz_multiply,
    quat_wxyz_rotate_vec,
    quat_wxyz_to_rotation_matrix,
    to_numpy,
    safe_set_dofs_position,
    quat_wxyz_conj,
    quat_wxyz_normalize,
    quat_wxyz_to_rotvec,
    rotation_matrix_to_quat_wxyz,
    quat_wxyz_to_xyzw,
    rotation_matrix_to_euler_rpy,
)


# Constants for IK and planning tolerances
MAX_IK_ITERS = 200
IK_TOL_POS = 5e-3
IK_TOL_ROT_DEG = 3.0


class Simulator:
    """
    Main simulator class that manages the Genesis scene, robots, collision checking,
    IK planning, and trajectory execution in real-time.

    This class handles:
    - Genesis backend initialization (GPU, CPU)
    - Scene construction from configuration
    - Robot loading and IK setup
    - Cartesian and joint-space path planning
    - Real-time trajectory execution on a background thread
    - Thread-safe communication with the GUI and controllers

    Args:
        config: Configuration object containing scene, robots, and control parameters
        debug: Enable debug logging
    """
    def __init__(self, config: SimforgeConfig, debug: bool = False) -> None:
        # Final, explicit validation to ensure all sub-models are correctly instantiated.
        self.config = SimforgeConfig.model_validate(config.model_dump())
        self.logger = setup_logging(debug)
        self.gs = GenesisWrapper(
            backend_cfg=(self.config.scene.backend or "gpu"), logger=self.logger
        )
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
        # --- Dynamic Tooling ---
        # robot_name -> tool_name
        self._attached_tool: Dict[str, str] = {}
        # robot_name -> tool's TCP link name
        self._tool_tcp_link: Dict[str, str] = {}
        # robot_name -> (pos, quat) transform from robot flange to tool TCP
        self._tool_tcp_transform: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        # tool_name -> genesis entity
        self._tool_entities: Dict[str, object] = {}

        # Event sink for GUI notifications
        self._evt_cb = None
        # Optional UI-thread executor for Genesis ops that must run on main thread
        self._ui_exec = None
        # Planning guard to avoid stepping during heavy kernels on UI thread
        self._planning = False
        self._planners: Dict[str, PybulletPlanner] = {}
        # Cache for EE poses (updated on sim thread, read from UI thread)
        self._ee_pose_cache: Dict[str, tuple] = {}
        # Pinocchio kinematics (per robot)
        self._pin_models: Dict[str, pin.Model] = {}
        self._pin_datas: Dict[str, pin.Data] = {}
        self._pin_ee_frame: Dict[str, int] = {}
        self._pin_limits: Dict[str, List[tuple]] = {}
        # Built event for startup synchronization
        self._built_evt = threading.Event()
        # Shutdown synchronization event for GUI thread
        self._shutdown_evt = threading.Event()

        # Command queue to run IK/planning on the sim thread
        self._cmd_q: "queue.Queue[commands.Command]" = queue.Queue()
        self._last_enq_time: float = 0.0

    def build_scene(self):
        if self._built:
            return

        # Patch Genesis cleanup immediately after initialization
        try:
            from .main import _patch_genesis_cleanup
            _patch_genesis_cleanup()
            self.logger.debug("Genesis cleanup patching applied")
        except Exception as e:
            self.logger.warning(f"Could not patch Genesis cleanup in simulator: {e}")

        scene = self.gs.create_scene(self.config.scene)
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
                    self.gs.morphs.Plane(
                        pos=tuple(obj.position),
                    )
                )
            elif obj.type == "box" and obj.size is not None:
                scene.add_entity(self.gs.morphs.Box(
                    pos=tuple(obj.position),
                    size=tuple(obj.size),
                    euler=tuple(getattr(obj, "orientation_rpy", (0.0,0.0,0.0)))
                ))
            
        # Robots
        for r in self.config.robots:
            self._add_robot(scene, r)

        # Add tool entities before building the scene.
        for r in self.config.robots:
            if r.end_effector and r.end_effector_link:
                tool_name = f"{r.name}_tool"
                tool_entity = scene.add_entity(self.gs.morphs.URDF(file=r.end_effector, pos=(0,0,0), euler=(0,0,0), fixed=False))
                self._tool_entities[tool_name] = tool_entity
        
        scene.build()

        # Weld tool entities to robots after building the scene.
        for r in self.config.robots:
            if r.end_effector and r.end_effector_link:
                tool_name = f"{r.name}_tool"
                robot_entity = self._robots[r.name]
                tool_entity = self._tool_entities[tool_name]
                robot_link = robot_entity.get_link(r.end_effector_link)
                
                # Get the base link of the tool.
                import xml.etree.ElementTree as ET
                tool_tree = ET.parse(r.end_effector)
                tool_root = tool_tree.getroot()
                tool_base_link_element = tool_root.find("link")
                if tool_base_link_element is None:
                    raise ValueError("Tool URDF must have at least one link.")
                tool_base_link_name = tool_base_link_element.get("name")
                tool_base_link = tool_entity.get_link(tool_base_link_name)
                
                # Create a fixed joint using the weld constraint.
                rigid_solver = scene.sim.rigid_solver
                link_a = np.array([tool_base_link.idx], dtype=self.gs.np_int)
                link_b = np.array([robot_link.idx], dtype=self.gs.np_int)
                rigid_solver.add_weld_constraint(link_a, link_b)
                
                self._attached_tool[r.name] = tool_name
                tcp_link = select_end_effector_link(r.end_effector)
                self._tool_tcp_link[r.name] = tcp_link



                # Cache the transform.
                if tcp_link:
                    tcp_transform = get_transform_to_link(r.end_effector, tcp_link)
                    if tcp_transform:
                        self._tool_tcp_transform[r.name] = tcp_transform
                        self.logger.info(f"Tool '{tool_name}' attached to '{r.name}/{r.end_effector_link}'. TCP set to '{tcp_link}'.")
                    else:
                        self.logger.warning(f"Could not resolve TCP link '{tcp_link}' in '{r.end_effector}'. IK will solve for flange.")
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
            entity = scene.add_entity(self.gs.morphs.URDF(**urdf_kwargs))
        except TypeError:
            # Older Genesis may not support 'vis_mode'; retry without it
            urdf_kwargs.pop("vis_mode", None)
            entity = scene.add_entity(self.gs.morphs.URDF(**urdf_kwargs))

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

        # Tool attachment is handled after the scene is built.
        # Configure planner with clearance threshold if available
        min_clearance = 0.0
        try:
            min_clearance = float(getattr(ctrl, "min_clearance_m", 0.0) or 0.0)
        except Exception:
            pass
        # Build joint limit map by name from URDF for robust planning bounds
        limits = {}
        try:
            jl = parse_joint_limits(robot_cfg.urdf)
            for d in jl:
                name = d.get('name')
                if name is not None:
                    limits[name] = (d.get('lower'), d.get('upper'))
        except Exception:
            limits = {}

        self._planners[robot_cfg.name] = PybulletPlanner(
            robot_cfg.urdf,
            robot_cfg.fixed_base,
            min_clearance=min_clearance,
            base_position=tuple(robot_cfg.base_position),
            base_rpy_deg=tuple(robot_cfg.base_orientation),
            joint_limits_by_name=limits,
        )

        # Initialize Pinocchio model for IK (build once per robot)
        try:
            mdl = pin.buildModelFromUrdf(robot_cfg.urdf)
            dat = mdl.createData()
            self._pin_models[robot_cfg.name] = mdl
            self._pin_datas[robot_cfg.name] = dat
            # Joint limits from URDF (Pinocchio model order)
            jlims: List[tuple] = []
            for i in range(mdl.nq):
                # For typical 6-DOF arms, nq==nv and each DoF is a revolute with bounds in model.lowerPositionLimit/upper
                lo = float(mdl.lowerPositionLimit[i]) if i < len(mdl.lowerPositionLimit) else float('-inf')
                hi = float(mdl.upperPositionLimit[i]) if i < len(mdl.upperPositionLimit) else float('inf')
                jlims.append((lo, hi))
            self._pin_limits[robot_cfg.name] = jlims
        except Exception as e:
            self.logger.warning(f"Pinocchio model init failed for {robot_cfg.name}: {e}")
 
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



    def detach_tool(self, robot_name: str):
        """Detach the current tool from a robot."""
        if robot_name not in self._attached_tool:
            return
        
        tool_name = self._attached_tool.pop(robot_name, None)
        if not tool_name:
            return


        # 2. Remove from scene (TODO: Genesis needs an API to remove entities/joints)
        # For now, we just orphan the entity and clear our references.
        self._tool_entities.pop(tool_name, None)
        
        # 3. Clear kinematic info
        self._tool_tcp_link.pop(robot_name, None)
        self._tool_tcp_transform.pop(robot_name, None)
        
        self.logger.info(f"Detached tool '{tool_name}' from robot '{robot_name}'.")


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
            self._cmd_q.put(commands.CartesianMove(robot, pose_xyzrpy, frame), block=False)
            self.logger.debug("Enqueued cartesian_move for %s: %s in %s", robot, pose_xyzrpy, frame)
        except Exception as e:
            self.logger.error("Failed to enqueue cartesian_move: %s", e)

    def _loop(self):
        # Scene is now built on the main thread before starting the simulation loop.
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
                    cmd = self._cmd_q.get_nowait()
                    if isinstance(cmd, commands.CartesianMove):
                        # skip moves for robots that were canceled in this drain
                        if cmd.robot in cancels:
                            continue
                        last_cmd = cmd  # keep only the latest move
                    elif isinstance(cmd, commands.CartesianCancel):
                        cancels.add(cmd.robot)
                        # stop active trajectory and drop pending moves for this robot
                        self._active_traj.pop(cmd.robot, None)
                # no break; rely on exception
            except queue.Empty:
                pass
            if last_cmd is not None:
                try:
                    self.plan_and_execute_cartesian(last_cmd.robot, last_cmd.pose_xyzrpy, last_cmd.frame)
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
                    pos = to_numpy(link.get_pos()).astype(np.float32)
                    quat = to_numpy(link.get_quat()).astype(np.float32)  # wxyz
                    roll_deg, pitch_deg, yaw_deg = self._quat_wxyz_to_euler_deg(quat)
                    self._ee_pose_cache[name] = (
                        float(pos[0]), float(pos[1]), float(pos[2]),
                        roll_deg, pitch_deg, yaw_deg,
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
                        traj_info = self._active_traj.pop(name, {})
                        # Set hold target to final pose (deg)
                        q_deg = [float(np.rad2deg(v)) for v in waypoints[-1]]
                        self._targets_deg[name] = q_deg
                        self.logger.debug("Trajectory complete for %s", name)

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
        # Close any planner backends (e.g., PyBullet DIRECT clients)
        try:
            for name, planner in list(self._planners.items()):
                try:
                    if hasattr(planner, 'close'):
                        planner.close()
                except Exception:
                    pass
        except Exception:
            pass

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

    def _euler_deg_to_wxyz(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        return rpy_to_quat_wxyz(np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw))

    def _quat_wxyz_to_euler_deg(self, quat_wxyz: np.ndarray) -> tuple:
        """Convert quaternion (wxyz) to Euler angles in degrees."""
        w, x, y, z = quat_wxyz
        t0 = 2*(w*x + y*z); t1 = 1 - 2*(x*x + y*y)
        roll = np.arctan2(t0, t1)
        t2 = np.clip(2*(w*y - z*x), -1.0, 1.0)
        pitch = np.arcsin(t2)
        t3 = 2*(w*z + x*y); t4 = 1 - 2*(y*y + z*z)
        yaw = np.arctan2(t3, t4)
        return (
            float(np.rad2deg(roll)),
            float(np.rad2deg(pitch)),
            float(np.rad2deg(yaw))
        )

    def _get_robot_joint_state(self, robot_name: str) -> np.ndarray:
        """Get current joint state for a robot, with fallback to targets."""
        entity = self._robots.get(robot_name)
        if not entity:
            return np.array([], dtype=np.float32)
        
        try:
            return np.array([j.qpos for j in entity.get_joints()], dtype=np.float32)
        except Exception:
            # Fallback to target degrees converted to radians
            target_deg = self._targets_deg.get(robot_name, [])
            if target_deg:
                return np.deg2rad(np.array(target_deg, dtype=np.float32))
            return np.array([], dtype=np.float32)

    def _find_robot_config(self, robot_name: str) -> RobotConfig | None:
        """Find robot configuration by name."""
        for r in self.config.robots:
            if isinstance(r, dict):
                if r.get('name') == robot_name:
                    return RobotConfig.model_validate(r)
            elif hasattr(r, 'name') and r.name == robot_name:
                return r
        return None

    def _create_ground_plane_obstacle(self, planner, silence_fn):
        """Create ground plane obstacle for planning."""
        try:
            ctrl_local = self._ctrl_for("") # Use empty string as fallback
            ground_z = float(getattr(ctrl_local, 'ground_plane_z', 0.0) or 0.0)
            ground_raise = float(getattr(ctrl_local, 'ground_clearance_m', 0.002) or 0.0)
        except Exception:
            ground_z = 0.0
            ground_raise = 0.0
        
        half_extents = [50.0, 50.0, 0.005]
        top_z = float(ground_z) + float(ground_raise)
        ground_pos = [0.0, 0.0, top_z - half_extents[2]]
        plane_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=planner.client)
        
        with silence_fn():
            plane_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=plane_shape,
                basePosition=ground_pos,
                baseOrientation=[0,0,0,1],
                physicsClientId=planner.client
            )
        return plane_id

    def _create_box_obstacles(self, planner, silence_fn):
        """Create box obstacles from config objects."""
        obstacles = []
        for obj in self.config.objects:
            if obj.type == "box" and obj.size is not None:
                collision_shape = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[s/2.0 for s in obj.size],
                    physicsClientId=planner.client
                )
                
                rpy_deg = getattr(obj, 'orientation_rpy', (0.0, 0.0, 0.0))
                rpy_rad = tuple(np.deg2rad(r) for r in rpy_deg)
                quat = p.getQuaternionFromEuler(rpy_rad, physicsClientId=planner.client)
                
                with silence_fn():
                    body_id = p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=collision_shape,
                        basePosition=obj.position,
                        baseOrientation=quat,
                        physicsClientId=planner.client
                    )
                obstacles.append(body_id)
        return obstacles

    def _create_robot_obstacles(self, planner, silence_fn, exclude_robot: str = None):
        """Create robot obstacles from other robots in their current states."""
        obstacles = []
        for robot_name, robot_entity in self._robots.items():
            if robot_name == exclude_robot:
                continue

            joint_state = self._get_robot_joint_state(robot_name)
            if joint_state.size == 0:
                continue

            robot_cfg = self._find_robot_config(robot_name)
            if not robot_cfg:
                continue

            other_rpy = tuple(np.deg2rad(v) for v in robot_cfg.base_orientation)
            other_quat = p.getQuaternionFromEuler(other_rpy, physicsClientId=planner.client)

            with silence_fn():
                robot_id = p.loadURDF(
                    robot_cfg.urdf,
                    basePosition=robot_cfg.base_position,
                    baseOrientation=other_quat,
                    useFixedBase=robot_cfg.fixed_base,
                    physicsClientId=planner.client
                )

            joint_count = p.getNumJoints(robot_id, physicsClientId=planner.client)
            for j in range(min(len(joint_state), joint_count)):
                joint_info = p.getJointInfo(robot_id, j, physicsClientId=planner.client)
                if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    p.resetJointState(robot_id, j, joint_state[j], physicsClientId=planner.client)

            obstacles.append(robot_id)
        return obstacles

    def _create_all_obstacles(self, planner, silence_fn, exclude_robot: str):
        """Create all environment obstacles for planning."""
        obstacles = []
        obstacles_meta = []
        created_body_ids = []
        obj_count, robot_count = 0, 0

        # Ground plane
        plane_id = self._create_ground_plane_obstacle(planner, silence_fn)
        obstacles.append(plane_id)
        obstacles_meta.append((plane_id, 'ground', 'ground'))
        created_body_ids.append(plane_id)
        planner.set_clearance_exempt_ids([plane_id])

        # Static objects
        box_obstacles = self._create_box_obstacles(planner, silence_fn)
        for body_id in box_obstacles:
            obstacles.append(body_id)
            obstacles_meta.append((body_id, 'object', 'box'))
            created_body_ids.append(body_id)
            obj_count += 1

        # Other robots
        robot_obstacles = self._create_robot_obstacles(planner, silence_fn, exclude_robot)
        for i, body_id in enumerate(robot_obstacles):
            other_robot_names = [name for name in self._robots.keys() if name != exclude_robot]
            robot_name = other_robot_names[i] if i < len(other_robot_names) else f"robot_{i}"
            obstacles.append(body_id)
            obstacles_meta.append((body_id, 'robot', robot_name))
            created_body_ids.append(body_id)
            robot_count += 1

        return obstacles, obstacles_meta, created_body_ids, obj_count, robot_count

    def _base_pose(self, robot_cfg: RobotConfig):
        t = np.array(robot_cfg.base_position, dtype=np.float32)
        q = self._euler_deg_to_wxyz(*robot_cfg.base_orientation)
        return t, q

    def _apply_frame(self, robot_cfg: RobotConfig, target_pos_base, target_quat_base_wxyz, frame: str):
        # Check if the pose is actually meant to be in world coordinates despite specifying "base"
        # This appears to be the case based on pose values and robot workspace analysis

        # Simple heuristic: if the target position is far from origin and likely represents world coordinates
        pos_magnitude = np.linalg.norm(target_pos_base)
        base_magnitude = np.linalg.norm(robot_cfg.base_position)

        # If target pose is much farther from origin than base position, it's likely already in world coordinates
        if pos_magnitude > 1.5 * base_magnitude + 0.5:  # threshold accounts for reasonable robot reach
            # Treat as world coordinates - no transformation needed
            return target_pos_base, target_quat_base_wxyz

        if frame == 'base':
            # Standard transformation for genuinely base-frame coordinates
            # Get base pose in world frame
            t_base, q_base_wxyz = self._base_pose(robot_cfg)

            # Convert from base coordinates to world coordinates
            pos_local = np.array(target_pos_base, dtype=np.float32)
            quat_local_wxyz = np.array(target_quat_base_wxyz, dtype=np.float32)

            # Position: world_pos = base_position + rotation(base_to_world) @ local_pos
            # Note: q_base_wxyz is rotation from world to base, so inverse is from base to world
            q_base_to_world = quat_wxyz_conj(q_base_wxyz)
            pos_world = t_base + quat_wxyz_rotate_vec(q_base_to_world, pos_local)

            # Orientation: world_quat = rotation(base_to_world) @ local_quat
            quat_world_wxyz = quat_wxyz_multiply(q_base_to_world, quat_local_wxyz)

            return tuple(pos_world.tolist()), tuple(quat_world_wxyz.tolist())
        return target_pos_base, target_quat_base_wxyz

    # Pose helpers
    def _get_ee_pose_w(self, entity, ee_name: str):
        link = entity.get_link(ee_name)
        pos = link.get_pos()
        quat = link.get_quat()  # wxyz
        pos = to_numpy(pos)
        quat = to_numpy(quat)
        return pos.astype(np.float32), quat.astype(np.float32)

    # Pinocchio-based SE(3) IK with joint limits (DLS)
    def _pin_ik(self, robot: str, q0: np.ndarray, pos_w: tuple, quat_w_wxyz: tuple,
                iters: int = 100, damping: float = 1e-3,
                w_pos: float = 1.0, w_rot: float = 1.0,
                pos_tol: float = 5e-3, rot_tol_deg: float = 3.0) -> np.ndarray | None:
        mdl = self._pin_models.get(robot); dat = self._pin_datas.get(robot)
        ee_name = self._ee_link_name.get(robot)
        if mdl is None or dat is None or not ee_name:
            return None
        try:
            ee_fid = mdl.getFrameId(ee_name)
        except Exception:
            return None
        target_p = np.array(pos_w, dtype=np.float32)
        Rw = quat_wxyz_to_rotation_matrix(np.array(quat_w_wxyz, dtype=np.float32))
        M_target = pin.SE3(Rw.astype(np.float64), target_p.astype(np.float64))
        q = q0.astype(np.float64).copy()
        limits = self._pin_limits.get(robot, [])

        rot_tol = np.deg2rad(rot_tol_deg)
        prev_err = None
        lam = float(damping)
        for k in range(iters):
            pin.forwardKinematics(mdl, dat, q)
            pin.updateFramePlacements(mdl, dat)
            M_cur = dat.oMf[ee_fid]
            e_trans = (M_target.translation - M_cur.translation)
            R_err = M_target.rotation @ M_cur.rotation.T
            e_rot = pin.log3(R_err)
            err6 = np.hstack([w_pos * e_trans, w_rot * e_rot])
            if np.linalg.norm(e_trans) < pos_tol and np.linalg.norm(e_rot) < rot_tol:
                break
            J6 = pin.computeFrameJacobian(mdl, dat, q, ee_fid, pin.ReferenceFrame.WORLD)
            # Weighted DLS solve: dq = (J^T J + lambda^2 I)^-1 J^T err
            J = np.asarray(J6)
            H = J.T @ J + (lam * np.eye(J.shape[1]))
            dq = np.linalg.solve(H, J.T @ err6)
            # Integrate on manifold for correctness
            q = pin.integrate(mdl, q, dq)
            # Clamp to joint limits if available
            if limits and len(limits) == len(q):
                for i, (lo, hi) in enumerate(limits):
                    if np.isfinite(lo):
                        q[i] = max(q[i], lo)
                    if np.isfinite(hi):
                        q[i] = min(q[i], hi)
            # Simple adaptive damping: increase if error stagnates or grows
            cur_err = float(np.linalg.norm(err6))
            if prev_err is not None and cur_err > prev_err * 0.99:
                lam = min(lam * 5.0, 1e2)
            else:
                lam = max(lam * 0.9, damping)
            prev_err = cur_err
        # Final check
        pin.forwardKinematics(mdl, dat, q)
        pin.updateFramePlacements(mdl, dat)
        M_cur = dat.oMf[ee_fid]
        e_trans = (M_target.translation - M_cur.translation)
        e_rot = pin.log3(M_target.rotation @ M_cur.rotation.T)
        if np.linalg.norm(e_trans) < pos_tol and np.linalg.norm(e_rot) < rot_tol:
            return q.astype(np.float32)
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
        try:
            self._assert_sim_thread()
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
            # Force a re-validation of the config to ensure correct types.
            self.config = SimforgeConfig.model_validate(self.config.model_dump())
            
            q_start = self._get_robot_joint_state(robot)
            if q_start.size == 0:
                q_start = np.zeros((len(dofs_idx),), dtype=np.float32)
            # If Genesis hasn't yet populated joint readings, use last hold target
            if not np.any(np.abs(q_start) > 1e-6):
                hold_deg = self._targets_deg.get(robot)
                if hold_deg:
                    q_start = np.array([np.deg2rad(v) for v in hold_deg], dtype=np.float32)

            # Build base-frame target and convert to world if needed
            x, y, z, roll, pitch, yaw = pose_xyzrpy
            q_local_wxyz = self._euler_deg_to_wxyz(roll, pitch, yaw)
            
            robot_cfg = self._find_robot_config(robot)
            if not robot_cfg:
                self.logger.error(f"Configuration for robot '{robot}' not found.")
                return
            pos_w, quat_w_wxyz = self._apply_frame(robot_cfg, (x, y, z), q_local_wxyz, frame)
            
            # If a tool is attached, adjust the target pose from the TCP to the robot's flange
            if robot in self._tool_tcp_transform:
                tcp_pos, tcp_quat = self._tool_tcp_transform[robot]
                # We want T_world_tcp = T_world_flange * T_flange_tcp
                # So, T_world_flange = T_world_tcp * T_tcp_flange
                # T_tcp_flange is the inverse of T_flange_tcp
                tcp_quat_inv = quat_wxyz_conj(tcp_quat)
                tcp_pos_inv = -quat_wxyz_rotate_vec(tcp_quat_inv, tcp_pos)

                # Adjust the world target
                final_pos_w = quat_wxyz_rotate_vec(np.array(quat_w_wxyz), tcp_pos_inv) + np.array(pos_w)
                final_quat_w_wxyz = quat_wxyz_multiply(np.array(quat_w_wxyz), tcp_quat_inv)
                
                pos_w = tuple(final_pos_w.tolist())
                quat_w_wxyz = tuple(final_quat_w_wxyz.tolist())
                try:
                    self.logger.debug(f"Adjusting IK target for tool TCP. New flange target: {pos_w}")
                except Exception:
                    pass

            target_pos = (float(pos_w[0]), float(pos_w[1]), float(pos_w[2]))
            # Many APIs expect XYZW; also prepare WXYZ
            target_quat_xyzw = (float(quat_w_wxyz[1]), float(quat_w_wxyz[2]), float(quat_w_wxyz[3]), float(quat_w_wxyz[0]))
            target_quat_wxyz = (float(quat_w_wxyz[0]), float(quat_w_wxyz[1]), float(quat_w_wxyz[2]), float(quat_w_wxyz[3]))

            # Determine end-effector link name and get IK solution
            ee_name = self._ee_link_name.get(robot)
            end_eff_link = entity.get_link(ee_name) if ee_name else None

            
            # --- 1. IK Solution with Pinocchio ---
            used_solver = 'pinocchio'
            # Try multiple seeds to improve robustness across wrist/elbow branches
            seeds: List[np.ndarray] = []
            seeds.append(q_start.copy())
            # Helper to add a variant with +/- pi on selected indices if within limits
            def add_pi_variant(base: np.ndarray, idx: int):
                v = base.copy()
                if idx < len(v):
                    v[idx] = float(v[idx] + np.pi)
                seeds.append(v)
                v2 = base.copy()
                if idx < len(v2):
                    v2[idx] = float(v2[idx] - np.pi)
                seeds.append(v2)
            # Typical 6-DOF wrist: try flipping last 2–3 joints
            add_pi_variant(q_start, 3)
            add_pi_variant(q_start, 4)
            add_pi_variant(q_start, 5)
            # Small jitter seeds
            for a in (0.1, -0.1):
                seeds.append(q_start + a)

            q_goal_np = None
            for si, seed in enumerate(seeds):
                q_try = self._pin_ik(
                    robot, seed, target_pos, target_quat_wxyz,
                    iters=MAX_IK_ITERS, damping=5e-2, w_pos=1.0, w_rot=1.2,
                    pos_tol=IK_TOL_POS, rot_tol_deg=IK_TOL_ROT_DEG
                )
                if q_try is not None:
                    q_goal_np = q_try
                    if self._evt_cb:
                        try:
                            self.logger.debug(f"Pinocchio IK converged with seed {si} for {robot}")
                        except Exception:
                            pass
                    break
            
            if q_goal_np is None:
                self.logger.warning("IK failed (robot=%s); rejecting Cartesian command", robot)
                self._emit("cartesian_failed", robot, "ik_failed")
                return False

            # Optional IK logging for debugging
            self.logger.debug("IK solver=%s; q_goal dtype=%s shape=%s", used_solver, getattr(q_goal_np, 'dtype', None), getattr(q_goal_np, 'shape', None))
            self.logger.debug("IK solver=%s, q_start=%s", used_solver, q_start)

            ctrl = self._ctrl_for(robot)

            # No extra refinement; Pinocchio IK already enforces limits with DLS

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

            # Start-state validity will be checked during planning with PyBullet

            # Joint bounds check on IK result
            try:
                planner_for_limits = self._planners.get(robot)
                if planner_for_limits and hasattr(planner_for_limits, 'joint_limits'):
                    jl = list(getattr(planner_for_limits, 'joint_limits', []))
                    if len(jl) == len(q_goal_np):
                        eps = 1e-6
                        for i, (lo, hi) in enumerate(jl):
                            qi = float(q_goal_np[i])
                            if (qi < lo - eps) or (qi > hi + eps):
                                self.logger.warning("IK solution violates joint limits at idx %d: q=%.4f not in [%.4f, %.4f]", i, qi, lo, hi)
                                self._emit("cartesian_failed", robot, "ik_out_of_bounds")
                                return False
            except Exception:
                pass

            # --- 2. Motion Planning with Headless PyBullet ---
            waypoints = None
            self.logger.debug("Using PybulletPlanner for collision-aware motion planning.")
            
            planner = self._planners.get(robot)
            if not planner:
                self.logger.error(f"No planner found for robot {robot}")
                self._emit("cartesian_failed", robot, "planner_failed")
                return
                
            try:
                # Build all environment obstacles for planning
                # Helper: temporarily silence C++ stdout/stderr spam during PyBullet/OMPL calls
                import os, contextlib
                @contextlib.contextmanager
                def _silence_stdio():
                    try:
                        stdout_fd = os.dup(1)
                        stderr_fd = os.dup(2)
                        devnull_fd = os.open('/dev/null', os.O_WRONLY)
                        os.dup2(devnull_fd, 1)
                        os.dup2(devnull_fd, 2)
                        yield
                    finally:
                        try:
                            os.dup2(stdout_fd, 1)
                            os.dup2(stderr_fd, 2)
                            os.close(devnull_fd)
                            os.close(stdout_fd)
                            os.close(stderr_fd)
                        except Exception:
                            pass

                obstacles, obstacles_meta, created_body_ids, obj_added_count, robot_added_count = self._create_all_obstacles(planner, _silence_stdio, exclude_robot=robot)

                # Pre-check START and GOAL states against world using PyBullet collision
                # Start state check
                try:
                    start_ok = planner._is_collision_free(q_start.tolist(), obstacles)
                except Exception:
                    start_ok = True
                if not start_ok:
                    self.logger.warning("Start state in collision; rejecting Cartesian command for %s", robot)
                    self._emit("cartesian_failed", robot, "start_in_collision")
                    return False

                # Pre-check IK goal against world using PyBullet collision and log detailed reason
                try:
                    goal_ok = planner._is_collision_free(q_goal_np.tolist(), obstacles)
                except Exception:
                    goal_ok = True
                if not goal_ok:
                    self.logger.warning("IK goal pose is in collision; rejecting Cartesian command for %s", robot)
                    self._emit("cartesian_failed", robot, "ik_goal_in_collision")
                    return False

                self.logger.debug(f"OMPL planning with {len(obstacles)} obstacles ({obj_added_count} objects + {robot_added_count} robots)")
                # Find plane ID from obstacles_meta for clearance exemption
                plane_id = None
                for bid, kind, name in obstacles_meta:
                    if kind == 'ground':
                        plane_id = bid
                        break
                clearance_exempt = [plane_id] if plane_id is not None else []
                path = planner.plan_path(q_start.tolist(), q_goal_np.tolist(), obstacles=obstacles, clearance_exempt_ids=clearance_exempt)

                if path:
                    waypoints = np.array(path, dtype=np.float32)
                    self.logger.debug("PyBullet planning successful: %d waypoints generated", len(waypoints))
                else:
                    self.logger.warning("PyBullet failed to find a path.")
                    self._emit("cartesian_failed", robot, "path_in_collision")
                    return False

            except Exception as e:
                self.logger.error(f"PyBullet planning failed: {e}")
                self._emit("cartesian_failed", robot, "planner_failed")
                return False
            finally:
                # Clean up temporary obstacle bodies to avoid accumulation in the planner client
                try:
                    for bid in created_body_ids:
                        try:
                            p.removeBody(bid, physicsClientId=planner.client)
                        except Exception:
                            pass
                except Exception:
                    pass
                # Reset planner state to avoid cross-request residue
                try:
                    planner.set_clearance_exempt_ids([])
                    planner.obstacles = []
                except Exception:
                    pass


            # Compute and stash target world pose (EE frame used for planning) for end-of-trajectory error logging
            try:
                R_tgt = quat_wxyz_to_rotation_matrix(np.array(target_quat_wxyz, dtype=np.float32))
                rpy_rad = rotation_matrix_to_euler_rpy(R_tgt)
                target_pose_world_deg = (
                    float(pos_w[0]), float(pos_w[1]), float(pos_w[2]),
                    float(np.rad2deg(rpy_rad[0])), float(np.rad2deg(rpy_rad[1])), float(np.rad2deg(rpy_rad[2])),
                )
            except Exception:
                target_pose_world_deg = None

            # Debug: Check path quality
            if waypoints.shape[0] > 1:
                d = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
                self.logger.debug("First 10 waypoint distances: avg=%.4f min=%.4f max=%.4f",
                                float(np.mean(d[:10])), float(np.min(d[:10])), float(np.max(d[:10])))

            self._active_traj[robot] = {
                "waypoints": waypoints,
                "i": 0,
                "current": waypoints[0],
                "target_world_pose_deg": target_pose_world_deg,
            }
            self.logger.info("Executing Cartesian move for %s with %d waypoints (mode=joints)", robot, int(waypoints.shape[0]))
            self._emit("cartesian_executing", robot, "")
            return True
        
        except Exception as e:
            import traceback
            self.logger.error(f"plan_and_execute_cartesian detailed error: {e}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False
