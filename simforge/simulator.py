from __future__ import annotations

import threading
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import commands
from .config import SimforgeConfig, RobotConfig
from .logging_utils import setup_logging
from .urdf_utils import parse_joint_limits, select_end_effector_link, get_transform_to_link, merge_urdfs, get_urdf_root_link_name
from ikpy.chain import Chain
import pybullet as p
# FCL collision infrastructure removed for PyBullet-only planning
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


class Simulator:
    """
    Main simulator class that manages the Genesis scene, robots, collision checking,
    IK planning, and trajectory execution in real-time.

    This class handles:
    - Genesis backend initialization (GPU, CPU)
    - Scene construction from configuration
    - Robot loading and IK setup
    - Collision checking with FCL
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
        self._colliders: Dict[str, CollisionChecker] = {}
        self._world: Optional[CollisionWorld] = None  # world-aware collision checker
        self._planners: Dict[str, PybulletPlanner] = {}
        # Cache for EE poses (updated on sim thread, read from UI thread)
        self._ee_pose_cache: Dict[str, tuple] = {}
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

                # Add to collision checker.
                # This needs to be done after the weld, so the tool is in the correct position.
                # We need to step the simulator a few times to ensure the weld is processed.
                for _ in range(5):
                    scene.step()

                collider = self._colliders.get(r.name)
                if collider:
                    collider.add_tool(tool_name, r.end_effector, r.end_effector_link)

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

        # FCL collision checkers removed for streamlined PyBullet-only planning
        self._colliders = {}

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

        # World-aware FCL collision manager removed
        self._world = None

        # Register robots
        # no-op: world collision removed

        # Add static objects
        # no-op: static objects not added to FCL world
            # 'plane' is already handled by per-robot ground_plane_z
            # (future: add meshes here via trimesh->BVH if needed)

        # Allowed pairs that involve robots and/or objects, e.g. ("ur5e_1/wrist_3_link", "obj:table1")
        # no-op: FCL world disabled

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

        # 1. Remove from collision checker
        collider = self._colliders.get(robot_name)
        if collider:
            collider.remove_tool(tool_name)

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
                        traj_info = self._active_traj.pop(name, {})
                        # Set hold target to final pose (deg)
                        q_deg = [float(np.rad2deg(v)) for v in waypoints[-1]]
                        self._targets_deg[name] = q_deg
                        # Compute final EE pose (world) and log error vs requested
                        try:
                            ee = self._ee_link_name.get(name)
                            if ee:
                                link = robot_entity.get_link(ee)
                                p = to_numpy(link.get_pos()).astype(np.float32)
                                q = to_numpy(link.get_quat()).astype(np.float32)  # wxyz
                                # Convert to RPY degrees
                                w,x,y,z = q
                                t0 = 2*(w*x + y*z); t1 = 1 - 2*(x*x + y*y)
                                roll = np.arctan2(t0, t1)
                                t2 = np.clip(2*(w*y - z*x), -1.0, 1.0)
                                pitch = np.arcsin(t2)
                                t3 = 2*(w*z + x*y); t4 = 1 - 2*(y*y + z*z)
                                yaw = np.arctan2(t3, t4)
                                final_pose_deg = (
                                    float(p[0]), float(p[1]), float(p[2]),
                                    float(np.rad2deg(roll)), float(np.rad2deg(pitch)), float(np.rad2deg(yaw)),
                                )
                            else:
                                final_pose_deg = None
                        except Exception:
                            final_pose_deg = None

                        # Compute deltas if we have a stored target pose in world
                        try:
                            target_pose = traj_info.get("target_world_pose_deg")
                        except Exception:
                            target_pose = None

                        if final_pose_deg and target_pose:
                            def ang_diff_deg(a, b):
                                d = a - b
                                # wrap to [-180, 180]
                                while d > 180.0:
                                    d -= 360.0
                                while d < -180.0:
                                    d += 360.0
                                return d
                            dx = final_pose_deg[0] - target_pose[0]
                            dy = final_pose_deg[1] - target_pose[1]
                            dz = final_pose_deg[2] - target_pose[2]
                            dr = ang_diff_deg(final_pose_deg[3], target_pose[3])
                            dp = ang_diff_deg(final_pose_deg[4], target_pose[4])
                            dyaw = ang_diff_deg(final_pose_deg[5], target_pose[5])
                            # Also compute base-frame RPY error vs requested RPY
                            try:
                                # target pose request we used came from (x,y,z, roll,pitch,yaw) in base
                                # logged as target_pose (already transformed to world); reconstruct base frame for logging
                                robot_cfg = next((r for r in self.config.robots if getattr(r, 'name', None) == name), None)
                                if robot_cfg is not None:
                                    t_base, q_base = self._base_pose(robot_cfg)
                                    # Convert world quaternion (final) to base frame: q_b = q_base_conj * q_world
                                    q_world_wxyz = np.array([0.0,0.0,0.0,0.0], dtype=np.float32)
                                    # Rebuild from final_pose_deg roll/pitch/yaw for consistency
                                    q_world_wxyz = rpy_to_quat_wxyz(
                                        np.deg2rad(final_pose_deg[3]),
                                        np.deg2rad(final_pose_deg[4]),
                                        np.deg2rad(final_pose_deg[5])
                                    )
                                    q_base_conj = quat_wxyz_conj(np.array(q_base, dtype=np.float32))
                                    q_b = quat_wxyz_multiply(q_base_conj, q_world_wxyz)
                                    Rb = quat_wxyz_to_rotation_matrix(q_b)
                                    rpy_b = rotation_matrix_to_euler_rpy(Rb)
                                    final_base_rpy_deg = (
                                        float(np.rad2deg(rpy_b[0])),
                                        float(np.rad2deg(rpy_b[1])),
                                        float(np.rad2deg(rpy_b[2])),
                                    )
                                    # Requested RPY was in base frame and is known from the controller at planning time, but we don't
                                    # persist it here; approximate using target_pose transformed back to base frame
                                    q_world_tgt = rpy_to_quat_wxyz(
                                        np.deg2rad(target_pose[3]),
                                        np.deg2rad(target_pose[4]),
                                        np.deg2rad(target_pose[5])
                                    )
                                    q_b_tgt = quat_wxyz_multiply(q_base_conj, q_world_tgt)
                                    Rb_t = quat_wxyz_to_rotation_matrix(q_b_tgt)
                                    rpy_b_t = rotation_matrix_to_euler_rpy(Rb_t)
                                    req_base_rpy_deg = (
                                        float(np.rad2deg(rpy_b_t[0])),
                                        float(np.rad2deg(rpy_b_t[1])),
                                        float(np.rad2deg(rpy_b_t[2])),
                                    )
                                    dRb = (
                                        ang_diff_deg(final_base_rpy_deg[0], req_base_rpy_deg[0]),
                                        ang_diff_deg(final_base_rpy_deg[1], req_base_rpy_deg[1]),
                                        ang_diff_deg(final_base_rpy_deg[2], req_base_rpy_deg[2]),
                                    )
                                else:
                                    final_base_rpy_deg = None
                                    dRb = None
                            except Exception:
                                final_base_rpy_deg = None
                                dRb = None

                            if final_base_rpy_deg and dRb:
                                self.logger.info(
                                    "%s: Final EE pose (world) pos=(%.3f, %.3f, %.3f) rpy=(%.1f, %.1f, %.1f) | error dpos=(%.3f, %.3f, %.3f) m drpy_world=(%.1f, %.1f, %.1f) deg | base_rpy=(%.1f, %.1f, %.1f) d_base_rpy=(%.1f, %.1f, %.1f) deg",
                                    name,
                                    final_pose_deg[0], final_pose_deg[1], final_pose_deg[2],
                                    final_pose_deg[3], final_pose_deg[4], final_pose_deg[5],
                                    dx, dy, dz, dr, dp, dyaw,
                                    final_base_rpy_deg[0], final_base_rpy_deg[1], final_base_rpy_deg[2],
                                    dRb[0], dRb[1], dRb[2],
                                )
                            else:
                                self.logger.info(
                                    "%s: Final EE pose (world) pos=(%.3f, %.3f, %.3f) rpy=(%.1f, %.1f, %.1f) | error dpos=(%.3f, %.3f, %.3f) m drpy=(%.1f, %.1f, %.1f) deg",
                                    name,
                                    final_pose_deg[0], final_pose_deg[1], final_pose_deg[2],
                                    final_pose_deg[3], final_pose_deg[4], final_pose_deg[5],
                                    dx, dy, dz, dr, dp, dyaw,
                                )
                        else:
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

    def _base_pose(self, robot_cfg: RobotConfig):
        t = np.array(robot_cfg.base_position, dtype=np.float32)
        q = self._euler_deg_to_wxyz(*robot_cfg.base_orientation)
        return t, q

    def _apply_frame(self, robot_cfg: RobotConfig, target_pos_base, target_quat_base_wxyz, frame: str):
        if frame == 'base':
            t_base, q_base = self._base_pose(robot_cfg)
            pos_local = np.array(target_pos_base, dtype=np.float32)
            pw = t_base + quat_wxyz_rotate_vec(q_base, pos_local)
            qw = quat_wxyz_multiply(q_base, np.array(target_quat_base_wxyz, dtype=np.float32))
            return tuple(pw.tolist()), tuple(qw.tolist())
        return target_pos_base, target_quat_base_wxyz

    # Pose helpers
    def _get_ee_pose_w(self, entity, ee_name: str):
        link = entity.get_link(ee_name)
        pos = link.get_pos()
        quat = link.get_quat()  # wxyz
        pos = to_numpy(pos)
        quat = to_numpy(quat)
        return pos.astype(np.float32), quat.astype(np.float32)



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

        # Optional: enforce URDF joint limits if we can resolve them for this entity
        limits: list[tuple[float,float]] | None = None
        try:
            robot_name = None
            for name, ent in self._robots.items():
                if ent is entity:
                    robot_name = name
                    break
            if robot_name is not None:
                cfg = next((r for r in self.config.robots if getattr(r, 'name', None) == robot_name), None)
                if cfg is not None:
                    jl = self.get_robot_joint_limits(cfg)  # list of dicts in URDF order (non-fixed)
                    if jl and len(jl) == len(dofs_idx):
                        limits = [(float(j.get('lower', -np.inf)), float(j.get('upper', np.inf))) for j in jl]
        except Exception:
            limits = None

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
            q_err = quat_wxyz_multiply(target_q, quat_wxyz_conj(q_cur))
            e_rot = quat_wxyz_to_rotvec(q_err)
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
                dq = quat_wxyz_multiply(q_p, quat_wxyz_conj(q_cur))
                drot = quat_wxyz_to_rotvec(dq) / eps
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
            # Enforce joint limits if available
            if limits is not None and len(limits) == len(q):
                for i, (lo, hi) in enumerate(limits):
                    if np.isfinite(lo):
                        q[i] = max(q[i], lo)
                    if np.isfinite(hi):
                        q[i] = min(q[i], hi)
            try:
                entity.set_dofs_position(q, dofs_idx_local=dofs_idx)
            except Exception:
                entity.set_dofs_position(q, dofs_idx)

        # Final pose check
        p_cur, q_cur = self._get_ee_pose_w(entity, ee_name)
        e_pos = np.linalg.norm(target_p - p_cur)
        q_err = quat_wxyz_multiply(target_q, quat_wxyz_conj(q_cur))
        e_rot = np.linalg.norm(quat_wxyz_to_rotvec(q_err))

        # Restore original q0; we'll return solution separately
        try:
            entity.set_dofs_position(q0, dofs_idx_local=dofs_idx)
        except Exception:
            entity.set_dofs_position(q0, dofs_idx)

        # Final clamp before returning
        if limits is not None and len(limits) == len(q):
            for i, (lo, hi) in enumerate(limits):
                if np.isfinite(lo):
                    q[i] = max(q[i], lo)
                if np.isfinite(hi):
                    q[i] = min(q[i], hi)

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
            
            # Removed verbose robot config logging for cleaner output
            
            # Handle both dict and RobotConfig objects
            robot_cfg = None
            for r in self.config.robots:
                if isinstance(r, dict):
                    if r.get('name') == robot:
                        self.logger.warning(f"Robot config for {robot} is dict, converting to RobotConfig")
                        robot_cfg = RobotConfig.model_validate(r)
                        break
                elif hasattr(r, 'name') and r.name == robot:
                    robot_cfg = r
                    break
            
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

            started_at = time.perf_counter()
            
            # --- 1. IK Solution with ikpy ---
            from io import StringIO
            # Use the raw, unresolved URDF path to correctly locate relative mesh files.
            # Handle both dict and RobotConfig objects
            if isinstance(robot_cfg, dict):
                raw_urdf_path = robot_cfg.get('_raw_urdf_path') or robot_cfg.get('urdf')
                end_effector = robot_cfg.get('end_effector')
            else:
                raw_urdf_path = getattr(robot_cfg, '_raw_urdf_path', None) or robot_cfg.urdf
                end_effector = robot_cfg.end_effector
                
            robot_urdf_path = Path(raw_urdf_path)

            # If a tool is attached, merge its URDF with the robot's for accurate IK.
            if robot in self._attached_tool and end_effector:
                tool_urdf_path = Path(end_effector)
                # Create an in-memory, merged URDF to pass to ikpy
                merged_urdf_xml = merge_urdfs(
                    robot_urdf_path,
                    tool_urdf_path,
                    self._ee_link_name[robot]
                )
                ik_urdf = StringIO(merged_urdf_xml)
                # The end-effector for IK is now the tool's TCP.
                ee_name = self._tool_tcp_link.get(robot, ee_name)
            else:
                # Use the original URDF file directly.
                with open(robot_urdf_path, "r") as f:
                    ik_urdf = StringIO(f.read())

            # Get the names of actuated joints from the original URDF.
            actuated_joint_names = {j["name"] for j in parse_joint_limits(robot_urdf_path)}
            
            # Create a temporary chain from the (potentially merged) URDF in memory.
            # Resolve URDF root base link to avoid ikpy default 'base_link' mismatch
            base_link_name = get_urdf_root_link_name(robot_urdf_path) or 'base_link'
            full_chain = Chain.from_urdf_file(ik_urdf, base_elements=[base_link_name])
            
            # The link names in ikpy correspond to the joint names from the URDF.
            # A link is active if its name is in the set of actuated joint names.
            link_names = [link.name for link in full_chain.links]
            active_links_mask = [link_name in actuated_joint_names for link_name in link_names]
            # The first link is the "Base link" and is always inactive
            if active_links_mask:
                active_links_mask[0] = False

            # Create the final chain, this time with the correct mask.
            # Reset the StringIO object to be read again.
            ik_urdf.seek(0)
            chain = Chain.from_urdf_file(ik_urdf, active_links_mask=active_links_mask, base_elements=[base_link_name])

            try:
                target_rotation_matrix = quat_wxyz_to_rotation_matrix(np.array(target_quat_wxyz))
                # Prefer calling ikpy without an initial guess to avoid its strict bounds mapping;
                # we'll refine with numeric IK if needed.
                q_goal_full = chain.inverse_kinematics(
                    target_position=np.array(target_pos),
                    target_orientation=target_rotation_matrix
                )
                # Extract only the active joint values from the full solution.
                q_goal_np = np.compress(active_links_mask, q_goal_full, axis=0).astype(np.float32)
                used_solver = 'ikpy'
            except Exception as e:
                # Retry once with a zero initial guess if available
                try:
                    q0_full = np.zeros(len(full_chain.links), dtype=np.float32)
                    q_goal_full = chain.inverse_kinematics(
                        target_position=np.array(target_pos),
                        target_orientation=target_rotation_matrix,
                        initial_position=q0_full
                    )
                    q_goal_np = np.compress(active_links_mask, q_goal_full, axis=0).astype(np.float32)
                    used_solver = 'ikpy'
                except Exception as e2:
                    self.logger.warning(f"ikpy failed for {robot}: {e2}")
                    q_goal_np = None

            if q_goal_np is None:
                self.logger.warning("IK failed (robot=%s); rejecting Cartesian command", robot)
                self._emit("cartesian_failed", robot, "ik_failed")
                return False

            # Optional IK logging for debugging
            self.logger.debug("IK solver=%s; q_goal dtype=%s shape=%s", used_solver, getattr(q_goal_np, 'dtype', None), getattr(q_goal_np, 'shape', None))
            self.logger.debug("IK solver=%s, q_start=%s", used_solver, q_start)

            ctrl = self._ctrl_for(robot)
            use_fcl = bool(getattr(ctrl, 'use_fcl_postcheck', False))
            if getattr(ctrl, 'collision_check', True) and use_fcl:
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

            # Orientation validation using Genesis FK at q_goal; refine with numeric IK if needed
            try:
                if ee_name and q_goal_np is not None:
                    # Temporarily set to q_goal to evaluate FK
                    try:
                        entity.set_dofs_position(q_goal_np, dofs_idx_local=dofs_idx)
                    except Exception:
                        entity.set_dofs_position(q_goal_np, dofs_idx)
                    try:
                        self._scene.step()
                    except Exception:
                        pass
                    p_cur, q_cur = self._get_ee_pose_w(entity, ee_name)
                    R_tgt = quat_wxyz_to_rotation_matrix(np.array(target_quat_wxyz, dtype=np.float32))
                    R_cur = quat_wxyz_to_rotation_matrix(q_cur.astype(np.float32))
                    R_err = R_tgt @ R_cur.T
                    trace = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
                    angle = float(np.arccos(trace))
                    pos_err = float(np.linalg.norm(np.array(target_pos, dtype=np.float32) - p_cur))
                    rot_err = angle
                    # Restore to q_start for planning baseline
                    try:
                        entity.set_dofs_position(q_start, dofs_idx_local=dofs_idx)
                    except Exception:
                        entity.set_dofs_position(q_start, dofs_idx)
                    if rot_err > np.deg2rad(3.0) or pos_err > 5e-3:
                        q_refined = self._numeric_ik(
                            entity, ee_name, q0=q_goal_np, pos_w=target_pos, quat_w_wxyz=target_quat_wxyz,
                            iters=80, step_scale=1.0, damping=1e-2
                        )
                        if q_refined is not None:
                            q_goal_np = q_refined.astype(np.float32)
                            self.logger.debug("Refined IK via numeric DLS (pos_err=%.4f, rot_err=%.3f rad)", pos_err, rot_err)
            except Exception:
                # Non-fatal; keep ikpy solution
                try:
                    entity.set_dofs_position(q_start, dofs_idx_local=dofs_idx)
                except Exception:
                    entity.set_dofs_position(q_start, dofs_idx)

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

            # Start-state validity: reject if current state is in collision with world
            if getattr(ctrl, 'collision_check', True):
                try:
                    if self._world is not None:
                        ok0, reason0 = self._world.check_state(robot, q_start, dofs_idx)
                        if not ok0:
                            self.logger.warning("Start state in collision (%s); rejecting Cartesian command for %s", reason0, robot)
                            self._emit("cartesian_failed", robot, "start_in_collision")
                            return False
                    elif self._colliders.get(robot) is not None:
                        ok0, reason0 = self._colliders[robot].check_state(q_start, dofs_idx)
                        if not ok0:
                            self.logger.warning("Start state in self/ground collision (%s); rejecting", reason0)
                            self._emit("cartesian_failed", robot, "start_in_collision")
                            return False
                except Exception:
                    # Non-fatal: proceed
                    pass

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
                # Build ALL environment bodies as obstacles for goal validation and OMPL
                obstacles = []
                obstacles_meta = []  # (body_id, kind, name)
                created_body_ids = []  # track temporary bodies to remove after planning
                obj_added_count = 0
                robot_added_count = 0

                # Include ALL obstacles (objects + other robots) for planning to be conservative

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
                
                # 0. Add ground plane if configured
                try:
                    ctrl_local = self._ctrl_for(robot)
                    ground_z = float(getattr(ctrl_local, 'ground_plane_z', 0.0) or 0.0)
                    # Raise the planning ground slightly to build in a conservative buffer
                    ground_raise = float(getattr(ctrl_local, 'ground_clearance_m', 0.002) or 0.0)
                except Exception:
                    ground_z = 0.0
                    ground_raise = 0.0
                half_extents = [50.0, 50.0, 0.005]
                top_z = float(ground_z) + float(ground_raise)
                ground_pos = [0.0, 0.0, top_z - half_extents[2]]
                plane_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=planner.client)
                with _silence_stdio():
                    plane_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_shape, basePosition=ground_pos, baseOrientation=[0,0,0,1], physicsClientId=planner.client)
                obstacles.append(plane_id)
                obstacles_meta.append((plane_id, 'ground', 'ground'))
                created_body_ids.append(plane_id)
                # Exempt the ground from clearance checks (contacts still block)
                try:
                    planner.set_clearance_exempt_ids([plane_id])
                except Exception:
                    pass

                # 1. Add static objects (tables, boxes, etc.)
                for obj in self.config.objects:
                    if obj.type == "box" and obj.size is not None:
                        collision_shape = p.createCollisionShape(
                            p.GEOM_BOX,
                            halfExtents=[s/2.0 for s in obj.size],
                            physicsClientId=planner.client
                        )

                        # orientation_rpy is configured in DEGREES; PyBullet expects RADIANS
                        rpy_deg = getattr(obj, 'orientation_rpy', (0.0, 0.0, 0.0))
                        rpy_rad = tuple(np.deg2rad(r) for r in rpy_deg)
                        quat = p.getQuaternionFromEuler(rpy_rad, physicsClientId=planner.client)

                        with _silence_stdio():
                            body_id = p.createMultiBody(
                                baseMass=0,
                                baseCollisionShapeIndex=collision_shape,
                                basePosition=obj.position,
                                baseOrientation=quat,
                                physicsClientId=planner.client
                            )
                        obstacles.append(body_id)
                        obstacles_meta.append((body_id, 'object', getattr(obj, 'name', 'unnamed')))
                        created_body_ids.append(body_id)
                        obj_added_count += 1
                
                # 2. Add OTHER robots as obstacles (in their current joint states)
                for other_robot_name, other_entity in self._robots.items():
                    if other_robot_name == robot:
                        continue  # Skip the planning robot itself
                    
                    # Get current joint state of the other robot
                    try:
                        other_q = np.array([j.qpos for j in other_entity.get_joints()], dtype=np.float32)
                    except Exception:
                        other_q = np.array(self._targets_deg.get(other_robot_name, [0]*6), dtype=np.float32)
                        other_q = np.deg2rad(other_q)  # Convert to radians
                    
                    # Load the other robot's URDF into PyBullet as an obstacle
                    other_robot_cfg = None
                    for r in self.config.robots:
                        if (hasattr(r, 'name') and r.name == other_robot_name) or \
                           (isinstance(r, dict) and r.get('name') == other_robot_name):
                            other_robot_cfg = r
                            break
                    
                    if other_robot_cfg:
                        other_urdf = other_robot_cfg.urdf if hasattr(other_robot_cfg, 'urdf') else other_robot_cfg.get('urdf')
                        other_pos = other_robot_cfg.base_position if hasattr(other_robot_cfg, 'base_position') else other_robot_cfg.get('base_position', [0, 0, 0])
                        other_rpy_deg = other_robot_cfg.base_orientation if hasattr(other_robot_cfg, 'base_orientation') else other_robot_cfg.get('base_orientation', [0, 0, 0])
                        # Convert degrees -> radians for PyBullet
                        other_rpy = tuple(np.deg2rad(v) for v in other_rpy_deg)
                        other_fixed = other_robot_cfg.fixed_base if hasattr(other_robot_cfg, 'fixed_base') else other_robot_cfg.get('fixed_base', True)

                        # Load other robot into PyBullet
                        other_quat = p.getQuaternionFromEuler(other_rpy, physicsClientId=planner.client)
                        with _silence_stdio():
                            other_robot_id = p.loadURDF(
                                other_urdf,
                                basePosition=other_pos,
                                baseOrientation=other_quat,
                                useFixedBase=other_fixed,
                                physicsClientId=planner.client
                            )
                        
                        # Set the other robot to its current joint configuration
                        other_joint_count = p.getNumJoints(other_robot_id, physicsClientId=planner.client)
                        for j in range(min(len(other_q), other_joint_count)):
                            joint_info = p.getJointInfo(other_robot_id, j, physicsClientId=planner.client)
                            if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                                p.resetJointState(other_robot_id, j, other_q[j], physicsClientId=planner.client)
                        
                        obstacles.append(other_robot_id)
                        obstacles_meta.append((other_robot_id, 'robot', other_robot_name))
                        created_body_ids.append(other_robot_id)
                        robot_added_count += 1

                # Pre-check IK goal against world using PyBullet collision and log detailed reason
                try:
                    goal_ok = planner._is_collision_free(q_goal_np.tolist(), obstacles)
                except Exception:
                    goal_ok = True
                if not goal_ok:
                    # Build detailed contact pairs using PyBullet contacts at the goal
                    details = []
                    # 1) Check self-collision pairs first
                    try:
                        cps_self = p.getContactPoints(bodyA=planner.robot_id, bodyB=planner.robot_id, physicsClientId=planner.client)
                        for c in cps_self or []:
                            la = int(c[3]); lb = int(c[4])
                            if la == lb:
                                continue
                            try:
                                ja = p.getJointInfo(planner.robot_id, la, physicsClientId=planner.client)
                                ln_a = (ja[12].decode('utf-8') if isinstance(ja[12], (bytes, bytearray)) else str(ja[12]))
                            except Exception:
                                ln_a = f"link{la}"
                            try:
                                jb = p.getJointInfo(planner.robot_id, lb, physicsClientId=planner.client)
                                ln_b = (jb[12].decode('utf-8') if isinstance(jb[12], (bytes, bytearray)) else str(jb[12]))
                            except Exception:
                                ln_b = f"link{lb}"
                            details.append(f"self:{ln_a}~{ln_b}")
                            if len(details) >= 2:
                                break
                    except Exception:
                        pass
                    try:
                        for (bid, kind, name) in obstacles_meta:
                            cps = p.getContactPoints(bodyA=planner.robot_id, bodyB=bid, physicsClientId=planner.client)
                            if not cps:
                                continue
                            # Map first contact to link names if possible
                            ca = cps[0]
                            rl_idx = int(ca[3]); ol_idx = int(ca[4])
                            # Robot link name
                            if rl_idx >= 0:
                                try:
                                    ji = p.getJointInfo(planner.robot_id, rl_idx, physicsClientId=planner.client)
                                    rlink = (ji[12].decode('utf-8') if isinstance(ji[12], (bytes, bytearray)) else str(ji[12]))
                                except Exception:
                                    rlink = f"link{rl_idx}"
                            else:
                                rlink = "base"
                            # Other body link name (if robot)
                            if kind == 'robot' and ol_idx >= 0:
                                try:
                                    ji2 = p.getJointInfo(bid, ol_idx, physicsClientId=planner.client)
                                    olink = (ji2[12].decode('utf-8') if isinstance(ji2[12], (bytes, bytearray)) else str(ji2[12]))
                                except Exception:
                                    olink = f"link{ol_idx}"
                            elif kind == 'ground':
                                olink = 'plane'
                            else:
                                olink = name
                            details.append(f"{kind}:{name} ({rlink}~{olink})")
                            if len(details) >= 2:
                                break
                    except Exception:
                        pass
                    reason_detail = "; ".join(details) if details else "unknown"
                    self.logger.warning("IK goal pose is in collision (%s); rejecting Cartesian command for %s", reason_detail, robot)
                    self._emit("cartesian_failed", robot, f"ik_goal_in_collision:{reason_detail}")
                    return False

                self.logger.debug(f"OMPL planning with {len(obstacles)} obstacles ({obj_added_count} objects + {robot_added_count} robots)")
                path = planner.plan_path(q_start.tolist(), q_goal_np.tolist(), obstacles=obstacles, clearance_exempt_ids=[plane_id])

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

            if getattr(ctrl, 'collision_check', True) and use_fcl:
                # Adaptive world post-check to avoid false rejects due to time budget exhaustion
                def _verify_with(check_fn, label: str):
                    trials = []
                    W = max(2, int(waypoints.shape[0]))
                    waypoints_list = [waypoints[i] for i in range(W)]
                    strict = bool(getattr(ctrl, 'strict_cartesian', True))
                    sub = max(5, getattr(ctrl, 'ccd_substeps', 5))
                    # 1) strict stride, increased budget
                    trials.append(dict(substeps=sub, stride=1 if strict else max(1, W // 30), max_time_s=max(0.6, getattr(ctrl, 'postcheck_time_s', 0.2))))
                    # 2) strict stride, bigger budget
                    trials.append(dict(substeps=sub, stride=1 if strict else max(1, W // 30), max_time_s=1.0))
                    # 3) relaxed stride, same substeps
                    trials.append(dict(substeps=sub, stride=max(1, W // 30), max_time_s=1.0))
                    for targs in trials:
                        t0 = time.perf_counter()
                        ok, reason, seg = check_fn(waypoints_list, dofs_idx, **targs)
                        dt = time.perf_counter() - t0
                        self.logger.debug(f"{label} postcheck: ok={ok} reason={reason} stride={targs['stride']} sub={targs['substeps']} dt={dt:.3f}")
                        # Immediate reject on explicit collision/clearance failure
                        if not ok and reason and ('collision' in reason or 'clearance' in reason):
                            return False
                        # Accept only when fully ok
                        if ok and (not reason or reason == 'ok'):
                            return True
                    # If all trials exhausted without a clean OK, reject safely
                    return False

                if self._world is not None:
                    def world_check(wp, idx, substeps, stride, max_time_s):
                        return self._world.check_path(robot, wp, idx, substeps=substeps, stride=stride, max_time_s=max_time_s)
                    if not _verify_with(world_check, 'World'):
                        # Fallback: fast PyBullet re-validation of all waypoints against same obstacle set
                        try:
                            planner_obstacles = []
                            temp_ids = []
                            # Ground plane
                            try:
                                ctrl_local = self._ctrl_for(robot)
                                ground_z = float(getattr(ctrl_local, 'ground_plane_z', 0.0) or 0.0)
                                ground_raise = float(getattr(ctrl_local, 'ground_clearance_m', 0.002) or 0.0)
                            except Exception:
                                ground_z = 0.0
                                ground_raise = 0.0
                            half_extents = [50.0, 50.0, 0.005]
                            top_z = float(ground_z) + float(ground_raise)
                            ground_pos = [0.0, 0.0, top_z - half_extents[2]]
                            plane_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=planner.client)
                            with _silence_stdio():
                                plane_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_shape, basePosition=ground_pos, baseOrientation=[0,0,0,1], physicsClientId=planner.client)
                            planner_obstacles.append(plane_id); temp_ids.append(plane_id)
                            # Rebuild static objects
                            for obj in self.config.objects:
                                if obj.type == "box" and obj.size is not None:
                                    cshape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2.0 for s in obj.size], physicsClientId=planner.client)
                                    rpy_deg = getattr(obj, 'orientation_rpy', (0.0, 0.0, 0.0))
                                    rpy_rad = tuple(np.deg2rad(r) for r in rpy_deg)
                                    quat = p.getQuaternionFromEuler(rpy_rad, physicsClientId=planner.client)
                                    with _silence_stdio():
                                        bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cshape, basePosition=obj.position, baseOrientation=quat, physicsClientId=planner.client)
                                    planner_obstacles.append(bid); temp_ids.append(bid)
                            # Rebuild other robots
                            for other_name, other_entity in self._robots.items():
                                if other_name == robot:
                                    continue
                                # joint state
                                try:
                                    other_q = np.array([j.qpos for j in other_entity.get_joints()], dtype=np.float32)
                                except Exception:
                                    other_q = np.array(self._targets_deg.get(other_name, []), dtype=np.float32)
                                    if other_q.size:
                                        other_q = np.deg2rad(other_q)
                                # config
                                cfg = next((r for r in self.config.robots if getattr(r, 'name', None) == other_name), None)
                                if cfg is None:
                                    continue
                                other_quat = p.getQuaternionFromEuler(tuple(np.deg2rad(v) for v in cfg.base_orientation), physicsClientId=planner.client)
                                with _silence_stdio():
                                    oid = p.loadURDF(cfg.urdf, basePosition=cfg.base_position, baseOrientation=other_quat, useFixedBase=cfg.fixed_base, physicsClientId=planner.client)
                                nj = p.getNumJoints(oid, physicsClientId=planner.client)
                                for j in range(min(nj, len(other_q))):
                                    ji = p.getJointInfo(oid, j, physicsClientId=planner.client)
                                    if ji[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                                        p.resetJointState(oid, j, other_q[j], physicsClientId=planner.client)
                                planner_obstacles.append(oid); temp_ids.append(oid)

                            # Ensure ground is exempt from clearance checks in validator
                            try:
                                planner.set_clearance_exempt_ids([plane_id])
                            except Exception:
                                pass
                            # Validate each waypoint
                            passed = True
                            for k in range(int(waypoints.shape[0])):
                                if not planner._is_collision_free(waypoints[k].tolist(), planner_obstacles):
                                    passed = False
                                    break
                            # Cleanup
                            for bid in temp_ids:
                                try:
                                    p.removeBody(bid, physicsClientId=planner.client)
                                except Exception:
                                    pass
                            if not passed:
                                self.logger.warning("Planned path rejected by PyBullet postcheck")
                                self._emit("cartesian_failed", robot, "path_in_collision")
                                return False
                            else:
                                self.logger.debug("PyBullet postcheck accepted path after world postcheck exhaustion")
                        except Exception as _:
                            self.logger.warning("World postcheck failed; fallback PyBullet postcheck unavailable; rejecting")
                            self._emit("cartesian_failed", robot, "path_in_collision")
                            return False
                elif self._colliders.get(robot) is not None:
                    coll = self._colliders[robot]
                    def robo_check(wp, idx, substeps, stride, max_time_s):
                        return coll.check_path(wp, idx, substeps=substeps, stride=stride, max_time_s=max_time_s)
                    if not _verify_with(robo_check, 'Robot'):
                        # As above, fallback to PyBullet validation
                        try:
                            planner_obstacles = []
                            temp_ids = []
                            for obj in self.config.objects:
                                if obj.type == "box" and obj.size is not None:
                                    cshape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2.0 for s in obj.size], physicsClientId=planner.client)
                                    rpy_deg = getattr(obj, 'orientation_rpy', (0.0, 0.0, 0.0))
                                    rpy_rad = tuple(np.deg2rad(r) for r in rpy_deg)
                                    quat = p.getQuaternionFromEuler(rpy_rad, physicsClientId=planner.client)
                                    with _silence_stdio():
                                        bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cshape, basePosition=obj.position, baseOrientation=quat, physicsClientId=planner.client)
                                    planner_obstacles.append(bid); temp_ids.append(bid)
                            for other_name, other_entity in self._robots.items():
                                if other_name == robot:
                                    continue
                                try:
                                    other_q = np.array([j.qpos for j in other_entity.get_joints()], dtype=np.float32)
                                except Exception:
                                    other_q = np.array(self._targets_deg.get(other_name, []), dtype=np.float32)
                                    if other_q.size:
                                        other_q = np.deg2rad(other_q)
                                cfg = next((r for r in self.config.robots if getattr(r, 'name', None) == other_name), None)
                                if cfg is None:
                                    continue
                                other_quat = p.getQuaternionFromEuler(tuple(np.deg2rad(v) for v in cfg.base_orientation), physicsClientId=planner.client)
                                with _silence_stdio():
                                    oid = p.loadURDF(cfg.urdf, basePosition=cfg.base_position, baseOrientation=other_quat, useFixedBase=cfg.fixed_base, physicsClientId=planner.client)
                                nj = p.getNumJoints(oid, physicsClientId=planner.client)
                                for j in range(min(nj, len(other_q))):
                                    ji = p.getJointInfo(oid, j, physicsClientId=planner.client)
                                    if ji[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                                        p.resetJointState(oid, j, other_q[j], physicsClientId=planner.client)
                                planner_obstacles.append(oid); temp_ids.append(oid)
                            passed = True
                            for k in range(int(waypoints.shape[0])):
                                if not planner._is_collision_free(waypoints[k].tolist(), planner_obstacles):
                                    passed = False
                                    break
                            for bid in temp_ids:
                                try:
                                    p.removeBody(bid, physicsClientId=planner.client)
                                except Exception:
                                    pass
                            if not passed:
                                self.logger.warning("Planned path rejected by PyBullet postcheck")
                                self._emit("cartesian_failed", robot, "path_in_collision")
                                return False
                            else:
                                self.logger.debug("PyBullet postcheck accepted path after robot postcheck exhaustion")
                        except Exception:
                            self.logger.warning("Robot postcheck failed; fallback PyBullet postcheck unavailable; rejecting")
                            self._emit("cartesian_failed", robot, "path_in_collision")
                            return False

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
            # If FCL is disabled, run PyBullet waypoint validation up-front
            if getattr(ctrl, 'collision_check', True) and not use_fcl:
                try:
                    planner_obstacles = []
                    temp_ids = []
                    # Ground plane
                    try:
                        ctrl_local = self._ctrl_for(robot)
                        ground_z = float(getattr(ctrl_local, 'ground_plane_z', 0.0) or 0.0)
                        ground_raise = float(getattr(ctrl_local, 'ground_clearance_m', 0.002) or 0.0)
                    except Exception:
                        ground_z = 0.0
                        ground_raise = 0.0
                    half_extents = [50.0, 50.0, 0.005]
                    top_z = float(ground_z) + float(ground_raise)
                    ground_pos = [0.0, 0.0, top_z - half_extents[2]]
                    plane_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=planner.client)
                    with _silence_stdio():
                        plane_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_shape, basePosition=ground_pos, baseOrientation=[0,0,0,1], physicsClientId=planner.client)
                    planner_obstacles.append(plane_id); temp_ids.append(plane_id)
                    # Objects
                    for obj in self.config.objects:
                        if obj.type == "box" and obj.size is not None:
                            cshape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2.0 for s in obj.size], physicsClientId=planner.client)
                            rpy_deg = getattr(obj, 'orientation_rpy', (0.0, 0.0, 0.0))
                            rpy_rad = tuple(np.deg2rad(r) for r in rpy_deg)
                            quat = p.getQuaternionFromEuler(rpy_rad, physicsClientId=planner.client)
                            with _silence_stdio():
                                bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cshape, basePosition=obj.position, baseOrientation=quat, physicsClientId=planner.client)
                            planner_obstacles.append(bid); temp_ids.append(bid)
                    # Other robots
                    for other_name, other_entity in self._robots.items():
                        if other_name == robot:
                            continue
                        try:
                            other_q = np.array([j.qpos for j in other_entity.get_joints()], dtype=np.float32)
                        except Exception:
                            other_q = np.array(self._targets_deg.get(other_name, []), dtype=np.float32)
                            if other_q.size:
                                other_q = np.deg2rad(other_q)
                        cfg = next((r for r in self.config.robots if getattr(r, 'name', None) == other_name), None)
                        if cfg is None:
                            continue
                        other_quat = p.getQuaternionFromEuler(tuple(np.deg2rad(v) for v in cfg.base_orientation), physicsClientId=planner.client)
                        with _silence_stdio():
                            oid = p.loadURDF(cfg.urdf, basePosition=cfg.base_position, baseOrientation=other_quat, useFixedBase=cfg.fixed_base, physicsClientId=planner.client)
                        nj = p.getNumJoints(oid, physicsClientId=planner.client)
                        for j in range(min(nj, len(other_q))):
                            ji = p.getJointInfo(oid, j, physicsClientId=planner.client)
                            if ji[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                                p.resetJointState(oid, j, other_q[j], physicsClientId=planner.client)
                        planner_obstacles.append(oid); temp_ids.append(oid)

                    # Validate all waypoints with PyBullet (contacts + min_clearance)
                    try:
                        planner.set_clearance_exempt_ids([plane_id])
                    except Exception:
                        pass
                    passed = True
                    for k in range(int(waypoints.shape[0])):
                        if not planner._is_collision_free(waypoints[k].tolist(), planner_obstacles):
                            passed = False
                            break
                finally:
                    for bid in locals().get('temp_ids', []):
                        try:
                            p.removeBody(bid, physicsClientId=planner.client)
                        except Exception:
                            pass
                if not passed:
                    self.logger.warning("Planned path rejected by PyBullet postcheck (no FCL)")
                    self._emit("cartesian_failed", robot, "path_in_collision")
                    return False
