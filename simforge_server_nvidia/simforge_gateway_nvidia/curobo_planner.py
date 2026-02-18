"""
cuRobo GPU motion planner — initialisation, planning, trajectory conversion.

Wraps NVIDIA cuRobo's MotionGen for per-robot GPU-accelerated:
  • Inverse kinematics
  • Motion planning (trajectory optimisation on CUDA)
  • Collision checking (signed distance fields on CUDA)
"""

import asyncio
import math
import tempfile
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from .config import (
    ROBOT_CONFIG, CUROBO_AVAILABLE, CUROBO_IMPORT_ERROR, RobotStateInfo,
    ENV_WORLD_COLLISION_CONFIG,
)
from .collision_matrix import compute_self_collision_ignore

if CUROBO_AVAILABLE:
    import torch
    from curobo.types.math import Pose as CuPose
    from curobo.types.robot import JointState as CuJointState
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.motion_gen import (
        MotionGen, MotionGenConfig, MotionGenPlanConfig,
    )
    from curobo.util.trajectory import InterpolateType
    from curobo.util_file import load_yaml
import numpy as np


# ── Trajectory wrapper ───────────────────────────────────────────────

class TrajectoryWrapper:
    """Minimal container that mimics RobotTrajectory."""
    def __init__(self):
        self.joint_trajectory = JointTrajectory()


class CuroboPlanner:
    """Per-robot cuRobo GPU motion planner."""

    def __init__(self, node, *, interpolation_dt: float = 0.02,
                 max_velocity_scaling: float = 0.5,
                 max_acceleration_scaling: float = 0.5,
                 config_dir=None):
        self._node = node
        self._log = node.get_logger()
        self.interpolation_dt = interpolation_dt
        self.max_velocity_scaling = max_velocity_scaling
        self.max_acceleration_scaling = max_acceleration_scaling
        self.config_dir = config_dir
        self._motion_gens: Dict[str, Any] = {}
        self._curobo_ready = False

    @property
    def motion_gens(self):
        return self._motion_gens

    @property
    def is_ready(self):
        return self._curobo_ready

    def has_robot(self, robot_name: str) -> bool:
        """Return True if cuRobo is initialised for this robot."""
        return robot_name in self._motion_gens

    # ── Initialisation ───────────────────────────────────────────

    def init(self, urdf_string: str, tf_buffer=None):
        """Initialise cuRobo MotionGen for each robot that has a config."""
        if not CUROBO_AVAILABLE:
            self._log.error(
                f"cuRobo not available: {CUROBO_IMPORT_ERROR}. "
                f"Motion planning will be DISABLED."
            )
            return

        if urdf_string is None:
            self._log.error("No URDF provided — cuRobo DISABLED.")
            return

        self._urdf_temp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".urdf", delete=False, prefix="curobo_"
        )
        self._urdf_temp.write(urdf_string)
        self._urdf_temp.flush()
        urdf_path = self._urdf_temp.name
        self._log.info(f"URDF written to temp file: {urdf_path}")

        tensor_args = TensorDeviceType()

        # World collision config — use env-specific config name
        world_cfg_dict = {"cuboid": {}}
        if self.config_dir:
            wc = self.config_dir / ENV_WORLD_COLLISION_CONFIG
            if wc.exists():
                world_cfg_dict = load_yaml(str(wc))
                self._log.info(f"Loaded world collision config: {wc}")
            else:
                self._log.warn(
                    f"World collision config not found: {wc}, "
                    f"using empty world"
                )

        for robot_name, cfg in ROBOT_CONFIG.items():
            curobo_config_name = cfg.get("curobo_config")
            if curobo_config_name is None:
                continue
            config_path = self.config_dir / curobo_config_name
            if not config_path.exists():
                self._log.error(f"cuRobo config not found: {config_path}")
                continue
            try:
                self._log.info(
                    f"Initialising cuRobo MotionGen for {robot_name}..."
                )
                robot_cfg_dict = load_yaml(str(config_path))
                if "robot_cfg" in robot_cfg_dict:
                    robot_cfg_dict["robot_cfg"]["kinematics"]["urdf_path"] = urdf_path
                else:
                    robot_cfg_dict["kinematics"]["urdf_path"] = urdf_path

                # ── Compute self_collision_ignore from sphere geometry ──
                # Start with any hardcoded pairs from the YAML config
                # (e.g. wrist_3 ↔ tool_base for deliberate overlaps),
                # then merge in pairs discovered by random sampling.
                kin_section = robot_cfg_dict
                if "robot_cfg" in robot_cfg_dict:
                    kin_section = robot_cfg_dict["robot_cfg"]
                if "kinematics" in kin_section:
                    kin_section = kin_section["kinematics"]

                # Preserve hardcoded ignore pairs from config
                hardcoded_ignore = dict(
                    kin_section.get("self_collision_ignore", {}) or {}
                )

                self._log.info(
                    f"Computing self-collision matrix for {robot_name} "
                    f"(sampling 5000 random configs)..."
                )
                computed_ignore = compute_self_collision_ignore(
                    robot_cfg_dict,
                    num_samples=5000,
                    collision_threshold=0.0,
                    ros_logger=self._log,
                )

                # Merge: hardcoded pairs take priority, computed pairs
                # are added on top.  This ensures deliberately-ignored
                # pairs (like tool mount overlaps) are always included
                # even if sampling doesn't detect them.
                merged_ignore = dict(hardcoded_ignore)
                for link, others in computed_ignore.items():
                    if link in merged_ignore:
                        existing = set(merged_ignore[link])
                        existing.update(others)
                        merged_ignore[link] = sorted(existing)
                    else:
                        merged_ignore[link] = others
                kin_section["self_collision_ignore"] = merged_ignore
                self._log.info(
                    f"Self-collision matrix for {robot_name}: "
                    f"{sum(len(v) for v in merged_ignore.values())} "
                    f"total ignored pairs "
                    f"({sum(len(v) for v in hardcoded_ignore.values())} "
                    f"hardcoded + computed)"
                )

                robot_world_cfg = _transform_world_to_robot_frame(
                    world_cfg_dict, robot_name
                )

                mg_config = MotionGenConfig.load_from_robot_config(
                    robot_cfg_dict,
                    robot_world_cfg,
                    tensor_args=tensor_args,
                    interpolation_dt=self.interpolation_dt,
                    trajopt_tsteps=32,
                    collision_checker_type="PRIMITIVE",
                    use_cuda_graph=True,
                    num_trajopt_seeds=4,
                    num_graph_seeds=4,
                    num_ik_seeds=32,
                    collision_activation_distance=0.025,
                    maximum_trajectory_dt=None,
                    interpolation_type=InterpolateType.CUBIC,
                )

                mg = MotionGen(mg_config)
                self._log.info(
                    f"Warming up cuRobo for {robot_name} "
                    f"(first CUDA compile)..."
                )
                mg.warmup(
                    enable_graph=True,
                    warmup_js_trajopt=True,
                    parallel_finetune=True,
                )
                self._motion_gens[robot_name] = mg
                self._log.info(f"cuRobo MotionGen ready for {robot_name} ✓")

            except Exception as e:
                self._log.error(
                    f"Failed to init cuRobo for {robot_name}: {e}\n"
                    f"{traceback.format_exc()}"
                )

        self._curobo_ready = len(self._motion_gens) > 0
        self._log.info(
            f"cuRobo initialisation complete: "
            f"{len(self._motion_gens)} robot(s) ready"
        )

    # ── Forward Kinematics ────────────────────────────────────────

    def compute_fk(
        self, robot_name: str, joint_positions: List[float],
    ) -> Optional[tuple]:
        """Compute forward kinematics for the given joint positions.

        Returns ``(position, quaternion)`` where:
          - position: [x, y, z] in metres (robot base frame)
          - quaternion: [qw, qx, qy, qz]
        Returns ``None`` if cuRobo is not initialised for this robot.
        """
        mg = self._motion_gens.get(robot_name)
        if mg is None:
            return None

        try:
            cfg = ROBOT_CONFIG[robot_name]
            js = CuJointState.from_position(
                torch.tensor([joint_positions], dtype=torch.float32).cuda(),
                joint_names=cfg["joints"],
            )
            kin_state = mg.compute_kinematics(js)
            # kin_state.ee_position: shape (1, 3)
            # kin_state.ee_quaternion: shape (1, 4) — [qw, qx, qy, qz]
            pos = kin_state.ee_position[0].cpu().tolist()
            quat = kin_state.ee_quaternion[0].cpu().tolist()
            return pos, quat
        except Exception as e:
            self._log.error(
                f"FK computation failed for {robot_name}: {e}"
            )
            return None

    # ── Planning ─────────────────────────────────────────────────

    def _plan_to_pose_sync(
        self,
        robot_name: str,
        target_position: List[float],
        target_orientation: List[float],
        current_joints: List[float],
        velocity_scaling: Optional[float] = None,
    ) -> Optional[Any]:
        """Synchronous collision-free planning (blocks on CUDA).

        ``target_orientation`` is [qx, qy, qz, qw] (client convention).
        """
        mg = self._motion_gens.get(robot_name)
        if mg is None:
            self._log.error(f"No cuRobo MotionGen for {robot_name}")
            return None

        cfg = ROBOT_CONFIG[robot_name]
        current = current_joints or list(cfg["home_position"])
        try:
            start = CuJointState.from_position(
                torch.tensor([current], dtype=torch.float32).cuda(),
                joint_names=cfg["joints"],
            )
            qx, qy, qz, qw = target_orientation
            goal = CuPose.from_list([
                target_position[0], target_position[1], target_position[2],
                qw, qx, qy, qz,
            ])
            plan_cfg = MotionGenPlanConfig(
                max_attempts=4,
                timeout=5.0,
                enable_graph=True, enable_opt=True,
                enable_finetune_trajopt=True,
                partial_ik_opt=False, parallel_finetune=True,
                time_dilation_factor=min(
                    velocity_scaling or self.max_velocity_scaling, 0.99
                ),
            )
            result = mg.plan_single(start, goal, plan_cfg)

            if result.success.item():
                traj = result.get_interpolated_plan()
                n = traj.position.shape[1]
                dur = n * result.interpolation_dt
                self._log.info(
                    f"cuRobo plan OK for {robot_name}: {n} waypoints, "
                    f"duration={dur:.2f}s, dt={result.interpolation_dt:.4f}s"
                )
                return result
            status = getattr(result, 'status', 'unknown')
            self._log.warn(
                f"cuRobo planning failed for {robot_name}: {status} "
                f"(pos={target_position}, "
                f"orient=[{qx:.3f},{qy:.3f},{qz:.3f},{qw:.3f}])"
            )
            return None
        except Exception as e:
            self._log.error(
                f"cuRobo planning error for {robot_name}: {e}\n"
                f"{traceback.format_exc()}"
            )
            return None

    async def plan_to_pose(
        self,
        robot_name: str,
        target_position: List[float],
        target_orientation: List[float],
        current_joints: List[float],
        velocity_scaling: Optional[float] = None,
    ) -> Optional[Any]:
        """Plan a collision-free trajectory to a Cartesian pose.

        Runs the blocking CUDA computation in a background thread so
        the asyncio event loop stays responsive during planning.

        ``target_orientation`` is [qx, qy, qz, qw] (client convention).
        """
        return await asyncio.to_thread(
            self._plan_to_pose_sync,
            robot_name, target_position, target_orientation,
            current_joints, velocity_scaling,
        )

    def _plan_to_joints_sync(
        self,
        robot_name: str,
        target_joints: List[float],
        current_joints: List[float],
        velocity_scaling: Optional[float] = None,
    ) -> Optional[Any]:
        """Synchronous joint planning (blocks on CUDA)."""
        mg = self._motion_gens.get(robot_name)
        if mg is None:
            self._log.error(f"No cuRobo MotionGen for {robot_name}")
            return None

        cfg = ROBOT_CONFIG[robot_name]
        current = current_joints or list(cfg["home_position"])
        try:
            start = CuJointState.from_position(
                torch.tensor([current], dtype=torch.float32).cuda(),
                joint_names=cfg["joints"],
            )
            goal = CuJointState.from_position(
                torch.tensor([target_joints], dtype=torch.float32).cuda(),
                joint_names=cfg["joints"],
            )
            plan_cfg = MotionGenPlanConfig(
                max_attempts=4, timeout=5.0,
                enable_graph=True, enable_opt=True,
                enable_finetune_trajopt=True,
                time_dilation_factor=min(
                    velocity_scaling or self.max_velocity_scaling, 0.99
                ),
            )
            result = mg.plan_single_js(start, goal, plan_cfg)

            if result.success.item():
                traj = result.get_interpolated_plan()
                self._log.info(
                    f"cuRobo joint plan OK for {robot_name}: "
                    f"{traj.position.shape[1]} waypoints"
                )
                return result
            self._log.warn(
                f"cuRobo joint planning failed for {robot_name}: "
                f"{getattr(result, 'status', 'unknown')}"
            )
            return None
        except Exception as e:
            self._log.error(
                f"cuRobo joint planning error: {e}\n"
                f"{traceback.format_exc()}"
            )
            return None

    async def plan_to_joints(
        self,
        robot_name: str,
        target_joints: List[float],
        current_joints: List[float],
        velocity_scaling: Optional[float] = None,
    ) -> Optional[Any]:
        """Plan to target joint positions.

        Runs the blocking CUDA computation in a background thread.
        """
        return await asyncio.to_thread(
            self._plan_to_joints_sync,
            robot_name, target_joints, current_joints,
            velocity_scaling,
        )

    # ── Trajectory conversion ────────────────────────────────────

    def result_to_ros_trajectory(
        self, result: Any, robot_name: str,
    ) -> Optional[TrajectoryWrapper]:
        """Convert a MotionGenResult to a ROS JointTrajectory."""
        cfg = ROBOT_CONFIG[robot_name]
        traj = result.get_interpolated_plan()

        traj_pos = traj.position.cpu().numpy()
        traj_vel = (traj.velocity.cpu().numpy()
                    if traj.velocity is not None else None)
        traj_acc = (traj.acceleration.cpu().numpy()
                    if traj.acceleration is not None else None)

        if traj_pos.ndim == 3:
            positions = traj_pos[0]
            velocities = traj_vel[0] if traj_vel is not None else None
            accelerations = traj_acc[0] if traj_acc is not None else None
        else:
            positions = traj_pos
            velocities = traj_vel
            accelerations = traj_acc

        dt = result.interpolation_dt

        wrapper = TrajectoryWrapper()
        wrapper.joint_trajectory.joint_names = list(cfg["joints"])

        n_pts = positions.shape[0]
        for i in range(n_pts):
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in positions[i]]
            pt.velocities = ([float(v) for v in velocities[i]]
                             if velocities is not None else [0.0] * 6)
            pt.accelerations = ([float(v) for v in accelerations[i]]
                                if accelerations is not None else [0.0] * 6)
            t = i * dt
            pt.time_from_start = Duration(
                sec=int(t), nanosec=int((t - int(t)) * 1e9)
            )
            wrapper.joint_trajectory.points.append(pt)

        # Zero velocity/acceleration at endpoints
        if wrapper.joint_trajectory.points:
            for pt in (wrapper.joint_trajectory.points[0],
                       wrapper.joint_trajectory.points[-1]):
                pt.velocities = [0.0] * 6
                pt.accelerations = [0.0] * 6

        return wrapper

    # ── Multi-waypoint planning ──────────────────────────────────

    async def plan_multi_waypoint(
        self,
        robot_name: str,
        poses: list,
        current_joints: List[float],
        velocity_scaling: Optional[float] = None,
        idle_time: float = 0.0,
        progress_callback: Optional[Callable] = None,
    ) -> Optional[tuple]:
        """Plan trajectories through multiple poses, concatenated with
        optional dwell periods.

        Parameters
        ----------
        progress_callback : Optional[Callable[[int, int, str], Awaitable[None]]]
            ``async callback(segment_index, total_segments, pose_name)``
            called after each segment is planned so callers (e.g. the
            protocol executor) can relay progress to the client.

        Returns ``(trajectory_wrapper, pose_times, valid_indices)`` or
        ``None``.
        """
        cfg = ROBOT_CONFIG[robot_name]
        vel_scale = velocity_scaling or self.max_velocity_scaling
        current = list(current_joints) if current_joints else list(
            cfg["home_position"]
        )
        original_current = list(current)

        all_points: List[JointTrajectoryPoint] = []
        pose_times: List[tuple] = []
        valid_indices: List[int] = []
        failed_names: List[str] = []
        cumulative_time = 0.0

        # Temporarily patch robot state for per-segment planning
        robot_states_ref = getattr(self, '_robot_states_ref', None)

        for i, pose_data in enumerate(poses):
            pose_name = pose_data.get("name", f"pose_{i}")
            position = pose_data.get("position", [0, 0, 0])
            orientation = pose_data.get("orientation", [0, 0, 0, 1])

            self._log.info(
                f"Planning segment {len(valid_indices)+1} "
                f"to pose {i} ({pose_name})"
            )

            result = await self.plan_to_pose(
                robot_name, position, orientation,
                current_joints=list(current),
                velocity_scaling=vel_scale,
            )

            # Notify caller about planning progress (keeps client alive)
            if progress_callback is not None:
                try:
                    await progress_callback(i, len(poses), pose_name)
                except Exception:
                    pass  # best-effort; don't abort planning

            if result is None:
                failed_names.append(pose_name)
                self._log.warn(
                    f"cuRobo planning failed for {pose_name} — skipping"
                )
                continue

            traj = result.get_interpolated_plan()
            self._log.info(
                f"Trajectory shapes: position={traj.position.shape}, "
                f"velocity={traj.velocity.shape if traj.velocity is not None else None}"
            )
            traj_pos = traj.position.cpu().numpy()
            traj_vel = (traj.velocity.cpu().numpy()
                        if traj.velocity is not None else None)
            traj_acc = (traj.acceleration.cpu().numpy()
                        if traj.acceleration is not None else None)

            if traj_pos.ndim == 3:
                positions = traj_pos[0]
                velocities = traj_vel[0] if traj_vel is not None else None
                accelerations = traj_acc[0] if traj_acc is not None else None
            else:
                positions = traj_pos
                velocities = traj_vel
                accelerations = traj_acc

            dt = result.interpolation_dt
            n_pts = positions.shape[0]
            start_idx = 1 if all_points and n_pts > 1 else 0

            for k in range(start_idx, n_pts):
                pt = JointTrajectoryPoint()
                pt.positions = [float(v) for v in positions[k]]
                pt.velocities = ([float(v) for v in velocities[k]]
                                 if velocities is not None else [0.0] * 6)
                pt.accelerations = ([float(v) for v in accelerations[k]]
                                    if accelerations is not None else [0.0] * 6)
                t = cumulative_time + (k - start_idx + 1) * dt
                pt.time_from_start = Duration(
                    sec=int(t), nanosec=int((t - int(t)) * 1e9)
                )
                all_points.append(pt)

            seg_duration = (n_pts - start_idx + 1) * dt
            cumulative_time += seg_duration
            pose_times.append((i, cumulative_time))
            valid_indices.append(i)

            # Dwell
            if idle_time > 0 and i < len(poses) - 1:
                dwell = JointTrajectoryPoint()
                dwell.positions = [float(v) for v in positions[-1]]
                dwell.velocities = [0.0] * 6
                dwell.accelerations = [0.0] * 6
                cumulative_time += idle_time
                t = cumulative_time
                dwell.time_from_start = Duration(
                    sec=int(t), nanosec=int((t - int(t)) * 1e9)
                )
                all_points.append(dwell)

            current = [float(v) for v in positions[-1]]
            self._log.info(
                f"Segment to {pose_name}: {n_pts} pts, "
                f"seg_dur={seg_duration:.2f}s"
            )

        if not all_points:
            self._log.error("All pose plans failed")
            return None

        if failed_names:
            self._log.warn(
                f"Planning failed for {len(failed_names)} poses: "
                f"{failed_names[:5]}"
            )

        all_points[0].velocities = [0.0] * 6
        all_points[0].accelerations = [0.0] * 6
        all_points[-1].velocities = [0.0] * 6
        all_points[-1].accelerations = [0.0] * 6

        wrapper = TrajectoryWrapper()
        wrapper.joint_trajectory.joint_names = list(cfg["joints"])
        wrapper.joint_trajectory.points = all_points

        total_dur = cumulative_time
        self._log.info(
            f"Multi-waypoint trajectory: {len(all_points)} points, "
            f"duration={total_dur:.2f}s, segments={len(valid_indices)}, "
            f"dwells={'yes' if idle_time > 0 else 'no'}"
        )
        return wrapper, pose_times, valid_indices


# ── World-frame transform helper ─────────────────────────────────


def _transform_world_to_robot_frame(
    world_cfg_dict: dict, robot_name: str
) -> dict:
    """Transform world collision objects from world frame to robot
    base_link frame (cuRobo operates in base_link)."""
    if robot_name == "nakul_ur5e":
        mount_pos = np.array([-0.6758, 0.0, 1.03])
        mount_yaw = -math.pi / 2
    elif robot_name == "sahadev_ur5e":
        mount_pos = np.array([0.6758, 0.0, 1.03])
        mount_yaw = math.pi / 2
    else:
        return world_cfg_dict

    cos_y = math.cos(mount_yaw)
    sin_y = math.sin(mount_yaw)

    def world_to_base(pos):
        rel = np.array(pos) - mount_pos
        return [
            float(cos_y * rel[0] + sin_y * rel[1]),
            float(-sin_y * rel[0] + cos_y * rel[1]),
            float(rel[2]),
        ]

    def rotate_quat(qw, qx, qy, qz):
        inv_qw = math.cos(-mount_yaw / 2)
        inv_qz = math.sin(-mount_yaw / 2)
        return [
            inv_qw * qw - inv_qz * qz,
            inv_qw * qx - inv_qz * qy,
            inv_qw * qy + inv_qz * qx,
            inv_qw * qz + inv_qz * qw,
        ]

    result = {}
    for obj_type in ("cuboid", "mesh"):
        if obj_type not in world_cfg_dict:
            continue
        result[obj_type] = {}
        for name, obj in world_cfg_dict[obj_type].items():
            new_obj = dict(obj)
            if "pose" in obj:
                p = obj["pose"]
                new_obj["pose"] = world_to_base(p[:3]) + rotate_quat(
                    p[3], p[4], p[5], p[6]
                )
            result[obj_type][name] = new_obj
    return result
