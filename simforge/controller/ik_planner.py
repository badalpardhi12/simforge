"""Inverse kinematics and motion planning helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np

from ..ik_drake import DrakeIKOptions, solve_ik_drake
from ..path_planner import ompl_parallel_plans, default_joint_planner_specs
from ..transformations import (
    rotation_matrix_to_quaternion,
    rpy_to_quaternion,
    rpy_to_rotation_matrix,
    quaternion_to_rotation_matrix,
    quaternion_multiply,
)
from .utils import clamp_vector, to_meters


@dataclass
class _TargetPose:
    position: np.ndarray
    quat_wxyz: np.ndarray
    R_des: np.ndarray
    orientation_deg: Tuple[float, float, float]
    frame: str


@dataclass
class _IKMetrics:
    pos_linf: float
    pos_err: float
    rot_err: float
    info: dict
    slack_used: bool = False


@dataclass
class CartesianPlanResult:
    plan_name: str
    waypoints: np.ndarray
    times: np.ndarray
    cost: float
    target_pos: np.ndarray
    target_quat: np.ndarray
    q_goal: np.ndarray


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def _frame_pose_world(config, frame_key: str, logger) -> Tuple[np.ndarray, np.ndarray]:
    key = (frame_key or "").lower()
    if key == "world" or key == "":
        return np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if key.startswith("obj:"):
        obj_name = key.split(":", 1)[1]
        for obj in config.objects:
            name = (obj.name or "").lower()
            if name == obj_name:
                pos = np.array(obj.position or (0.0, 0.0, 0.0), dtype=np.float64)
                rpy = np.array(obj.orientation_rpy or (0.0, 0.0, 0.0), dtype=np.float64)
                roll, pitch, yaw = np.deg2rad(rpy)
                quat = _normalize_quaternion(rpy_to_quaternion(roll, pitch, yaw))
                return pos, quat
        logger.warning(f"Unknown reference frame '{frame_key}', using world frame")
    return np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _pose_in_base(robot_config, config, frame_key: str, pos_local: np.ndarray, quat_local: np.ndarray, logger) -> Tuple[np.ndarray, np.ndarray]:
    pos_local = np.asarray(pos_local, dtype=np.float64)
    quat_local = _normalize_quaternion(quat_local)

    key = (frame_key or "base").lower()
    if key == "base":
        return pos_local, quaternion_to_rotation_matrix(quat_local)

    frame_pos, frame_quat = _frame_pose_world(config, key, logger)
    frame_quat = _normalize_quaternion(frame_quat)
    R_frame = quaternion_to_rotation_matrix(frame_quat)
    pos_world = R_frame @ pos_local + frame_pos
    quat_world = quaternion_multiply(frame_quat, quat_local)
    quat_world = _normalize_quaternion(quat_world)
    R_world = quaternion_to_rotation_matrix(quat_world)

    base_pos = np.array(robot_config.base_position or (0.0, 0.0, 0.0), dtype=np.float64)
    base_rpy = np.array(robot_config.base_orientation or (0.0, 0.0, 0.0), dtype=np.float64)
    roll_b, pitch_b, yaw_b = np.deg2rad(base_rpy)
    base_quat = _normalize_quaternion(rpy_to_quaternion(roll_b, pitch_b, yaw_b))
    R_world_base = quaternion_to_rotation_matrix(base_quat)
    R_base_world = R_world_base.T

    pos_base = R_base_world @ (pos_world - base_pos)
    R_base = R_base_world @ R_world
    return pos_base, R_base


def _coerce_q_dim(cache, q: np.ndarray) -> np.ndarray:
    nq = int(cache.plant.num_positions())
    q = np.asarray(q, dtype=np.float64).flatten()
    if q.shape[0] == nq:
        return q
    if q.shape[0] > nq:
        return q[:nq]
    return np.concatenate([q, np.zeros(nq - q.shape[0], dtype=np.float64)], axis=0)


def _initial_joint_state(runtime, cache, read_actual_joints, logger) -> np.ndarray:
    raw = read_actual_joints()
    if raw is None:
        actual = np.empty(0, dtype=np.float64)
    else:
        actual = np.asarray(raw, dtype=np.float64).flatten()
    nq = cache.plant.num_positions()
    if actual.size >= nq:
        logger.info(f"[{runtime.name}] Using Genesis joint state ({actual.size} dof) for planning")
        return actual[:nq]
    if actual.size > 0:
        q = np.zeros(nq, dtype=np.float64)
        q[: actual.size] = actual
        logger.info(f"[{runtime.name}] Using partial Genesis joint state ({actual.size} dof) for planning")
        return q
    if runtime.joint_targets:
        logger.info(
            f"[{runtime.name}] Genesis joints unavailable; falling back to GUI targets {runtime.joint_targets} deg"
        )
        return np.array([np.deg2rad(d) for d in runtime.joint_targets], dtype=np.float64)
    logger.warning(f"[{runtime.name}] No valid joint state found, using zeros")
    return np.zeros(nq, dtype=np.float64)


def _ik_slack(ctrl) -> float:
    slack = getattr(ctrl, "ik_tolerance_slack", None)
    if slack is None or slack < 1.0:
        return 1.002
    return float(slack)


def _resolve_target_pose(robot_config, config, ctrl, command, logger) -> _TargetPose:
    pos_local = np.array(to_meters(ctrl, command.position), dtype=np.float64)
    orientation_deg = command.orientation_deg
    if any(not np.isfinite(val) for val in orientation_deg):
        logger.warning(f"Invalid orientation values detected: {orientation_deg}, using (0,0,0)")
        orientation_deg = (0.0, 0.0, 0.0)
    roll, pitch, yaw = [np.deg2rad(x) for x in orientation_deg]
    quat_local = _normalize_quaternion(np.array(rpy_to_quaternion(roll, pitch, yaw), dtype=np.float64))

    frame_key = (command.frame or "base").lower()
    if frame_key == "base":
        target_pos_base = pos_local.copy()
        R_des_base = quaternion_to_rotation_matrix(quat_local)
    else:
        target_pos_base, R_des_base = _pose_in_base(
            robot_config,
            config,
            frame_key,
            pos_local,
            quat_local,
            logger,
        )

    z_min = float(ctrl.ground_plane_z) + 0.04
    if target_pos_base[2] < z_min - 1e-6:
        logger.error(
            f"Requested Z {target_pos_base[2]:.3f}m is below safe clearance {z_min:.3f}m. Aborting Cartesian move."
        )
        raise ValueError("target_below_ground")
    if target_pos_base[2] < z_min:
        target_pos_base[2] = z_min

    target_quat = np.array(rotation_matrix_to_quaternion(R_des_base), dtype=np.float64)
    target_quat = _normalize_quaternion(target_quat)
    logger.info(
        f"[{robot_config.name}] Target in BASE frame (frame={frame_key}): pos={np.round(target_pos_base,4)}, "
        f"quat_wxyz={np.round(target_quat,4)}"
    )

    return _TargetPose(
        position=target_pos_base,
        quat_wxyz=target_quat,
        R_des=R_des_base,
        orientation_deg=tuple(float(x) for x in orientation_deg),
        frame=frame_key,
    )


def _current_pose(cache, q) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    context = cache.plant.CreateDefaultContext()
    cache.plant.SetPositions(context, q)
    ee_pose = cache.plant.CalcRelativeTransform(context, cache.base_frame, cache.ee_frame)
    pos = ee_pose.translation()
    rpy = ee_pose.rotation().ToRollPitchYaw()
    return pos, (
        np.rad2deg(rpy.roll_angle()),
        np.rad2deg(rpy.pitch_angle()),
        np.rad2deg(rpy.yaw_angle()),
    )


def _progressive_ik(cache, q_start, target, is_valid, pos_tol, rot_tol, logger):
    start_pos, _ = _current_pose(cache, q_start)
    q_curr = q_start.copy()
    steps = 20
    for i in range(1, steps + 1):
        wp = start_pos + (i / steps) * (target.position - start_pos)
        q_next, info = solve_ik_drake(
            cache,
            q_seed=q_curr,
            target_pos_base_m=tuple(wp),
            target_quat_base_wxyz=tuple(target.quat_wxyz),
            is_state_valid=is_valid,
            opts=DrakeIKOptions(
                pos_tolerance_m=pos_tol,
                rot_tolerance_deg=float(max(rot_tol, 0.5)),
                max_random_seeds=8,
                seed_noise_rad=0.25,
                center_bias_weight=5e-3,
                seed_stick_weight=5e-3,
            ),
        )
        if q_next is None:
            logger.debug("Progressive IK failed at step %d/%d: %s", i, steps, info)
            return None, info
        q_curr = q_next
        if not is_valid(q_curr):
            logger.warning(
                "Progressive IK produced in-collision waypoint (%d/%d) - GOAL IN COLLISION!", i, steps
            )
            return None, {"reason": "goal_in_collision", "seed_idx": i, "ori_mode": info.get("ori_mode")}
    if not is_valid(q_curr):
        logger.warning("Progressive IK produced in-collision goal - GOAL IN COLLISION!")
        return None, {"reason": "goal_in_collision"}
    return q_curr, {"reason": "progressive_success"}


def _evaluate_pose(cache, q_goal, target) -> Tuple[_IKMetrics, np.ndarray, Tuple[float, float, float]]:
    context = cache.plant.CreateDefaultContext()
    cache.plant.SetPositions(context, q_goal)
    achieved_pose = cache.plant.CalcRelativeTransform(context, cache.base_frame, cache.ee_frame)
    achieved_pos = achieved_pose.translation()
    achieved_rot = achieved_pose.rotation()
    achieved_rpy = achieved_rot.ToRollPitchYaw()
    achieved_rpy_deg = (
        np.rad2deg(achieved_rpy.roll_angle()),
        np.rad2deg(achieved_rpy.pitch_angle()),
        np.rad2deg(achieved_rpy.yaw_angle()),
    )

    R_ach = achieved_rot.matrix()
    R_err = R_ach.T @ target.R_des
    tr = float(np.trace(R_err))
    tr_clamped = max(-1.0, min(3.0, tr))
    cos_theta = (tr_clamped - 1.0) / 2.0
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta_deg = float(np.degrees(np.arccos(cos_theta)))

    pos_diff = achieved_pos - target.position
    pos_err = float(np.linalg.norm(pos_diff))
    pos_linf = float(np.max(np.abs(pos_diff)))

    metrics = _IKMetrics(
        pos_linf=pos_linf,
        pos_err=pos_err,
        rot_err=theta_deg,
        info={},
    )
    return metrics, achieved_pos, achieved_rpy_deg


def _within_tolerance(metrics: _IKMetrics, pos_tol: float, rot_tol: float) -> bool:
    if metrics.pos_linf > pos_tol + 1e-6:
        return False
    if metrics.rot_err is not None and metrics.rot_err > rot_tol + 1e-6:
        return False
    return True


def _solve_cartesian_ik(cache, q_seed, target, ctrl, is_valid, logger, robot_name: str):
    pos_tol = float(ctrl.ik_pos_tolerance_m)
    rot_tol = float(ctrl.ik_rot_tolerance_deg)

    q_goal, info = solve_ik_drake(
        cache,
        q_seed=q_seed,
        target_pos_base_m=tuple(target.position),
        target_quat_base_wxyz=tuple(target.quat_wxyz),
        is_state_valid=is_valid,
        opts=DrakeIKOptions(
            pos_tolerance_m=pos_tol,
            rot_tolerance_deg=rot_tol,
            max_random_seeds=16,
            seed_noise_rad=0.35,
            center_bias_weight=1e-2,
            seed_stick_weight=5e-3,
        ),
    )

    if q_goal is None:
        logger.info("Single-shot IK failed, trying progressive IK...")
        q_goal, info = _progressive_ik(cache, q_seed, target, is_valid, pos_tol, rot_tol, logger)
        if q_goal is None:
            return None, None, None, info or {}

    metrics, achieved_pos, achieved_rpy = _evaluate_pose(cache, q_goal, target)
    metrics.info = info or {}
    logger.info(f"IK achieved: pos={tuple(achieved_pos)} RPY={achieved_rpy}")
    logger.info(
        "Orientation geodesic error = %.3f° (mode=%s)",
        metrics.rot_err,
        metrics.info.get("ori_mode", "n/a"),
    )

    if _within_tolerance(metrics, pos_tol, rot_tol):
        return q_goal, metrics, achieved_pos, metrics.info

    slack = _ik_slack(ctrl)
    if _within_tolerance(metrics, pos_tol * slack, rot_tol * slack):
        metrics.slack_used = True
        logger.warning(
            "IK solution for %s used tolerance slack (%.3fx): pos_linf=%.4fmm, ang_err=%.4f°",
            robot_name,
            slack,
            metrics.pos_linf * 1000.0,
            metrics.rot_err,
        )
        return q_goal, metrics, achieved_pos, metrics.info

    logger.info("Retrying IK with relaxed tolerances (slack %.3fx)", slack)
    q_retry, retry_info = solve_ik_drake(
        cache,
        q_seed=q_goal,
        target_pos_base_m=tuple(target.position),
        target_quat_base_wxyz=tuple(target.quat_wxyz),
        is_state_valid=is_valid,
        opts=DrakeIKOptions(
            pos_tolerance_m=pos_tol * slack,
            rot_tolerance_deg=rot_tol * slack,
            max_random_seeds=8,
            seed_noise_rad=0.25,
            center_bias_weight=5e-3,
            seed_stick_weight=5e-3,
        ),
    )

    if q_retry is not None:
        metrics_retry, achieved_pos, achieved_rpy = _evaluate_pose(cache, q_retry, target)
        metrics_retry.info = retry_info or {}
        if _within_tolerance(metrics_retry, pos_tol * slack, rot_tol * slack):
            metrics_retry.slack_used = True
            logger.warning(
                "IK solution for %s accepted with relaxed tolerance: pos_linf=%.4fmm, ang_err=%.4f°",
                robot_name,
                metrics_retry.pos_linf * 1000.0,
                metrics_retry.rot_err,
            )
            return q_retry, metrics_retry, achieved_pos, metrics_retry.info

    logger.error(
        "IK tolerance violated: pos_linf=%.3fmm (limit %.3fmm), ang_err=%.3f° (limit %.3f°)",
        metrics.pos_linf * 1000.0,
        pos_tol * 1000.0,
        metrics.rot_err,
        rot_tol,
    )
    return None, metrics, achieved_pos, metrics.info


def _repair_start_state(cache, q_current, is_valid, pos_tol, rot_tol, logger):
    if is_valid(q_current):
        return q_current
    try:
        context = cache.plant.CreateDefaultContext()
        cache.plant.SetPositions(context, q_current)
        ee_pose = cache.plant.CalcRelativeTransform(context, cache.base_frame, cache.ee_frame)
        pos_repair = ee_pose.translation().copy()
        pos_repair[2] += 0.03
        rot = ee_pose.rotation()
        quat = rot.ToQuaternion()
        q_repair, _ = solve_ik_drake(
            cache,
            q_seed=q_current,
            target_pos_base_m=tuple(pos_repair),
            target_quat_base_wxyz=(quat.w(), quat.x(), quat.y(), quat.z()),
            is_state_valid=is_valid,
            opts=DrakeIKOptions(
                pos_tolerance_m=pos_tol,
                rot_tolerance_deg=float(max(rot_tol, 0.5)),
                max_random_seeds=4,
                seed_noise_rad=0.1,
                center_bias_weight=1e-3,
                seed_stick_weight=1e-2,
            ),
        )
        if q_repair is not None and is_valid(q_repair):
            logger.debug("Start state invalid; applied +3cm Z IK repair to clear collisions.")
            return q_repair
        logger.error("Start state is invalid and repair IK failed; aborting planning")
        return None
    except Exception as exc:
        logger.error("Start state repair errored: %s", exc)
        return None


def plan_cartesian_move(
    runtime,
    command,
    config,
    logger,
    read_actual_joints: Callable[[], np.ndarray],
    update_all_states: Callable[[], None],
    build_state_valid: Callable[[], Callable[[np.ndarray], bool]],
    apply_runtime: bool = True,
) -> Optional[CartesianPlanResult]:
    """Plan a Cartesian move for ``runtime``.

    Returns a :class:`CartesianPlanResult` when successful.
    """
    cache = runtime.drake_cache
    if cache is None:
        logger.error(f"No IK cache for robot {runtime.name}")
        return None

    robot_config = runtime.config
    if not robot_config.end_effector_link:
        logger.error(f"No end effector link configured for {runtime.name}")
        return None

    ctrl = config.control_for(runtime.name)
    gui_units = (ctrl.cartesian_units or "m").lower()
    logger.info(
        f"[{runtime.name}] Cartesian request: gui_pos={tuple(command.position)} {gui_units}, "
        f"gui_rpy_deg={tuple(command.orientation_deg)}, frame={command.frame or 'base'}"
    )

    q_current = _coerce_q_dim(cache, _initial_joint_state(runtime, cache, read_actual_joints, logger))
    try:
        target = _resolve_target_pose(robot_config, config, ctrl, command, logger)
    except ValueError:
        return None

    current_pos, current_rpy_deg = _current_pose(cache, q_current)
    gap = float(np.linalg.norm(target.position - current_pos))
    if target.position[2] < 0.1:
        logger.warning(f"Target Z={target.position[2]:.3f}m may be too low (below robot base)")
    if float(np.linalg.norm(target.position)) > 1.5:
        logger.warning("Target distance may exceed workspace")

    logger.info(
        f"IK: Current EE at {tuple(current_pos)} RPY={current_rpy_deg}; target {tuple(target.position)} RPY={target.orientation_deg}"
    )
    logger.info(f"IK: Distance to target = {gap:.3f}m")

    update_all_states()
    is_valid = build_state_valid()
    logger.debug(f"Current position validity check: {is_valid(q_current)}")

    q_goal, metrics, achieved_pos, ik_info = _solve_cartesian_ik(
        cache, q_current, target, ctrl, is_valid, logger, runtime.name
    )
    logger.info(f"IK result: success={q_goal is not None}")
    if q_goal is None:
        logger.error(f"IK failed for {runtime.name}: {ik_info or 'unknown reason'}")
        return None

    if metrics and metrics.slack_used:
        logger.debug(
            f"[{runtime.name}] IK accepted with slack: pos_linf={metrics.pos_linf*1000:.3f}mm, ang_err={metrics.rot_err:.3f}°"
        )

    if not is_valid(q_goal):
        logger.error(f"[{runtime.name}] IK solution is in collision (environment/ground)")
        return None

    lower, upper = cache.lower, cache.upper
    q_current = clamp_vector(q_current, lower, upper)
    q_goal = clamp_vector(q_goal, lower, upper)

    repaired = _repair_start_state(cache, q_current, is_valid, float(ctrl.ik_pos_tolerance_m), float(ctrl.ik_rot_tolerance_deg), logger)
    if repaired is None:
        return None
    q_current = repaired

    timeout = float(ctrl.planner_timeout)
    planner_specs = default_joint_planner_specs(float(ctrl.planner_resolution))
    if not planner_specs:
        logger.error(f"[{runtime.name}] OMPL planners unavailable")
        return None

    plans = ompl_parallel_plans(
        planner_specs,
        q_current,
        q_goal,
        lower,
        upper,
        is_state_valid=is_valid,
        timeout_s=timeout,
        simplify=True,
    )

    if not plans:
        logger.warning(f"[{runtime.name}] No OMPL planners produced a path")
        return None

    plan_name, waypoints, times, cost = plans[0]
    logger.info(
        f"Selected {plan_name} plan for {runtime.name} ({waypoints.shape[0]} waypoints, {times[-1]:.2f}s duration, cost={cost:.3f})"
    )

    result = CartesianPlanResult(
        plan_name=plan_name,
        waypoints=waypoints,
        times=times,
        cost=cost,
        target_pos=np.array(target.position, dtype=np.float64),
        target_quat=np.array(target.quat_wxyz, dtype=np.float64),
        q_goal=q_goal.copy(),
    )

    if apply_runtime:
        runtime.last_target_pose = (result.target_pos.copy(), result.target_quat.copy())
        runtime.last_planned_q = result.q_goal.copy()

    return result


__all__ = ["plan_cartesian_move", "CartesianPlanResult"]
