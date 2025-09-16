"""Inverse kinematics and motion planning helpers."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import time
from typing import Callable, Optional, Tuple
import numpy as np

from ..ik_drake import DrakeIKOptions, solve_ik_drake
from ..path_planner import cartesian_linear_plan, ompl_rrt_connect_plan
from ..transformations import (
    rotation_matrix_to_quaternion,
    rpy_to_quaternion,
    rpy_to_rotation_matrix,
)
from .utils import clamp_vector, rad_to_deg_list, to_meters


def _coerce_q_dim(cache, q: np.ndarray) -> np.ndarray:
    nq = int(cache.plant.num_positions())
    q = np.asarray(q, dtype=np.float64).flatten()
    if q.shape[0] == nq:
        return q
    if q.shape[0] > nq:
        return q[:nq]
    return np.concatenate([q, np.zeros(nq - q.shape[0], dtype=np.float64)], axis=0)


def plan_cartesian_move(
    runtime,
    command,
    config,
    logger,
    read_actual_joints: Callable[[], np.ndarray],
    update_all_states: Callable[[], None],
    build_state_valid: Callable[[], Callable[[np.ndarray], bool]],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Plan a Cartesian move for ``runtime``.

    Returns ``(waypoints, times)`` in radians and seconds when successful.
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

    actual_joints = read_actual_joints()
    if actual_joints.size >= 6:
        q_current = np.zeros(cache.plant.num_positions())
        q_current[: actual_joints.size] = actual_joints
        logger.info(
            f"[{runtime.name}] Using Genesis joint state ({actual_joints.size} dof) for planning"
        )
    elif runtime.joint_targets:
        q_current = np.array([np.deg2rad(d) for d in runtime.joint_targets], dtype=np.float64)
        logger.info(
            f"[{runtime.name}] Genesis joints unavailable; falling back to GUI targets {runtime.joint_targets} deg"
        )
    else:
        q_current = np.zeros(cache.plant.num_positions())
        logger.warning("No valid joint state found, using zeros")

    q_current = _coerce_q_dim(cache, q_current)
    pos_world_m = to_meters(ctrl, command.position)
    orientation_deg = command.orientation_deg
    if any(not np.isfinite(val) for val in orientation_deg):
        logger.warning(
            f"Invalid orientation values detected: {orientation_deg}, using (0,0,0)"
        )
        orientation_deg = (0.0, 0.0, 0.0)
    roll, pitch, yaw = [np.deg2rad(x) for x in orientation_deg]
    quat_wxyz = rpy_to_quaternion(roll, pitch, yaw)
    quat_mag = np.linalg.norm(quat_wxyz)
    if quat_mag < 1e-10:
        logger.error("Invalid quaternion magnitude; using identity quaternion")
        quat_wxyz = (1.0, 0.0, 0.0, 0.0)
    else:
        quat_wxyz = tuple(q / quat_mag for q in quat_wxyz)

    gui_frame = (command.frame or "base").lower()
    R_des_bl = rpy_to_rotation_matrix(roll, pitch, yaw)

    if gui_frame == "base":
        target_pos_base = np.array(pos_world_m, dtype=np.float64)
        R_des_base = R_des_bl
        target_quat_base = tuple(rotation_matrix_to_quaternion(R_des_base))
    else:
        robot_base_pos = np.array(robot_config.base_position or [0.0, 0.0, 0.0])
        p_bl = np.array(pos_world_m, dtype=np.float64) - robot_base_pos
        plant_ctx_tmp = cache.plant.CreateDefaultContext()
        frame_base = cache.base_frame
        try:
            frame_bl = cache.plant.GetFrameByName(getattr(cache, "base_link", "base_link"))
        except Exception:
            frame_bl = cache.plant.GetFrameByName("base_link")
        T_base_bl = cache.plant.CalcRelativeTransform(plant_ctx_tmp, frame_base, frame_bl)
        R_base_bl = T_base_bl.rotation().matrix()
        t_base_bl = T_base_bl.translation()
        target_pos_base = R_base_bl @ p_bl + t_base_bl
        R_des_base = R_base_bl @ R_des_bl
        target_quat_base = tuple(rotation_matrix_to_quaternion(R_des_base))

        logger.info(
            f"World target: {pos_world_m}, base_link->BASE translation {np.round(t_base_bl, 4)}"
        )

    logger.info(
        f"[{runtime.name}] Target in BASE frame: pos={np.round(target_pos_base,4)}, "
        f"quat_wxyz={np.round(target_quat_base,4)}"
    )

    z_min = float(ctrl.ground_plane_z) + 0.04
    if target_pos_base[2] < z_min:
        logger.warning(
            f"Clamping target Z from {target_pos_base[2]:.3f}m to {z_min:.3f}m to avoid near-ground singularities"
        )
        target_pos_base[2] = z_min

    # FK for logging
    plant_context = cache.plant.CreateDefaultContext()
    cache.plant.SetPositions(plant_context, q_current)
    ee_pose = cache.plant.CalcRelativeTransform(plant_context, cache.base_frame, cache.ee_frame)
    current_pos = ee_pose.translation()
    current_rot = ee_pose.rotation()
    current_rpy = current_rot.ToRollPitchYaw()
    current_rpy_deg = (
        np.rad2deg(current_rpy.roll_angle()),
        np.rad2deg(current_rpy.pitch_angle()),
        np.rad2deg(current_rpy.yaw_angle()),
    )

    start_gap = float(np.linalg.norm(np.array(target_pos_base) - current_pos))
    if target_pos_base[2] < 0.1:
        logger.warning(
            f"Target Z={target_pos_base[2]:.3f}m may be too low (below robot base)"
        )
    if float(np.linalg.norm(target_pos_base)) > 1.5:
        logger.warning("Target distance may exceed workspace")

    logger.info(
        f"IK: Current EE at {tuple(current_pos)} RPY={current_rpy_deg}; target {tuple(target_pos_base)} RPY={orientation_deg}"
    )
    logger.info(f"IK: Distance to target = {start_gap:.3f}m")

    def _clamp_to_limits(vec: np.ndarray) -> np.ndarray:
        lower, upper = cache.lower, cache.upper
        nq = cache.plant.num_positions()
        vec = np.asarray(vec, dtype=np.float64).flatten()
        if vec.shape[0] > nq:
            vec = vec[:nq]
        if vec.shape[0] < nq:
            vec = np.concatenate([vec, np.zeros(nq - vec.shape[0])], axis=0)
        return clamp_vector(vec, lower, upper)

    seeds = [q_current.copy()]
    nq = cache.plant.num_positions()
    zero_config = np.zeros(nq, dtype=np.float64)
    seeds.append(_clamp_to_limits(zero_config))
    if nq >= 6:
        elbow_up = np.array([0.0, -np.pi / 3, np.pi / 3, -np.pi / 3, -np.pi / 3, 0.0], dtype=np.float64)
        elbow_down = np.array([0.0, np.pi / 4, -np.pi / 4, np.pi / 4, np.pi / 4, 0.0], dtype=np.float64)
        home_pose = np.array([0.0, -np.pi / 6, np.pi / 4, 0.0, np.pi / 3, 0.0], dtype=np.float64)
        seeds.extend([
            _clamp_to_limits(elbow_up),
            _clamp_to_limits(elbow_down),
            _clamp_to_limits(home_pose),
        ])
    try:
        target_angle = float(np.arctan2(target_pos_base[1], target_pos_base[0]))
        if nq >= 1:
            biased = q_current.copy()
            biased[0] = target_angle
            seeds.append(_clamp_to_limits(biased))
    except Exception:
        pass

    rng = np.random.RandomState(42)
    for base in [q_current, zero_config]:
        for _ in range(6):
            noise = rng.uniform(-0.2, 0.2, size=nq)
            seeds.append(_clamp_to_limits(base + noise))

    logger.debug(f"[{runtime.name}] IK seed count: {len(seeds)} (base nq={nq})")

    update_all_states()
    is_valid = build_state_valid()
    logger.debug(f"Current position validity check: {is_valid(q_current)}")

    q_goal = None
    last_info = {}
    q_try, info = solve_ik_drake(
        cache,
        q_seed=q_current,
        target_pos_base_m=tuple(target_pos_base),
        target_quat_base_wxyz=tuple(quat_wxyz),
        is_state_valid=is_valid,
        opts=DrakeIKOptions(
            pos_tolerance_m=float(ctrl.ik_pos_tolerance_m),
            rot_tolerance_deg=float(ctrl.ik_rot_tolerance_deg),
            max_random_seeds=16,
            seed_noise_rad=0.35,
            center_bias_weight=1e-2,
            seed_stick_weight=5e-3,
        ),
    )
    last_info = info
    if q_try is not None:
        q_goal = q_try
    else:
        logger.info("Single-shot IK failed, trying progressive IK...")
        start_pos = current_pos
        q_curr = q_current.copy()
        steps = 20
        for i in range(1, steps + 1):
            wp = start_pos + (i / steps) * (np.array(target_pos_base) - start_pos)
            q_next, info = solve_ik_drake(
                cache,
                q_seed=q_curr,
                target_pos_base_m=tuple(wp),
                target_quat_base_wxyz=tuple(quat_wxyz),
                is_state_valid=is_valid,
                opts=DrakeIKOptions(
                    pos_tolerance_m=float(ctrl.ik_pos_tolerance_m),
                    rot_tolerance_deg=float(max(ctrl.ik_rot_tolerance_deg, 0.5)),
                    max_random_seeds=8,
                    seed_noise_rad=0.25,
                    center_bias_weight=5e-3,
                    seed_stick_weight=5e-3,
                ),
            )
            last_info = info
            if q_next is None:
                logger.debug(
                    f"[{runtime.name}] Progressive IK failed at step {i}/{steps}: {info}"
                )
                break
            q_curr = q_next
            if not is_valid(q_curr):
                last_info = {
                    "reason": "goal_in_collision",
                    "seed_idx": i,
                    "ori_mode": info.get("ori_mode"),
                }
                logger.debug(
                    f"[{runtime.name}] Progressive IK produced in-collision waypoint ({i}/{steps})"
                )
                break
            if i == steps and is_valid(q_curr):
                q_goal = q_curr
                break
        if q_goal is not None and not is_valid(q_goal):
            logger.debug("Progressive IK produced in-collision goal")
            q_goal = None

    logger.info(f"IK result: success={q_goal is not None}")
    if q_goal is None:
        if last_info.get("reason") == "did_not_converge":
            logger.error(
                "IK failed to converge to the requested pose after %d seeds (pos_err=%.3fm, rot_err=%.3frad)."
                % (len(seeds), last_info.get("pos_err", 0), last_info.get("rot_err", 0))
            )
        else:
            logger.error(
                f"IK failed for {runtime.name} after {len(seeds)} seeds: {last_info or {'reason': 'unknown'}}"
            )
        return None

    plant_context_check = cache.plant.CreateDefaultContext()
    cache.plant.SetPositions(plant_context_check, q_goal)
    achieved_pose = cache.plant.CalcRelativeTransform(
        plant_context_check, cache.base_frame, cache.ee_frame
    )
    achieved_pos = achieved_pose.translation()
    achieved_rot = achieved_pose.rotation()
    achieved_rpy = achieved_rot.ToRollPitchYaw()
    achieved_rpy_deg = (
        np.rad2deg(achieved_rpy.roll_angle()),
        np.rad2deg(achieved_rpy.pitch_angle()),
        np.rad2deg(achieved_rpy.yaw_angle()),
    )
    logger.info(f"IK achieved: pos={tuple(achieved_pos)} RPY={achieved_rpy_deg}")

    try:
        R_ach = achieved_rot.matrix()
        R_err = R_ach.T @ (R_des_base)
        tr = float(np.trace(R_err))
        tr_clamped = max(-1.0, min(3.0, tr))
        cos_theta = (tr_clamped - 1.0) / 2.0
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta_deg = float(np.degrees(np.arccos(cos_theta)))
        logger.info(
            f"Orientation geodesic error = {theta_deg:.3f}° (mode={last_info.get('ori_mode')})"
        )
    except Exception as exc:
        logger.debug(f"Geodesic orientation error computation failed: {exc}")

    lower = cache.lower
    upper = cache.upper
    q_current = clamp_vector(q_current, lower, upper)
    q_goal = clamp_vector(q_goal, lower, upper)

    if not is_valid(q_current):
        try:
            plant_context = cache.plant.CreateDefaultContext()
            cache.plant.SetPositions(plant_context, q_current)
            ee_pose = cache.plant.CalcRelativeTransform(
                plant_context, cache.base_frame, cache.ee_frame
            )
            pos_repair = ee_pose.translation().copy()
            pos_repair[2] += 0.03
            rot = ee_pose.rotation()
            quat_repair = rot.ToQuaternion()
            quat_repair_wxyz = (
                quat_repair.w(),
                quat_repair.x(),
                quat_repair.y(),
                quat_repair.z(),
            )
            q_repair, _ = solve_ik_drake(
                cache,
                q_seed=q_current,
                target_pos_base_m=tuple(pos_repair),
                target_quat_base_wxyz=quat_repair_wxyz,
                is_state_valid=is_valid,
                opts=DrakeIKOptions(
                    pos_tolerance_m=float(ctrl.ik_pos_tolerance_m),
                    rot_tolerance_deg=float(max(ctrl.ik_rot_tolerance_deg, 0.5)),
                    max_random_seeds=4,
                    seed_noise_rad=0.1,
                    center_bias_weight=1e-3,
                    seed_stick_weight=1e-2,
                ),
            )
            if q_repair is not None and is_valid(q_repair):
                logger.debug("Start state invalid; applied +3cm Z IK repair to clear collisions.")
                q_current = q_repair
            else:
                logger.error("Start state is invalid and repair IK failed; aborting planning")
                return None
        except Exception as exc:
            logger.debug(f"Start repair IK failed: {exc}")
            logger.error("Start state is invalid and repair IK errored; aborting planning")
            return None

    timeout = float(ctrl.planner_timeout)
    num_waypoints = int(ctrl.cartesian_waypoints)

    def plan_rrt():
        logger.info(
            f"[{runtime.name}] Launching RRTConnect (timeout={timeout:.2f}s, range={float(ctrl.planner_resolution):.3f})"
        )
        return ompl_rrt_connect_plan(
            q_current,
            q_goal,
            lower,
            upper,
            is_state_valid=is_valid,
            timeout_s=timeout,
            range_rad=float(ctrl.planner_resolution),
            simplify=True,
        )

    plant_context = cache.plant.CreateDefaultContext()
    cache.plant.SetPositions(plant_context, q_current)
    start_pose = cache.plant.CalcRelativeTransform(
        plant_context, cache.base_frame, cache.ee_frame
    )
    start_pos = start_pose.translation()
    start_rot = start_pose.rotation()
    start_quat = start_rot.ToQuaternion()
    start_pose_tuple = (
        start_pos,
        (
            start_quat.w(),
            start_quat.x(),
            start_quat.y(),
            start_quat.z(),
        ),
    )
    target_pose_tuple = (target_pos_base, tuple(quat_wxyz))

    def _ik(q_seed, pose):
        pos, quat = pose
        result, _ = solve_ik_drake(
            cache,
            q_seed=q_seed,
            target_pos_base_m=tuple(pos),
            target_quat_base_wxyz=tuple(quat),
            is_state_valid=is_valid,
            opts=DrakeIKOptions(
                pos_tolerance_m=float(ctrl.ik_pos_tolerance_m),
                rot_tolerance_deg=float(ctrl.ik_rot_tolerance_deg),
                max_random_seeds=4,
                seed_noise_rad=0.25,
                center_bias_weight=5e-3,
                seed_stick_weight=5e-3,
            ),
        )
        return result

    def plan_cart():
        logger.info(
            f"[{runtime.name}] Launching Cartesian linear planner with {num_waypoints} waypoints"
        )
        return cartesian_linear_plan(
            start_q=q_current,
            start_pose_se3=start_pose_tuple,
            target_pose_se3=target_pose_tuple,
            solve_ik=_ik,
            is_state_valid=is_valid,
            num_waypoints=num_waypoints,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_rrt = executor.submit(plan_rrt)
        future_lin = executor.submit(plan_cart)
        wait({future_rrt, future_lin}, timeout=timeout, return_when=ALL_COMPLETED)

    plans = []
    if future_lin.done():
        result = future_lin.result()
        if result is not None:
            plans.append(("cartesian", *result))
        else:
            logger.info(f"[{runtime.name}] Cartesian planner returned no solution")
    else:
        logger.warning(f"[{runtime.name}] Cartesian planner timed out")
    if future_rrt.done():
        result = future_rrt.result()
        if result is not None:
            plans.append(("rrt", *result))
        else:
            logger.info(f"[{runtime.name}] RRT planner returned no solution")
    else:
        logger.warning(f"[{runtime.name}] RRT planner timed out")

    def _validate_path(way):
        from ..path_planner import _check_segment_collision_free

        for i in range(len(way) - 1):
            if not _check_segment_collision_free(way[i], way[i + 1], is_valid, resolution=20):
                logger.debug(
                    f"[{runtime.name}] Path segment {i}->{i+1} failed collision validation"
                )
                return False
        return True

    if not plans:
        logger.warning(f"[{runtime.name}] No planners produced a path")
        return None

    valid = [item for item in plans if _validate_path(item[1])]
    if not valid:
        logger.warning(
            f"Planning failed or produced colliding paths for {runtime.name}; candidates={[(name, way.shape[0]) for name, way, _ in plans]}"
        )
        return None

    valid.sort(key=lambda x: 0 if x[0] == "cartesian" else 1)
    plan_name, waypoints, times = valid[0]
    logger.info(
        f"Selected {plan_name} plan for {runtime.name} ({waypoints.shape[0]} waypoints, {times[-1]:.2f}s duration)"
    )

    runtime.last_target_pose = (
        np.array(target_pos_base, dtype=np.float64),
        np.array(quat_wxyz, dtype=np.float64),
    )
    runtime.last_planned_q = waypoints[-1].copy()

    return waypoints, times


__all__ = ["plan_cartesian_move"]
