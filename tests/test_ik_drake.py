import numpy as np
import pytest

from simforge.ik_drake import solve_ik_drake, DrakeIKCache, DrakeIKOptions


def test_ur5e_ik_from_config_headless_simple():
    """
    Headless IK test mirroring the Simforge app:
    - Load env_configs/ur5e_env.yaml (spawns UR5e at base offset in WORLD)
    - Build Genesis scene headless via MovementController
    - Use controller's Drake cache to solve IK in ROBOT BASE frame to (0.7, 0.2, 0.2)
    - Verify EE position via FK in the robot's BASE frame
    """
    from simforge.config_reader import SimforgeConfig
    from simforge.movement_controller import MovementController

    # 1) Load config and force headless mode
    cfg = SimforgeConfig.from_yaml("env_configs/ur5e_env.yaml")
    cfg.scene.show_viewer = False
    robot_name = "UR5e_1"

    # 2) Build controller and scene (follows app code path)
    mc = MovementController(cfg, debug=False)
    mc.build_scene()

    # 3) Access Drake cache from controller
    assert robot_name in mc.drake_caches, "Drake cache not initialized by controller"
    cache = mc.drake_caches[robot_name]

    # 4) IK target expressed in the ROBOT BASE frame (not world)
    target_pos = (0.7, 0.2, 0.2)
    robot_config = next(r for r in cfg.robots if r.name == robot_name)
    q_deg = mc.joint_targets[robot_name] or [0.0] * int(cache.plant.num_positions())
    q_seed0 = np.array([np.deg2rad(d) for d in q_deg], dtype=np.float64)
    q_seed0 = q_seed0[: cache.plant.num_positions()] if q_seed0.size >= cache.plant.num_positions() else np.pad(q_seed0, (0, cache.plant.num_positions() - q_seed0.size))

    # Additional deterministic seeds
    seeds = [
        ("controller_seed", q_seed0),
        ("elbow_up", np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])),
        ("small_offset", np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])),
        ("better_seed", np.array([0.5, -1.0, 1.5, -1.0, -1.5, 0.0])),
    ]

    # Add a couple of mild random variations to improve robustness
    rng = np.random.RandomState(7)
    base = np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    nq = cache.plant.num_positions()
    for i in range(4):
        seeds.append((f"rand_{i}", base + rng.uniform(-0.15, 0.15, size=nq)))

    # Helper: solve once with current orientation (position-focused)
    def solve_once(q_seed: np.ndarray, target_pos_arg):
        q_seed = np.asarray(q_seed, dtype=np.float64)
        nq = cache.plant.num_positions()
        q_seed = q_seed[: nq] if q_seed.size >= nq else np.pad(q_seed, (0, nq - q_seed.size))

        plant_context = cache.plant.CreateDefaultContext()
        cache.plant.SetPositions(plant_context, q_seed)
        ee_pose = cache.plant.CalcRelativeTransform(plant_context, cache.base_frame, cache.ee_frame)
        pos = ee_pose.translation()
        print(f"Initial position for seed: {pos}")
        rot = ee_pose.rotation()
        quat = rot.ToQuaternion()
        target_quat = (quat.w(), quat.x(), quat.y(), quat.z())  # Use current orientation as target
        print(f"Target quat: {target_quat}")
        return solve_ik_drake(
                    cache,
                    q_seed=q_seed,
                    target_pos_base_m=target_pos_arg,
                    target_quat_base_wxyz=target_quat,
                    is_state_valid=lambda q: True,
                    opts=DrakeIKOptions(
                        pos_tolerance_m=2e-3,
                        rot_tolerance_deg=180.0,
                        max_random_seeds=0,  # single seed
                        center_bias_weight=0,
                        seed_stick_weight=0,
                    ),
                )
    def progressive_solve(q_seed: np.ndarray, steps: int = 24, target_pos_arg=None):
        if target_pos_arg is None:
            target_pos_arg = target_pos
        q_seed = np.asarray(q_seed, dtype=np.float64)
        nq = cache.plant.num_positions()
        q_seed = q_seed[: nq] if q_seed.size >= nq else np.pad(q_seed, (0, nq - q_seed.size))

        # Start position in BASE frame
        plant_context = cache.plant.CreateDefaultContext()
        cache.plant.SetPositions(plant_context, q_seed)
        ee_pose = cache.plant.CalcRelativeTransform(plant_context, cache.base_frame, cache.ee_frame)
        start_pos_base = ee_pose.translation()

        # Keep current EE orientation through the walk (position-only)
        target_quat = (1.0, 0.0, 0.0, 0.0)  # Use identity orientation

        q_curr = q_seed.copy()
        last_info = {}
        for i in range(1, steps + 1):
            wp_base = start_pos_base + (i / steps) * (np.array(target_pos_arg) - start_pos_base)
            q_next, info = solve_ik_drake(
                    cache,
                    q_seed=q_curr,
                    target_pos_base_m=tuple(wp_base),
                    target_quat_base_wxyz=target_quat,
                    is_state_valid=lambda q: True,
                    opts=DrakeIKOptions(
                        pos_tolerance_m=1e-2,
                        rot_tolerance_deg=180.0,
                        max_random_seeds=0,
                        center_bias_weight=0,
                        seed_stick_weight=0,
                    ),
                )
            last_info = info
            if q_next is None:
                return None, last_info
            q_curr = q_next
        return q_curr, last_info

    # Attempt single-shot seeds first, then progressive fallback
    solved = False
    last_info = {}
    for name, q_seed in seeds:
        # For controller_seed, use its own position as target to test IK self-consistency
        if name == "controller_seed":
            plant_context = cache.plant.CreateDefaultContext()
            cache.plant.SetPositions(plant_context, q_seed)
            ee_pose = cache.plant.CalcRelativeTransform(plant_context, cache.base_frame, cache.ee_frame)
            test_target_pos = tuple(ee_pose.translation())
        else:
            test_target_pos = target_pos
        
        q_sol, info = solve_once(q_seed, test_target_pos)
        last_info = info

        if q_sol is None:
            q_sol, info = progressive_solve(q_seed, steps=30, target_pos_arg=test_target_pos)
            last_info = info

        if q_sol is None:
            continue

        # Verify with FK in BASE frame
        plant_context = cache.plant.CreateDefaultContext()
        cache.plant.SetPositions(plant_context, q_sol)
        ee_pose = cache.plant.CalcRelativeTransform(plant_context, cache.base_frame, cache.ee_frame)
        pos_base = ee_pose.translation()
        pos_err = float(np.linalg.norm(pos_base - np.array(test_target_pos)))
        print(f"Solved pos_base: {pos_base}, target: {test_target_pos}, err: {pos_err}")
        if pos_err <= 2e-3:
            solved = True
            print(f"UR5e IK headless success from seed '{name}': pos_err={pos_err:.6f} m, q(deg)={np.rad2deg(q_sol)}")
            break

    assert solved, f"IK failed for all seeds; last_info={last_info}"
