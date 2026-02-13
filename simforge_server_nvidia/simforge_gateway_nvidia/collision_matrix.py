"""
Programmatic self-collision matrix generation for cuRobo.

MoveIt's Setup Assistant generates the Allowed Collision Matrix (ACM) by
sampling random joint configurations and checking which link-pairs collide
using *mesh* geometry.  cuRobo, however, uses *collision spheres* which are
larger than the actual meshes.  A link-pair that "never collides" in mesh
geometry may collide constantly in sphere geometry.

This module generates the self_collision_ignore list by sampling random
joint configurations and checking sphere-sphere distances for every link
pair.  Pairs that are **always** in collision (at *every* sampled config)
or **never** in collision are added to self_collision_ignore.  Only pairs
that collide in *some* configurations (i.e., the robot can move into and
out of collision) are left for cuRobo to check at runtime.

This matches MoveIt's algorithm but applied to cuRobo's sphere geometry.

Usage at startup
----------------
Called automatically during MotionGen initialization to compute
self_collision_ignore from the cuRobo robot config's collision spheres
and the URDF's kinematic chain.  No manual SRDF or hardcoded pairs
needed.
"""

import copy
import logging
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import numpy as np
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.types.robot import RobotConfig
    from curobo.types.base import TensorDeviceType
    _CUROBO_AVAILABLE = True
except ImportError:
    _CUROBO_AVAILABLE = False


def compute_self_collision_ignore(
    robot_cfg_dict: dict,
    num_samples: int = 5000,
    collision_threshold: float = 0.0,
    never_fraction: float = 0.0,
    always_fraction: float = 0.98,
) -> Dict[str, List[str]]:
    """Generate self_collision_ignore from cuRobo's own sphere geometry.

    Algorithm (matching MoveIt Setup Assistant logic):
    1. Build the cuRobo kinematic model (no collision checking — just FK).
    2. Sample ``num_samples`` random joint configurations within limits.
    3. For each sample, run FK to get sphere world positions.
    4. For every non-adjacent link pair, check if any sphere-sphere
       distance is less than ``collision_threshold``.
    5. Classify each pair:
       - **Adjacent**: parent–child in kinematic chain → always ignore
       - **Never**: collides in 0% of samples → ignore (can't collide)
       - **Always**: collides in ≥98% of samples → ignore (always in
         collision — checking would make all configs infeasible)
       - **Sometimes**: collides in some samples → DO NOT ignore (cuRobo
         should check these at runtime)

    Parameters
    ----------
    robot_cfg_dict : dict
        The cuRobo robot config dict (with ``robot_cfg.kinematics``).
    num_samples : int
        Number of random joint configurations to sample.
    collision_threshold : float
        Surface distance threshold for collision (0.0 = touching).
    never_fraction : float
        Maximum collision fraction to classify as "never" (default 0.0).
    always_fraction : float
        Minimum collision fraction to classify as "always" (default 0.98).

    Returns
    -------
    Dict[str, List[str]]
        The self_collision_ignore dict ready for cuRobo config.
    """
    if not _CUROBO_AVAILABLE:
        logger.warning("cuRobo not available — returning empty ignore list")
        return {}

    # ── Extract config fields ────────────────────────────────────
    kin_cfg = robot_cfg_dict
    if "robot_cfg" in robot_cfg_dict:
        kin_cfg = robot_cfg_dict["robot_cfg"]
    if "kinematics" in kin_cfg:
        kin_cfg = kin_cfg["kinematics"]

    collision_link_names: List[str] = kin_cfg.get("collision_link_names", [])
    collision_spheres: dict = kin_cfg.get("collision_spheres", {})

    if not collision_link_names or not collision_spheres:
        logger.warning(
            "No collision_link_names or collision_spheres in config "
            "— returning empty ignore list"
        )
        return {}

    # ── Build kinematic model ────────────────────────────────────
    # We need a temporary config with self_collision_ignore set to
    # ignore ALL pairs (disable self-collision entirely) since we
    # only need FK — not collision checking.
    cfg_copy = copy.deepcopy(robot_cfg_dict)
    kin_copy = cfg_copy
    if "robot_cfg" in cfg_copy:
        kin_copy = cfg_copy["robot_cfg"]
    if "kinematics" in kin_copy:
        kin_copy = kin_copy["kinematics"]

    # Ignore all collision pairs so RobotConfig doesn't reject configs
    all_ignore: Dict[str, List[str]] = {}
    for link in collision_link_names:
        others = [l for l in collision_link_names if l != link]
        if others:
            all_ignore[link] = others
    kin_copy["self_collision_ignore"] = all_ignore

    try:
        robot_config = RobotConfig.from_dict(cfg_copy)
        kin_model = CudaRobotModel(robot_config.kinematics)
    except Exception as e:
        logger.error(f"Failed to build kinematic model: {e}")
        return {}

    # ── Get joint limits ─────────────────────────────────────────
    # get_joint_limits() returns a JointLimits dataclass with:
    #   .position: (2, n_dof) — [0] = lower, [1] = upper
    joint_limits = kin_model.get_joint_limits()
    lower = joint_limits.position[0].cpu().numpy().flatten()  # (n_dof,)
    upper = joint_limits.position[1].cpu().numpy().flatten()
    n_dof = len(lower)

    # ── Identify adjacent link pairs ─────────────────────────────
    # Adjacent = directly connected by a joint in the collision chain.
    # These are always ignored since collision spheres of connected
    # links inherently overlap at the joint.
    adjacent_pairs: Set[Tuple[str, str]] = set()
    for i in range(len(collision_link_names) - 1):
        pair = tuple(sorted([
            collision_link_names[i], collision_link_names[i + 1]
        ]))
        adjacent_pairs.add(pair)

    logger.info(f"Adjacent pairs (always ignored): {len(adjacent_pairs)}")
    for a, b in sorted(adjacent_pairs):
        logger.info(f"  Adjacent: {a} ↔ {b}")

    # ── Map link names to sphere indices in the flat tensor ──────
    # CudaRobotModel stores sphere-to-link mapping in kinematics_config.
    # link_sphere_idx_map: (total_spheres,) — each element is the link
    # index that sphere belongs to.
    # link_name_to_idx_map: Dict[str, int] — link name → link index.
    kc = kin_model.kinematics_config
    link_name_to_idx: Dict[str, int] = kc.link_name_to_idx_map
    sphere_idx_map = kc.link_sphere_idx_map.cpu()  # (total_spheres,)

    link_to_sphere_indices: Dict[str, List[int]] = {}
    for link_name in collision_link_names:
        if link_name not in link_name_to_idx:
            logger.warning(
                f"Link '{link_name}' not in link_name_to_idx_map — "
                f"available: {list(link_name_to_idx.keys())}"
            )
            continue
        link_idx = link_name_to_idx[link_name]
        sphere_indices = torch.nonzero(
            sphere_idx_map == link_idx
        ).view(-1).tolist()
        link_to_sphere_indices[link_name] = sphere_indices
        logger.info(
            f"  Link {link_name}: {len(sphere_indices)} sphere(s) "
            f"(indices {sphere_indices})"
        )

    # ── Sample random joint configs ──────────────────────────────
    logger.info(
        f"Sampling {num_samples} random configs to build collision matrix "
        f"({len(collision_link_names)} links, {n_dof} DOF)"
    )

    rng = np.random.default_rng(42)
    q_samples = rng.uniform(
        lower, upper, size=(num_samples, n_dof)
    ).astype(np.float32)

    # Include important known configs: zeros and retract/home
    retract = kin_copy.get("cspace", {}).get(
        "retract_config", [0.0] * n_dof
    )
    special_configs = np.array([
        [0.0] * n_dof,
        retract,
    ], dtype=np.float32)
    q_all = np.vstack([special_configs, q_samples])
    total_samples = len(q_all)

    # Build non-adjacent link pairs to check
    pairs_to_check: List[Tuple[str, str]] = []
    for i in range(len(collision_link_names)):
        for j in range(i + 1, len(collision_link_names)):
            a, b = collision_link_names[i], collision_link_names[j]
            pair = tuple(sorted([a, b]))
            if pair not in adjacent_pairs:
                pairs_to_check.append((a, b))

    logger.info(f"Non-adjacent pairs to check: {len(pairs_to_check)}")

    # ── Run FK in batches and count per-sample collisions ────────
    collision_counts: Dict[Tuple[str, str], int] = {
        (a, b): 0 for a, b in pairs_to_check
    }
    batch_size = min(512, total_samples)
    q_tensor = torch.tensor(q_all, dtype=torch.float32).cuda()

    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        q_batch = q_tensor[batch_start:batch_end]
        bs = batch_end - batch_start

        kin_state = kin_model.get_state(q_batch)
        # link_spheres_tensor: (batch, total_spheres, 4) — [x,y,z,r]
        spheres = kin_state.link_spheres_tensor
        if spheres.dim() == 4:
            # Some versions return (batch, 1, total_spheres, 4)
            spheres = spheres.squeeze(1)

        spheres_np = spheres.cpu().numpy()  # (batch, total_spheres, 4)

        for pair_a, pair_b in pairs_to_check:
            idx_a = link_to_sphere_indices.get(pair_a, [])
            idx_b = link_to_sphere_indices.get(pair_b, [])
            if not idx_a or not idx_b:
                continue

            # For each sample in this batch, check if ANY sphere pair
            # between these two links collides
            colliding_any = np.zeros(bs, dtype=bool)
            for ia in idx_a:
                for ib in idx_b:
                    pos_a = spheres_np[:, ia, :3]  # (batch, 3)
                    pos_b = spheres_np[:, ib, :3]
                    rad_a = spheres_np[:, ia, 3]   # (batch,)
                    rad_b = spheres_np[:, ib, 3]
                    dist = np.linalg.norm(pos_a - pos_b, axis=1)
                    surface_dist = dist - rad_a - rad_b
                    colliding_any |= (surface_dist < collision_threshold)

            collision_counts[(pair_a, pair_b)] += int(colliding_any.sum())

    # ── Classify pairs ───────────────────────────────────────────
    ignore_pairs: Set[Tuple[str, str]] = set()

    # 1. Adjacent pairs: always ignore
    for pair in adjacent_pairs:
        ignore_pairs.add(pair)

    # 2. Never/Always pairs from sampling
    for (a, b), count in collision_counts.items():
        fraction = count / total_samples
        pair = tuple(sorted([a, b]))

        if fraction <= never_fraction:
            ignore_pairs.add(pair)
            logger.info(
                f"  Never collides: {a} ↔ {b} "
                f"({count}/{total_samples} = {fraction:.1%}) → IGNORE"
            )
        elif fraction >= always_fraction:
            # Always in collision — must ignore or cuRobo will reject
            # all configurations.  The spheres permanently overlap.
            ignore_pairs.add(pair)
            logger.info(
                f"  Always collides: {a} ↔ {b} "
                f"({count}/{total_samples} = {fraction:.1%}) → IGNORE "
                f"(spheres permanently overlap)"
            )
        else:
            logger.info(
                f"  Sometimes collides: {a} ↔ {b} "
                f"({count}/{total_samples} = {fraction:.1%}) → CHECK"
            )

    # ── Build the output dict ────────────────────────────────────
    # Format: { linkA: [linkB, linkC], ... } — each pair listed once
    # under the link that comes first in collision_link_names order.
    result: Dict[str, List[str]] = {}
    seen: Set[Tuple[str, str]] = set()

    for link in collision_link_names:
        ignore_list: List[str] = []
        for other in collision_link_names:
            if other == link:
                continue
            pair = tuple(sorted([link, other]))
            if pair in seen:
                continue
            if pair in ignore_pairs:
                ignore_list.append(other)
                seen.add(pair)
        if ignore_list:
            result[link] = ignore_list

    checked = len(pairs_to_check) - len(
        [p for p in pairs_to_check if tuple(sorted(p)) in ignore_pairs]
    )
    logger.info(
        f"Self-collision matrix complete: "
        f"{len(ignore_pairs)} ignored pairs, {checked} checked pairs"
    )

    return result
