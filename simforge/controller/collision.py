"""Collision checker helpers for movement controller."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

from ..collision_checker import CollisionChecker

if TYPE_CHECKING:
    from ..config_reader import SimforgeConfig
    from .robot_runtime import RobotRuntime
    from ..tooling import ToolManager


def create_collision_checker(
    robot_config,
    config: "SimforgeConfig",
    logger,
    tool_manager: Optional["ToolManager"] = None,
) -> Optional[CollisionChecker]:
    """Build a ``CollisionChecker`` for the given robot configuration."""
    ctrl = config.control_for(robot_config.name)

    world_boxes: List[Tuple[str, Any, Tuple[float, float, float], Tuple[float, float, float]]] = []
    for obj in config.objects:
        if obj.collision_enabled is False:
            continue
        if obj.type == "box" and obj.size:
            world_boxes.append(
                (
                    obj.name or "box",
                    tuple(obj.size),
                    tuple(obj.position),
                    tuple(obj.orientation_rpy),
                )
            )
        elif obj.type == "sphere" and obj.size:
            radius = float(obj.size[0])
            world_boxes.append(
                (
                    obj.name or "sphere",
                    {"type": "sphere", "radius": radius},
                    tuple(obj.position),
                    tuple(obj.orientation_rpy or (0.0, 0.0, 0.0)),
                )
            )
        elif obj.type == "plane":
            size = obj.size or [10.0, 10.0, 0.02]
            thickness = float(size[2])
            gz = float(ctrl.ground_plane_z)
            plane_center = list(obj.position or [0.0, 0.0, 0.0])
            plane_center[2] = gz - thickness * 0.5
            world_boxes.append(
                (
                    obj.name or "plane",
                    tuple(size),
                    tuple(plane_center),
                    tuple(obj.orientation_rpy or [0.0, 0.0, 0.0]),
                )
            )

    world_allowed_pairs: List[Tuple[str, str]] = []
    if robot_config.control and robot_config.control.world_allowed_pairs:
        def _norm_side(side: str) -> str:
            side = str(side)
            if side.startswith("obj:"):
                return side
            if side.startswith("robot:"):
                side = side.split(":")[-1]
            return side.split("/")[-1]

        for a, b in robot_config.control.world_allowed_pairs:
            world_allowed_pairs.append((_norm_side(a), _norm_side(b)))

    allowed_link_pairs = None
    if not ctrl.self_collision_check:
        allowed_link_pairs = []
        
    # Note: if robot has tool, the URDF should already be the combined one
    # from controller initialization, so no need to add tool geometries separately

    try:
        return CollisionChecker(
            robot_config.urdf,
            logger,
            base_position=tuple(robot_config.base_position or (0.0, 0.0, 0.0)),
            base_orientation_rpy=tuple(robot_config.base_orientation or (0.0, 0.0, 0.0)),
            allowed_link_pairs=allowed_link_pairs,
            world_allowed_pairs=world_allowed_pairs,
            world_boxes=world_boxes,
            ground_plane_z=ctrl.ground_plane_z,
            collision_mesh_shrink=getattr(ctrl, "collision_mesh_shrink", 1.0),
        )
    except Exception as exc:
        logger.warning(f"Collision checker init failed for {robot_config.name}: {exc}")
        return None


def register_env_robots(robots: Dict[str, "RobotRuntime"], logger) -> None:
    """Register every robot as an environment obstacle in others."""
    for runtime_a in robots.values():
        checker = runtime_a.collision_checker
        if not checker:
            continue
        for runtime_b in robots.values():
            if runtime_a.name == runtime_b.name:
                continue
            try:
                # Use the actual URDF that was used for collision checking
                # (which might be the combined URDF with tool if tool is attached)
                urdf_to_register = runtime_b.config.urdf
                # Check if runtime_b has a combined URDF with tool
                from pathlib import Path
                combined_path = Path(runtime_b.config.urdf).parent / f"{Path(runtime_b.config.urdf).stem}_with_tool.urdf"
                if runtime_b.config.tool and combined_path.exists():
                    urdf_to_register = str(combined_path)
                    logger.info(f"Using combined URDF for env robot {runtime_b.name}: {combined_path}")
                    
                checker.register_env_robot(runtime_b.name, urdf_to_register)
                logger.info(
                    f"Registered env robot '{runtime_b.name}' (urdf: {urdf_to_register}) into checker for '{runtime_a.name}'"
                )
            except Exception as exc:
                logger.debug(
                    f"Register env robot '{runtime_b.name}' into '{runtime_a.name}' checker failed: {exc}"
                )


def log_collision_status(robots: Dict[str, "RobotRuntime"], logger) -> None:
    for runtime in robots.values():
        checker = runtime.collision_checker
        if checker and checker.available:
            env_robots = len(checker.env_robot_geoms)
            world_objs = len(checker.env_objs)
            robot_links = len(checker.robot_geoms)
            logger.info(
                f"Collision checking ON for {runtime.name}: "
                f"{robot_links} links, {world_objs} world objects, {env_robots} env robots"
            )
        else:
            logger.warning(f"Collision checking OFF for {runtime.name}")


def make_state_valid_fn(
    runtime: "RobotRuntime",
    robots: Dict[str, "RobotRuntime"],
    logger,
    state_getter: Callable[[str], Optional[np.ndarray]],
) -> Callable[[np.ndarray], bool]:
    """Return a collision validity checker for ``runtime``."""
    checker = runtime.collision_checker
    mdl = runtime.pin_model
    dat = runtime.pin_data
    if not (checker and mdl and dat):
        logger.warning(
            f"Collision checking disabled for {runtime.name} - missing components"
        )
        return lambda q: True

    others: List[Tuple[str, Any, Any, Tuple[float, float, float], Tuple[float, float, float]]] = []
    for other in robots.values():
        if other.name == runtime.name:
            continue
        mdl_o = other.pin_model
        dat_o = other.pin_data
        if not (mdl_o and dat_o):
            logger.error(
                f"Missing Pinocchio model/data for env robot {other.name} in {runtime.name}'s checker"
            )
            continue
        base_pos = tuple(other.config.base_position or (0.0, 0.0, 0.0))
        base_rpy = tuple(other.config.base_orientation or (0.0, 0.0, 0.0))
        others.append((other.name, mdl_o, dat_o, base_pos, base_rpy))

    def _valid(q: np.ndarray) -> bool:
        for name_o, mdl_o, dat_o, base_pos, base_rpy in others:
            q_o = state_getter(name_o)
            if q_o is None or q_o.size == 0:
                fallback = robots[name_o].joint_targets
                if fallback:
                    q_o = np.array([np.deg2rad(d) for d in fallback], dtype=np.float64)
                    logger.warning(
                        f"Using {name_o} targets instead of Genesis state: {fallback} deg"
                    )
                else:
                    logger.error(
                        f"No state available for env robot {name_o} - collision checking incomplete"
                    )
                    continue
            q_o = np.asarray(q_o, dtype=np.float64).flatten()
            if q_o.size != mdl_o.nq:
                padded = np.zeros(mdl_o.nq, dtype=np.float64)
                padded[: min(mdl_o.nq, q_o.size)] = q_o[: min(mdl_o.nq, q_o.size)]
                q_o = padded
            try:
                checker.update_env_robot_from_pin(
                    name_o,
                    mdl_o,
                    dat_o,
                    q_o,
                    base_position=base_pos,
                    base_orientation_rpy=base_rpy,
                )
                # logger.debug(f"Updated env robot {name_o} at base {base_pos} with q={q_o[:3] if len(q_o) > 3 else q_o}")
            except Exception as exc:
                logger.debug(f"Env robot update failed for {name_o}: {exc}")

        q = np.asarray(q, dtype=np.float64).flatten()
        if q.size != mdl.nq:
            padded = np.zeros(mdl.nq, dtype=np.float64)
            padded[: min(mdl.nq, q.size)] = q[: min(mdl.nq, q.size)]
            q_use = padded
        else:
            q_use = q
        try:
            in_collision = checker.in_collision_from_pin(mdl, dat, q_use)
            if in_collision:
                logger.debug(f"[{runtime.name}] State in collision at q={q_use[:3] if len(q_use) > 3 else q_use}")
            return not in_collision
        except Exception as exc:
            logger.warning(f"Collision check error for {runtime.name}: {exc}")
            return False

    return _valid


__all__ = [
    "create_collision_checker",
    "register_env_robots",
    "log_collision_status",
    "make_state_valid_fn",
]
