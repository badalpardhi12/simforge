"""Scene construction helpers for Genesis."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Tuple, TYPE_CHECKING

from ...core.models import EnvironmentSpec, RobotProfile
from .io import set_joint_positions

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .client import GenesisClient


def _quat_to_euler_deg(quat: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    w, x, y, z = quat
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


@dataclass
class SceneBuildResult:
    scene: object
    robot_entities: Dict[str, object]
    viewer_active: bool = True


def _check_opengl_available() -> bool:
    """Check if OpenGL 3+ context can be initialized."""
    import os
    # Quick heuristic checks
    # If DISPLAY is not set and not using EGL, viewer likely won't work
    display = os.environ.get("DISPLAY")
    egl_platform = os.environ.get("PYOPENGL_PLATFORM", "").lower()
    
    if not display and egl_platform != "egl":
        return False
    return True


def build_scene(client: "GenesisClient", spec: EnvironmentSpec, logger: logging.Logger) -> SceneBuildResult:
    """Build a Genesis scene with automatic fallback to headless mode if viewer fails."""
    scene_cfg = spec.scene
    viewer_requested = scene_cfg.viewer.enabled
    
    # Check if viewer is likely to work
    if viewer_requested and not _check_opengl_available():
        logger.warning(
            "Display environment not detected. Starting in headless mode. "
            "Set DISPLAY or PYOPENGL_PLATFORM=egl for viewer support."
        )
        viewer_requested = False
    
    # First attempt: build with requested viewer setting
    scene, robot_entities, viewer_active, build_ok = _attempt_scene_build(
        client, spec, logger, show_viewer=viewer_requested
    )
    
    # If viewer was requested but failed, we need to restart Genesis and retry
    if viewer_requested and not build_ok:
        logger.info("Viewer initialization failed. Restarting Genesis in headless mode...")
        try:
            # Destroy current Genesis state
            client.destroy()
            # Re-initialize Genesis
            client._initialise_backend()
            # Rebuild scene without viewer
            scene, robot_entities, viewer_active, build_ok = _attempt_scene_build(
                client, spec, logger, show_viewer=False
            )
            if build_ok:
                logger.info("Scene built successfully in headless mode")
            else:
                logger.error("Failed to build scene even in headless mode")
        except Exception as exc:
            logger.error("Failed to rebuild scene in headless mode: %s", exc)
    
    return SceneBuildResult(scene=scene, robot_entities=robot_entities, viewer_active=viewer_active)


def _attempt_scene_build(
    client: "GenesisClient",
    spec: EnvironmentSpec,
    logger: logging.Logger,
    show_viewer: bool,
) -> Tuple[object, Dict[str, object], bool, bool]:
    """Attempt to build a scene with the specified viewer setting.
    
    Returns:
        Tuple of (scene, robot_entities, viewer_active, build_succeeded)
    """
    scene_cfg = spec.scene
    scene, _ = client.create_scene(
        dt=scene_cfg.dt,
        gravity=scene_cfg.gravity,
        show_viewer=show_viewer,
        max_fps=scene_cfg.viewer.max_fps,
    )

    _build_world(scene, client, spec, logger)

    robot_entities: Dict[str, object] = {}
    for robot in spec.robots:
        entity = _spawn_robot(scene, client, robot, logger)
        robot_entities[robot.name] = entity

    viewer_active = show_viewer
    build_succeeded = True
    
    try:
        scene.build()
    except Exception as exc:  # pragma: no cover - genesis internals
        exc_msg = str(exc).lower()
        if "opengl" in exc_msg or "viewer" in exc_msg or "unable to initialize" in exc_msg:
            logger.warning(
                "Genesis viewer initialization failed: %s. "
                "To fix: ensure NVIDIA drivers are active "
                "(try: sudo prime-select nvidia, or set PYOPENGL_PLATFORM=egl)",
                exc
            )
            viewer_active = False
            build_succeeded = False
        else:
            logger.warning("Genesis scene build reported: %s", exc)

    # Apply initial joint positions AFTER scene.build()
    for robot in spec.robots:
        entity = robot_entities[robot.name]
        _apply_initial_joints(robot, entity, logger)

    # Try initial steps only if build succeeded
    if build_succeeded:
        for _ in range(2):
            try:
                scene.step()
            except Exception as exc:  # pragma: no cover
                exc_msg = str(exc).lower()
                if "viewer closed" in exc_msg or "viewer" in exc_msg:
                    logger.debug("Viewer not available during initial step: %s", exc)
                    viewer_active = False
                    build_succeeded = False
                else:
                    logger.debug("Initial scene step failed: %s", exc)
                break

    return scene, robot_entities, viewer_active, build_succeeded


def _build_world(scene, client: "GenesisClient", spec: EnvironmentSpec, logger: logging.Logger) -> None:
    gs = client.gs
    for obj in spec.world.objects:
        if obj.type.value == "plane":
            scene.add_entity(
                gs.morphs.Plane(
                    pos=obj.pose_position,
                )
            )
        elif obj.type.value == "box" and obj.size:
            scene.add_entity(
                gs.morphs.Box(
                    pos=obj.pose_position,
                    size=obj.size,
                    euler=obj.pose_orientation_rpy,
                    fixed=not obj.dynamic,
                    is_free=obj.dynamic,
                )
            )
        elif obj.type.value == "sphere":
            radius = obj.radius if obj.radius is not None else (obj.size[0] if obj.size else 0.05)
            scene.add_entity(
                gs.morphs.Sphere(
                    pos=obj.pose_position,
                    radius=radius,
                    fixed=not obj.dynamic,
                    is_free=obj.dynamic,
                )
            )
        elif obj.type.value == "urdf" and obj.urdf:
            scene.add_entity(
                gs.morphs.URDF(
                    file=obj.urdf,
                    pos=obj.pose_position,
                    euler=obj.pose_orientation_rpy,
                    fixed=not obj.dynamic,
                )
            )
        else:
            logger.debug("Skipping unsupported world object %s", obj)


def _spawn_robot(scene, client: "GenesisClient", robot: RobotProfile, logger: logging.Logger):
    gs = client.gs
    position = robot.mount.position
    euler = _quat_to_euler_deg(robot.mount.orientation)
    entity = scene.add_entity(
        gs.morphs.URDF(
            file=robot.urdf,
            pos=position,
            euler=euler,
            fixed=robot.fixed_base,
            requires_jac_and_IK=True,
        )
    )
    logger.info("Spawned robot %s from %s", robot.name, robot.urdf)
    return entity


def _apply_initial_joints(robot: RobotProfile, entity, logger: logging.Logger) -> None:
    joints = robot.initial_joint_positions_deg
    if not joints:
        return
    joints_rad = tuple(math.radians(float(v)) for v in joints)
    set_joint_positions(entity, joints_rad, len(joints_rad), logger)


__all__ = ["SceneBuildResult", "build_scene"]
