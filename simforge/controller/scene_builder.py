"""Scene construction helpers."""
from __future__ import annotations

from typing import Dict, TYPE_CHECKING

from .utils import deg_to_rad_list
from .robot_io import set_robot_joints

if TYPE_CHECKING:
    from .robot_runtime import RobotRuntime


def build_scene(renderer, config, runtimes: Dict[str, "RobotRuntime"], logger):
    """Create the Genesis scene and populate robots and world objects."""
    scene_cfg = config.scene
    scene = renderer.create_scene(
        dt=scene_cfg.dt,
        gravity=scene_cfg.gravity,
        show_viewer=scene_cfg.show_viewer,
        max_fps=scene_cfg.max_fps,
    )

    # World primitives
    for obj in config.objects:
        if obj.type == "plane":
            scene.add_entity(renderer.morphs.Plane(pos=obj.position))
        elif obj.type == "box" and obj.size:
            scene.add_entity(
                renderer.morphs.Box(
                    pos=obj.position,
                    size=obj.size,
                    euler=obj.orientation_rpy,
                    is_free=obj.dynamic,
                    fixed=not obj.dynamic,
                )
            )
        elif obj.type == "sphere":
            radius = float(obj.size[0]) if obj.size else 0.05
            scene.add_entity(
                renderer.morphs.Sphere(
                    pos=obj.position,
                    radius=radius,
                    is_free=obj.dynamic,
                    fixed=not obj.dynamic,
                )
            )

    # Robots
    for robot in config.robots:
        entity = scene.add_entity(
            renderer.morphs.URDF(
                file=robot.urdf,
                pos=robot.base_position,
                euler=robot.base_orientation,
                fixed=robot.fixed_base,
            )
        )
        runtimes[robot.name].entity = entity

    scene.build()

    # Apply initial joint targets if provided
    for robot in config.robots:
        runtime = runtimes[robot.name]
        if robot.initial_joint_positions:
            q_rad = deg_to_rad_list(robot.initial_joint_positions)
            expected = len(robot.initial_joint_positions)
            set_robot_joints(runtime.entity, q_rad, expected, logger)
            runtime.joint_targets = list(robot.initial_joint_positions)

    if scene:
        for _ in range(2):
            try:
                scene.step()
            except Exception as exc:
                logger.debug(f"Initial scene step failed: {exc}")

    logger.info("Scene built successfully")
    return scene


__all__ = ["build_scene"]
