"""Scene construction helpers."""
from __future__ import annotations

from typing import Dict, TYPE_CHECKING, Optional
import numpy as np

from .utils import deg_to_rad_list
from .robot_io import set_robot_joints

if TYPE_CHECKING:
    from .robot_runtime import RobotRuntime
    from ..tooling import ToolManager


def build_scene(renderer, config, runtimes: Dict[str, "RobotRuntime"], logger, tool_manager: Optional["ToolManager"] = None):
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
        # Check if robot has a tool attached
        urdf_file = robot.urdf
        if robot.tool and tool_manager:
            # Create a combined URDF with tool attached
            logger.info(f"Attaching tool to {robot.name} at {robot.end_effector_link}")
            try:
                from ..tooling import ToolDefinition
                tool_def = ToolDefinition(
                    name=f"{robot.name}_tool",
                    urdf_path=robot.tool.urdf,
                    tcp_offset=robot.tool.tcp_offset,
                    attach_offset=(
                        robot.tool.position[0], robot.tool.position[1], robot.tool.position[2],
                        robot.tool.orientation_rpy[0], robot.tool.orientation_rpy[1], robot.tool.orientation_rpy[2]
                    ) if robot.tool else None
                )
                tool_manager.register_tool(tool_def)
                
                # Create combined URDF
                from pathlib import Path
                output_path = Path(robot.urdf).parent / f"{Path(robot.urdf).stem}_with_tool.urdf"
                urdf_file = tool_manager.attach_tool_to_robot_urdf(
                    robot.urdf,
                    tool_def.name,
                    robot.end_effector_link,
                    str(output_path)
                )
                logger.info(f"Using combined URDF: {urdf_file}")
                
                # Store tool info in runtime
                runtimes[robot.name].tool_name = tool_def.name
                runtimes[robot.name].tool_tcp_offset = tool_def.get_tcp_transform()
                
            except Exception as e:
                logger.error(f"Failed to attach tool to {robot.name}: {e}")
                # Fall back to original URDF
                urdf_file = robot.urdf
        
        entity = scene.add_entity(
            renderer.morphs.URDF(
                file=urdf_file,
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
