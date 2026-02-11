"""
Start Robots Launch File

Main launch file for starting both UR5e robots with controllers.
Based on Universal Robots ROS2 Dual Robot Tutorial.

For real robot operation in headless mode, the following controllers are
essential for each robot:
  - io_and_status_controller (GPIOController) — exposes the
    ``resend_robot_program`` service so the URScript program can be
    re-sent to the robot when needed.
  - speed_scaling_state_broadcaster — publishes the speed scaling factor
    that the ``scaled_joint_trajectory_controller`` reads to adjust
    trajectory timing.
  - force_torque_sensor_broadcaster — publishes FT sensor data.

In addition, the ``controller_stopper_node`` (one per robot) monitors the
robot program state and automatically re-sends the URScript program in
headless mode when the connection is lost.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    # Get launch configuration values
    nakul_robot_ip = LaunchConfiguration("nakul_robot_ip")
    sahadev_robot_ip = LaunchConfiguration("sahadev_robot_ip")
    
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    headless_mode = LaunchConfiguration("headless_mode")
    launch_rviz = LaunchConfiguration("launch_rviz")

    # Resolve use_fake_hardware to decide whether to spawn UR-specific
    # controllers and nodes (only needed for real hardware).
    use_fake_hw_str = use_fake_hardware.perform(context)
    is_real_hardware = use_fake_hw_str.lower() in ("false", "0", "no")

    combined_controller_config = PathJoinSubstitution(
        [
            FindPackageShare("valid8_dual_cell_control"),
            "config",
            "combined_controllers.yaml",
        ]
    )
    update_rate_config = PathJoinSubstitution(
        [
            FindPackageShare("valid8_dual_cell_control"),
            "config",
            "update_rate.yaml",
        ]
    )

    # Robot State Publisher
    rsp_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("valid8_dual_cell_control"),
                    "launch",
                    "rsp.launch.py",
                ]
            )
        ),
        launch_arguments={
            "nakul_robot_ip": nakul_robot_ip,
            "sahadev_robot_ip": sahadev_robot_ip,
            "nakul_use_fake_hardware": use_fake_hardware,
            "sahadev_use_fake_hardware": use_fake_hardware,
            "nakul_fake_sensor_commands": "false",
            "sahadev_fake_sensor_commands": "false",
            "headless_mode": headless_mode,
        }.items(),
    )

    # Controller Manager Node
    controller_manager_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            combined_controller_config,
            update_rate_config,
        ],
        output="both",
        remappings=[
            ("~/robot_description", "/robot_description"),
        ],
    )

    # Joint State Broadcaster Spawner
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "-c", "/controller_manager",
            "--controller-manager-timeout", "120",
        ],
        output="screen",
    )

    # Nakul Scaled Joint Trajectory Controller
    nakul_scaled_jtc_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "nakul_scaled_joint_trajectory_controller",
            "-c", "/controller_manager",
            "--controller-manager-timeout", "120",
        ],
        output="screen",
    )

    # Sahadev Scaled Joint Trajectory Controller
    sahadev_scaled_jtc_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "sahadev_scaled_joint_trajectory_controller",
            "-c", "/controller_manager",
            "--controller-manager-timeout", "120",
        ],
        output="screen",
    )

    # RViz (only if display available)
    rviz_config = PathJoinSubstitution(
        [
            FindPackageShare("valid8_dual_cell_description"),
            "rviz",
            "urdf.rviz",
        ]
    )
    
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        condition=IfCondition(launch_rviz),
    )

    nodes_to_start = [
        rsp_launch,
        controller_manager_node,
        joint_state_broadcaster_spawner,
        nakul_scaled_jtc_spawner,
        sahadev_scaled_jtc_spawner,
        rviz_node,
    ]

    # ── Real-hardware-only controllers and nodes ──────────────────
    # These are required for the UR driver to actually move the robot.
    # The io_and_status_controller provides the resend_robot_program
    # service, and the controller_stopper_node uses it to automatically
    # re-send the URScript program in headless mode.
    if is_real_hardware:
        # Nakul IO and Status Controller (GPIOController)
        nakul_io_status_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "nakul_io_and_status_controller",
                "-c", "/controller_manager",
                "--controller-manager-timeout", "120",
            ],
            output="screen",
        )

        # Sahadev IO and Status Controller (GPIOController)
        sahadev_io_status_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "sahadev_io_and_status_controller",
                "-c", "/controller_manager",
                "--controller-manager-timeout", "120",
            ],
            output="screen",
        )

        # Nakul Speed Scaling State Broadcaster
        nakul_speed_scaling_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "nakul_speed_scaling_state_broadcaster",
                "-c", "/controller_manager",
                "--controller-manager-timeout", "120",
            ],
            output="screen",
        )

        # Sahadev Speed Scaling State Broadcaster
        sahadev_speed_scaling_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "sahadev_speed_scaling_state_broadcaster",
                "-c", "/controller_manager",
                "--controller-manager-timeout", "120",
            ],
            output="screen",
        )

        # Nakul Force Torque Sensor Broadcaster
        nakul_fts_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "nakul_force_torque_sensor_broadcaster",
                "-c", "/controller_manager",
                "--controller-manager-timeout", "120",
            ],
            output="screen",
        )

        # Sahadev Force Torque Sensor Broadcaster
        sahadev_fts_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "sahadev_force_torque_sensor_broadcaster",
                "-c", "/controller_manager",
                "--controller-manager-timeout", "120",
            ],
            output="screen",
        )

        # Controller Stopper for Nakul — monitors robot program state
        # and automatically resends the URScript in headless mode.
        #
        # CRITICAL: In a dual-robot setup with a shared controller_manager,
        # the controller_stopper calls list_controllers on the GLOBAL
        # controller_manager and deactivates everything NOT in
        # consistent_controllers.  We MUST include the other robot's
        # controllers here so that when nakul's program drops, it does
        # NOT deactivate sahadev's controllers (and vice-versa).
        nakul_controller_stopper = Node(
            package="ur_robot_driver",
            executable="controller_stopper_node",
            name="nakul_controller_stopper",
            output="screen",
            emulate_tty=True,
            parameters=[
                {"headless_mode": headless_mode},
                {"joint_controller_active": True},
                {
                    "consistent_controllers": [
                        # Nakul's own always-on controllers
                        "nakul_io_and_status_controller",
                        "nakul_force_torque_sensor_broadcaster",
                        "nakul_speed_scaling_state_broadcaster",
                        # Sahadev's controllers — MUST be protected
                        # so nakul's stopper doesn't deactivate them
                        "sahadev_io_and_status_controller",
                        "sahadev_force_torque_sensor_broadcaster",
                        "sahadev_speed_scaling_state_broadcaster",
                        "sahadev_scaled_joint_trajectory_controller",
                        # Shared controllers
                        "joint_state_broadcaster",
                    ],
                },
                {
                    "controller_names": [
                        "nakul_scaled_joint_trajectory_controller",
                    ],
                },
                {"tf_prefix": "nakul_"},
            ],
            remappings=[
                ("io_and_status_controller/robot_program_running", "nakul_io_and_status_controller/robot_program_running"),
            ],
        )

        # Controller Stopper for Sahadev
        # (see nakul_controller_stopper comment for why we include
        # the other robot's controllers in consistent_controllers)
        sahadev_controller_stopper = Node(
            package="ur_robot_driver",
            executable="controller_stopper_node",
            name="sahadev_controller_stopper",
            output="screen",
            emulate_tty=True,
            parameters=[
                {"headless_mode": headless_mode},
                {"joint_controller_active": True},
                {
                    "consistent_controllers": [
                        # Sahadev's own always-on controllers
                        "sahadev_io_and_status_controller",
                        "sahadev_force_torque_sensor_broadcaster",
                        "sahadev_speed_scaling_state_broadcaster",
                        # Nakul's controllers — MUST be protected
                        # so sahadev's stopper doesn't deactivate them
                        "nakul_io_and_status_controller",
                        "nakul_force_torque_sensor_broadcaster",
                        "nakul_speed_scaling_state_broadcaster",
                        "nakul_scaled_joint_trajectory_controller",
                        # Shared controllers
                        "joint_state_broadcaster",
                    ],
                },
                {
                    "controller_names": [
                        "sahadev_scaled_joint_trajectory_controller",
                    ],
                },
                {"tf_prefix": "sahadev_"},
            ],
            remappings=[
                ("io_and_status_controller/robot_program_running", "sahadev_io_and_status_controller/robot_program_running"),
            ],
        )

        # URScript Interface nodes — allow sending URScript snippets
        nakul_urscript_interface = Node(
            package="ur_robot_driver",
            executable="urscript_interface",
            name="nakul_urscript_interface",
            parameters=[{"robot_ip": nakul_robot_ip}],
            output="screen",
        )

        sahadev_urscript_interface = Node(
            package="ur_robot_driver",
            executable="urscript_interface",
            name="sahadev_urscript_interface",
            parameters=[{"robot_ip": sahadev_robot_ip}],
            output="screen",
        )

        nodes_to_start.extend([
            nakul_io_status_spawner,
            sahadev_io_status_spawner,
            nakul_speed_scaling_spawner,
            sahadev_speed_scaling_spawner,
            nakul_fts_spawner,
            sahadev_fts_spawner,
            nakul_controller_stopper,
            sahadev_controller_stopper,
            nakul_urscript_interface,
            sahadev_urscript_interface,
        ])

    return nodes_to_start


def generate_launch_description():
    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            "nakul_robot_ip",
            default_value="192.168.1.9",
            description="IP address of nakul UR5e robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "sahadev_robot_ip",
            default_value="192.168.1.16",
            description="IP address of sahadev UR5e robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description="Start robots with fake/simulated hardware.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "headless_mode",
            default_value="false",
            description="Enable headless mode for robot control.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="false",
            description="Launch RViz for visualization.",
        )
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
