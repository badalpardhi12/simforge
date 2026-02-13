"""
Start Robot Launch File — Face Robot UR20

Main launch file for a single UR20 with controllers.
Supports both simulated (mock) and real hardware modes.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    headless_mode = LaunchConfiguration("headless_mode")
    launch_rviz = LaunchConfiguration("launch_rviz")

    use_fake_hw_str = use_fake_hardware.perform(context)
    is_real_hardware = use_fake_hw_str.lower() in ("false", "0", "no")

    controller_config = PathJoinSubstitution(
        [FindPackageShare("face_robot_ur20_control"), "config", "controllers.yaml"]
    )
    update_rate_config = PathJoinSubstitution(
        [FindPackageShare("face_robot_ur20_control"), "config", "update_rate.yaml"]
    )

    # Robot State Publisher
    rsp_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("face_robot_ur20_control"), "launch", "rsp.launch.py"]
            )
        ),
        launch_arguments={
            "robot_ip": robot_ip,
            "use_fake_hardware": use_fake_hardware,
            "fake_sensor_commands": "false",
            "headless_mode": headless_mode,
        }.items(),
    )

    # Controller Manager
    controller_manager_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[controller_config, update_rate_config],
        output="both",
        remappings=[("~/robot_description", "/robot_description")],
    )

    # Joint State Broadcaster
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

    # Scaled Joint Trajectory Controller
    scaled_jtc_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "scaled_joint_trajectory_controller",
            "-c", "/controller_manager",
            "--controller-manager-timeout", "120",
        ],
        output="screen",
    )

    nodes_to_start = [
        rsp_launch,
        controller_manager_node,
        joint_state_broadcaster_spawner,
        scaled_jtc_spawner,
    ]

    # Real-hardware-only controllers
    if is_real_hardware:
        io_status_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "io_and_status_controller",
                "-c", "/controller_manager",
                "--controller-manager-timeout", "120",
            ],
            output="screen",
        )

        speed_scaling_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "speed_scaling_state_broadcaster",
                "-c", "/controller_manager",
                "--controller-manager-timeout", "120",
            ],
            output="screen",
        )

        fts_spawner = Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "force_torque_sensor_broadcaster",
                "-c", "/controller_manager",
                "--controller-manager-timeout", "120",
            ],
            output="screen",
        )

        controller_stopper = Node(
            package="ur_robot_driver",
            executable="controller_stopper_node",
            name="controller_stopper",
            output="screen",
            emulate_tty=True,
            parameters=[
                {"headless_mode": headless_mode},
                {"joint_controller_active": True},
                {
                    "consistent_controllers": [
                        "io_and_status_controller",
                        "force_torque_sensor_broadcaster",
                        "speed_scaling_state_broadcaster",
                        "joint_state_broadcaster",
                    ],
                },
                {
                    "controller_names": [
                        "scaled_joint_trajectory_controller",
                    ],
                },
                {"tf_prefix": ""},
            ],
            remappings=[
                ("io_and_status_controller/robot_program_running",
                 "io_and_status_controller/robot_program_running"),
            ],
        )

        urscript_interface = Node(
            package="ur_robot_driver",
            executable="urscript_interface",
            name="urscript_interface",
            parameters=[{"robot_ip": robot_ip}],
            output="screen",
        )

        nodes_to_start.extend([
            io_status_spawner,
            speed_scaling_spawner,
            fts_spawner,
            controller_stopper,
            urscript_interface,
        ])

    return nodes_to_start


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("robot_ip", default_value="0.0.0.0",
                              description="IP address of UR20 robot."),
        DeclareLaunchArgument("use_fake_hardware", default_value="false",
                              description="Start with fake/simulated hardware."),
        DeclareLaunchArgument("headless_mode", default_value="false",
                              description="Enable headless mode."),
        DeclareLaunchArgument("launch_rviz", default_value="false",
                              description="Launch RViz."),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
