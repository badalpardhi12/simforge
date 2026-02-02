#!/usr/bin/env python3
"""
Robot Control Node

Provides ROS 2 Action Servers for robot motion control.
Wraps ur_rtde library for communication with UR robots.

This node:
1. Connects to UR robot via RTDE protocol
2. Exposes MoveJ, MoveL, and MoveRobot actions
3. Publishes robot state at 50Hz
4. Handles protective/emergency stops
"""

import time
import threading
from typing import Optional, List
from dataclasses import dataclass
from enum import IntEnum

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import String
from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose, Wrench, Twist
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

try:
    import rtde_control
    import rtde_receive
    HAS_RTDE = True
except ImportError:
    HAS_RTDE = False
    print("Warning: ur_rtde not installed. Using simulation mode.")


class RobotMode(IntEnum):
    """Robot operational mode."""
    DISCONNECTED = 0
    IDLE = 1
    RUNNING = 2
    FREEDRIVE = 3
    ERROR = 4


@dataclass
class RobotConfig:
    """Robot configuration."""
    robot_ip: str = "192.168.1.9"
    robot_name: str = "ur20"
    control_frequency: float = 500.0  # Hz
    state_publish_rate: float = 50.0  # Hz
    max_velocity: float = 1.0  # rad/s
    max_acceleration: float = 1.0  # rad/s^2
    default_velocity_scale: float = 0.5
    default_acceleration_scale: float = 0.5


class RobotControlNode(Node):
    """
    Robot Control Node - interfaces with UR robot via RTDE.
    """

    def __init__(self):
        super().__init__('robot_control')
        
        # Declare parameters
        self.declare_parameter('robot_ip', '192.168.1.9')
        self.declare_parameter('robot_name', 'ur20')
        self.declare_parameter('control_frequency', 500.0)
        self.declare_parameter('state_publish_rate', 50.0)
        self.declare_parameter('simulation_mode', not HAS_RTDE)
        
        # Load configuration
        self.config = RobotConfig(
            robot_ip=self.get_parameter('robot_ip').value,
            robot_name=self.get_parameter('robot_name').value,
            control_frequency=self.get_parameter('control_frequency').value,
            state_publish_rate=self.get_parameter('state_publish_rate').value,
        )
        
        self.simulation_mode = self.get_parameter('simulation_mode').value
        
        # Robot state
        self.current_joint_positions = [0.0] * 6
        self.current_joint_velocities = [0.0] * 6
        self.current_joint_torques = [0.0] * 6
        self.current_tcp_pose = Pose()
        self.current_tcp_wrench = Wrench()
        self.robot_mode = RobotMode.DISCONNECTED
        self.protective_stop = False
        self.emergency_stop = False
        
        # RTDE interfaces (if available)
        self.rtde_control: Optional[rtde_control.RTDEControlInterface] = None
        self.rtde_receive: Optional[rtde_receive.RTDEReceiveInterface] = None
        
        # Callback group for async operations
        self.callback_group = ReentrantCallbackGroup()
        
        # === Publishers ===
        # Joint state (standard ROS message)
        self.joint_state_pub = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )
        
        # Full robot state (custom message - using String as placeholder)
        self.robot_state_pub = self.create_publisher(
            String,
            f'/{self.config.robot_name}/robot_state',
            10
        )
        
        # TCP wrench
        self.wrench_pub = self.create_publisher(
            Wrench,
            '/robot/tcp_wrench',
            10
        )
        
        # === Services ===
        # Protective stop
        self.protective_stop_srv = self.create_service(
            Trigger,
            '/robot/protective_stop',
            self.protective_stop_callback,
            callback_group=self.callback_group
        )
        
        # Emergency stop
        self.emergency_stop_srv = self.create_service(
            Trigger,
            '/robot/emergency_stop',
            self.emergency_stop_callback,
            callback_group=self.callback_group
        )
        
        # Reset (unlock brakes after stop)
        self.reset_srv = self.create_service(
            Trigger,
            '/robot/reset',
            self.reset_callback,
            callback_group=self.callback_group
        )
        
        # Freedrive mode
        self.freedrive_srv = self.create_service(
            Trigger,
            '/robot/freedrive',
            self.freedrive_callback,
            callback_group=self.callback_group
        )
        
        # === Timers ===
        # State publishing timer
        state_period = 1.0 / self.config.state_publish_rate
        self.state_timer = self.create_timer(state_period, self.publish_state)
        
        # Connect to robot
        if not self.simulation_mode:
            self.connect_to_robot()
        else:
            self.get_logger().warn("Running in SIMULATION MODE - no real robot")
            self.robot_mode = RobotMode.IDLE
            # Initialize simulated position
            self.current_joint_positions = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        
        self.get_logger().info(
            f"Robot Control Node initialized for {self.config.robot_name} "
            f"at {self.config.robot_ip}"
        )

    def connect_to_robot(self) -> bool:
        """Connect to the UR robot via RTDE."""
        if not HAS_RTDE:
            self.get_logger().error("ur_rtde not available")
            return False
        
        try:
            self.get_logger().info(f"Connecting to robot at {self.config.robot_ip}...")
            
            # Connect receive interface first
            self.rtde_receive = rtde_receive.RTDEReceiveInterface(
                self.config.robot_ip
            )
            
            # Then control interface
            self.rtde_control = rtde_control.RTDEControlInterface(
                self.config.robot_ip
            )
            
            self.robot_mode = RobotMode.IDLE
            self.get_logger().info("Connected to robot successfully")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Failed to connect to robot: {e}")
            self.robot_mode = RobotMode.DISCONNECTED
            return False

    def disconnect_from_robot(self):
        """Disconnect from the robot."""
        if self.rtde_control:
            try:
                self.rtde_control.stopScript()
            except:
                pass
            self.rtde_control = None
        
        if self.rtde_receive:
            self.rtde_receive = None
        
        self.robot_mode = RobotMode.DISCONNECTED
        self.get_logger().info("Disconnected from robot")

    def update_robot_state(self):
        """Update robot state from RTDE."""
        if self.simulation_mode:
            return
        
        if not self.rtde_receive:
            return
        
        try:
            # Get joint positions
            self.current_joint_positions = list(
                self.rtde_receive.getActualQ()
            )
            
            # Get joint velocities
            self.current_joint_velocities = list(
                self.rtde_receive.getActualQd()
            )
            
            # Get TCP pose
            tcp = self.rtde_receive.getActualTCPPose()
            self.current_tcp_pose.position.x = tcp[0]
            self.current_tcp_pose.position.y = tcp[1]
            self.current_tcp_pose.position.z = tcp[2]
            # Convert rotation vector to quaternion (simplified)
            # In production, use proper conversion
            
            # Get TCP force
            force = self.rtde_receive.getActualTCPForce()
            self.current_tcp_wrench.force.x = force[0]
            self.current_tcp_wrench.force.y = force[1]
            self.current_tcp_wrench.force.z = force[2]
            self.current_tcp_wrench.torque.x = force[3]
            self.current_tcp_wrench.torque.y = force[4]
            self.current_tcp_wrench.torque.z = force[5]
            
            # Check robot mode
            safety_status = self.rtde_receive.getSafetyMode()
            if safety_status == 1:  # Normal
                self.robot_mode = RobotMode.IDLE
                self.protective_stop = False
                self.emergency_stop = False
            elif safety_status == 3:  # Protective stop
                self.robot_mode = RobotMode.ERROR
                self.protective_stop = True
            elif safety_status == 4:  # Emergency stop
                self.robot_mode = RobotMode.ERROR
                self.emergency_stop = True
                
        except Exception as e:
            self.get_logger().error(f"Error reading robot state: {e}")

    def publish_state(self):
        """Publish robot state at configured rate."""
        # Update state from robot
        self.update_robot_state()
        
        # Publish JointState
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = [
            f'{self.config.robot_name}_shoulder_pan_joint',
            f'{self.config.robot_name}_shoulder_lift_joint',
            f'{self.config.robot_name}_elbow_joint',
            f'{self.config.robot_name}_wrist_1_joint',
            f'{self.config.robot_name}_wrist_2_joint',
            f'{self.config.robot_name}_wrist_3_joint',
        ]
        joint_state.position = self.current_joint_positions
        joint_state.velocity = self.current_joint_velocities
        joint_state.effort = self.current_joint_torques
        self.joint_state_pub.publish(joint_state)
        
        # Publish TCP wrench
        self.wrench_pub.publish(self.current_tcp_wrench)
        
        # Publish full robot state (simplified - will use proper msg type)
        state_msg = String()
        state_msg.data = (
            f"mode:{self.robot_mode.value}|"
            f"pstop:{self.protective_stop}|"
            f"estop:{self.emergency_stop}|"
            f"joints:{','.join(f'{j:.4f}' for j in self.current_joint_positions)}"
        )
        self.robot_state_pub.publish(state_msg)

    def move_joints(
        self,
        target_joints: List[float],
        velocity: float,
        acceleration: float,
        asynchronous: bool = False
    ) -> bool:
        """
        Move robot to target joint positions.
        
        Args:
            target_joints: Target joint positions in radians
            velocity: Joint velocity in rad/s
            acceleration: Joint acceleration in rad/s^2
            asynchronous: If True, return immediately
            
        Returns:
            True if motion started/completed successfully
        """
        if self.protective_stop or self.emergency_stop:
            self.get_logger().error("Cannot move - robot in stop state")
            return False
        
        if self.simulation_mode:
            # Simulate motion
            self.get_logger().info(f"[SIM] Moving to joints: {target_joints}")
            self.robot_mode = RobotMode.RUNNING
            
            # Simulate gradual motion
            steps = 10
            for i in range(steps):
                alpha = (i + 1) / steps
                self.current_joint_positions = [
                    start + alpha * (end - start)
                    for start, end in zip(self.current_joint_positions, target_joints)
                ]
                time.sleep(0.1)
            
            self.robot_mode = RobotMode.IDLE
            return True
        
        if not self.rtde_control:
            self.get_logger().error("Not connected to robot")
            return False
        
        try:
            self.robot_mode = RobotMode.RUNNING
            
            if asynchronous:
                self.rtde_control.moveJ(target_joints, velocity, acceleration, True)
            else:
                self.rtde_control.moveJ(target_joints, velocity, acceleration)
            
            self.robot_mode = RobotMode.IDLE
            return True
            
        except Exception as e:
            self.get_logger().error(f"Motion failed: {e}")
            self.robot_mode = RobotMode.ERROR
            return False

    def move_linear(
        self,
        target_pose: List[float],
        velocity: float,
        acceleration: float,
        asynchronous: bool = False
    ) -> bool:
        """
        Move robot linearly to target TCP pose.
        
        Args:
            target_pose: Target pose [x, y, z, rx, ry, rz]
            velocity: Linear velocity in m/s
            acceleration: Linear acceleration in m/s^2
            asynchronous: If True, return immediately
            
        Returns:
            True if motion started/completed successfully
        """
        if self.protective_stop or self.emergency_stop:
            self.get_logger().error("Cannot move - robot in stop state")
            return False
        
        if self.simulation_mode:
            self.get_logger().info(f"[SIM] Linear move to: {target_pose}")
            self.robot_mode = RobotMode.RUNNING
            time.sleep(1.0)  # Simulate motion
            self.robot_mode = RobotMode.IDLE
            return True
        
        if not self.rtde_control:
            self.get_logger().error("Not connected to robot")
            return False
        
        try:
            self.robot_mode = RobotMode.RUNNING
            
            if asynchronous:
                self.rtde_control.moveL(target_pose, velocity, acceleration, True)
            else:
                self.rtde_control.moveL(target_pose, velocity, acceleration)
            
            self.robot_mode = RobotMode.IDLE
            return True
            
        except Exception as e:
            self.get_logger().error(f"Motion failed: {e}")
            self.robot_mode = RobotMode.ERROR
            return False

    def stop_motion(self, deceleration: float = 2.0):
        """Stop current motion with specified deceleration."""
        self.get_logger().warn(f"Stopping motion (decel={deceleration})")
        
        if self.simulation_mode:
            self.robot_mode = RobotMode.IDLE
            return
        
        if self.rtde_control:
            try:
                self.rtde_control.stopJ(deceleration)
            except Exception as e:
                self.get_logger().error(f"Stop failed: {e}")

    def protective_stop_callback(self, request, response):
        """Handle protective stop request."""
        self.get_logger().warn("PROTECTIVE STOP triggered")
        
        self.protective_stop = True
        self.stop_motion(deceleration=2.0)
        
        response.success = True
        response.message = "Protective stop activated"
        return response

    def emergency_stop_callback(self, request, response):
        """Handle emergency stop request."""
        self.get_logger().fatal("EMERGENCY STOP triggered")
        
        self.emergency_stop = True
        self.protective_stop = True
        
        # Immediate stop
        self.stop_motion(deceleration=10.0)
        
        # Disconnect from robot for safety
        if self.rtde_control:
            try:
                self.rtde_control.triggerProtectiveStop()
            except:
                pass
        
        response.success = True
        response.message = "Emergency stop activated - manual reset required"
        return response

    def reset_callback(self, request, response):
        """Handle reset request."""
        self.get_logger().info("Reset requested")
        
        if self.emergency_stop:
            response.success = False
            response.message = "E-Stop active - manual reset required on robot"
            return response
        
        self.protective_stop = False
        
        # Reconnect if needed
        if self.robot_mode == RobotMode.DISCONNECTED:
            if self.connect_to_robot():
                response.success = True
                response.message = "Robot reconnected and reset"
            else:
                response.success = False
                response.message = "Failed to reconnect to robot"
        else:
            # Unlock brakes
            if not self.simulation_mode and self.rtde_control:
                try:
                    self.rtde_control.unlockProtectiveStop()
                except Exception as e:
                    response.success = False
                    response.message = f"Failed to unlock: {e}"
                    return response
            
            self.robot_mode = RobotMode.IDLE
            response.success = True
            response.message = "Robot reset complete"
        
        return response

    def freedrive_callback(self, request, response):
        """Toggle freedrive mode."""
        if self.protective_stop or self.emergency_stop:
            response.success = False
            response.message = "Cannot enable freedrive - robot in stop state"
            return response
        
        if self.simulation_mode:
            if self.robot_mode == RobotMode.FREEDRIVE:
                self.robot_mode = RobotMode.IDLE
                response.message = "Freedrive disabled"
            else:
                self.robot_mode = RobotMode.FREEDRIVE
                response.message = "Freedrive enabled"
            response.success = True
            return response
        
        if not self.rtde_control:
            response.success = False
            response.message = "Not connected to robot"
            return response
        
        try:
            if self.robot_mode == RobotMode.FREEDRIVE:
                self.rtde_control.endFreedriveMode()
                self.robot_mode = RobotMode.IDLE
                response.message = "Freedrive disabled"
            else:
                self.rtde_control.freedriveMode()
                self.robot_mode = RobotMode.FREEDRIVE
                response.message = "Freedrive enabled"
            
            response.success = True
            
        except Exception as e:
            response.success = False
            response.message = f"Freedrive toggle failed: {e}"
        
        return response


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    node = RobotControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Robot Control Node...")
    finally:
        node.disconnect_from_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
