#!/usr/bin/env python3
"""
Robot Control Node (v2)

Provides ROS 2 interface for UR robot motion control.
Uses the new ur_communication module which avoids RTDE conflicts.

This node:
1. Connects to UR robot via Dashboard + Secondary/Realtime interfaces
2. Exposes ROS 2 services and publishers for robot control
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
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import String
from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose, Wrench
from sensor_msgs.msg import JointState

# Import our new UR communication module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ur_communication import URRobotController, RobotState, RobotMode, SafetyMode
    HAS_UR_COMM = True
except ImportError as e:
    print(f"Warning: ur_communication module not available: {e}")
    HAS_UR_COMM = False
    
    # Define fallback classes
    class RobotMode(IntEnum):
        DISCONNECTED = -1
        IDLE = 2
        RUNNING = 1
        ERROR = 4
    
    class SafetyMode(IntEnum):
        NORMAL = 1


@dataclass
class RobotConfig:
    """Robot configuration."""
    robot_ip: str = "192.168.1.9"
    robot_name: str = "ur5e"
    state_publish_rate: float = 50.0  # Hz
    max_velocity: float = 1.0  # rad/s
    max_acceleration: float = 1.0  # rad/s^2
    default_velocity_scale: float = 0.5
    default_acceleration_scale: float = 0.5


class RobotControlNodeV2(Node):
    """
    Robot Control Node v2 - uses Dashboard + Secondary/Realtime interfaces.
    
    Avoids RTDE "input registers in use" conflict with EtherNet/IP adapter.
    """

    def __init__(self):
        super().__init__('robot_control')
        
        # Declare parameters
        self.declare_parameter('robot_ip', '192.168.1.9')
        self.declare_parameter('robot_name', 'ur5e')
        self.declare_parameter('state_publish_rate', 50.0)
        self.declare_parameter('simulation_mode', not HAS_UR_COMM)
        
        # Load configuration
        self.config = RobotConfig(
            robot_ip=self.get_parameter('robot_ip').value,
            robot_name=self.get_parameter('robot_name').value,
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
        
        # UR Robot Controller (new communication module)
        self.robot: Optional[URRobotController] = None
        
        # Callback group for async operations
        self.callback_group = ReentrantCallbackGroup()
        
        # === Publishers ===
        self.joint_state_pub = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )
        
        self.robot_state_pub = self.create_publisher(
            String,
            f'/{self.config.robot_name}/robot_state',
            10
        )
        
        self.wrench_pub = self.create_publisher(
            Wrench,
            '/robot/tcp_wrench',
            10
        )
        
        # === Services ===
        self.protective_stop_srv = self.create_service(
            Trigger,
            '/robot/protective_stop',
            self.protective_stop_callback,
            callback_group=self.callback_group
        )
        
        self.emergency_stop_srv = self.create_service(
            Trigger,
            '/robot/emergency_stop',
            self.emergency_stop_callback,
            callback_group=self.callback_group
        )
        
        self.reset_srv = self.create_service(
            Trigger,
            '/robot/reset',
            self.reset_callback,
            callback_group=self.callback_group
        )
        
        self.freedrive_srv = self.create_service(
            Trigger,
            '/robot/freedrive',
            self.freedrive_callback,
            callback_group=self.callback_group
        )
        
        self.power_on_srv = self.create_service(
            Trigger,
            '/robot/power_on',
            self.power_on_callback,
            callback_group=self.callback_group
        )
        
        self.power_off_srv = self.create_service(
            Trigger,
            '/robot/power_off',
            self.power_off_callback,
            callback_group=self.callback_group
        )
        
        # === Timers ===
        state_period = 1.0 / self.config.state_publish_rate
        self.state_timer = self.create_timer(state_period, self.publish_state)
        
        # Connect to robot
        if not self.simulation_mode:
            self.connect_to_robot()
        else:
            self.get_logger().warn("Running in SIMULATION MODE - no real robot")
            self.robot_mode = RobotMode.IDLE
            self.current_joint_positions = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        
        self.get_logger().info(
            f"Robot Control Node v2 initialized for {self.config.robot_name} "
            f"at {self.config.robot_ip}"
        )

    def connect_to_robot(self) -> bool:
        """Connect to the UR robot via Dashboard + Secondary/Realtime interfaces."""
        if not HAS_UR_COMM:
            self.get_logger().error("ur_communication module not available")
            return False
        
        try:
            self.get_logger().info(f"Connecting to robot at {self.config.robot_ip}...")
            
            self.robot = URRobotController(self.config.robot_ip)
            
            if self.robot.connect():
                self.robot_mode = RobotMode.IDLE
                self.get_logger().info("Connected to robot successfully (Dashboard + Realtime)")
                
                # Log robot mode
                mode = self.robot.get_robot_mode()
                safety = self.robot.get_safety_mode()
                self.get_logger().info(f"Robot mode: {mode}, Safety: {safety}")
                
                return True
            else:
                self.get_logger().error("Failed to connect to robot")
                self.robot_mode = RobotMode.DISCONNECTED
                return False
            
        except Exception as e:
            self.get_logger().error(f"Failed to connect to robot: {e}")
            self.robot_mode = RobotMode.DISCONNECTED
            return False

    def disconnect_from_robot(self):
        """Disconnect from the robot."""
        if self.robot:
            try:
                self.robot.disconnect()
            except:
                pass
            self.robot = None
        
        self.robot_mode = RobotMode.DISCONNECTED
        self.get_logger().info("Disconnected from robot")

    def update_robot_state(self):
        """Update robot state from realtime interface."""
        if self.simulation_mode or not self.robot:
            return
        
        try:
            state = self.robot.get_state()
            
            if state.connected:
                self.current_joint_positions = state.joint_positions
                self.current_joint_velocities = state.joint_velocities
                self.current_joint_torques = state.joint_currents
                
                # TCP force
                if state.tcp_force:
                    self.current_tcp_wrench.force.x = state.tcp_force[0] if len(state.tcp_force) > 0 else 0.0
                    self.current_tcp_wrench.force.y = state.tcp_force[1] if len(state.tcp_force) > 1 else 0.0
                    self.current_tcp_wrench.force.z = state.tcp_force[2] if len(state.tcp_force) > 2 else 0.0
                    self.current_tcp_wrench.torque.x = state.tcp_force[3] if len(state.tcp_force) > 3 else 0.0
                    self.current_tcp_wrench.torque.y = state.tcp_force[4] if len(state.tcp_force) > 4 else 0.0
                    self.current_tcp_wrench.torque.z = state.tcp_force[5] if len(state.tcp_force) > 5 else 0.0
                
                # Check safety state
                if state.safety_mode == SafetyMode.PROTECTIVE_STOP:
                    self.protective_stop = True
                    self.robot_mode = RobotMode.ERROR
                elif state.safety_mode in [SafetyMode.SYSTEM_EMERGENCY_STOP, SafetyMode.ROBOT_EMERGENCY_STOP]:
                    self.emergency_stop = True
                    self.robot_mode = RobotMode.ERROR
                elif state.safety_mode == SafetyMode.NORMAL:
                    self.protective_stop = False
                    self.emergency_stop = False
                    if state.robot_mode == RobotMode.RUNNING:
                        self.robot_mode = RobotMode.RUNNING
                    else:
                        self.robot_mode = RobotMode.IDLE
                        
        except Exception as e:
            self.get_logger().debug(f"Error reading robot state: {e}")

    def publish_state(self):
        """Publish robot state at configured rate."""
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
        
        # Publish full robot state
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
        velocity: float = 1.05,
        acceleration: float = 1.4,
    ) -> bool:
        """Move robot to target joint positions."""
        if self.protective_stop or self.emergency_stop:
            self.get_logger().error("Cannot move - robot in stop state")
            return False
        
        if self.simulation_mode:
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
        
        if not self.robot:
            self.get_logger().error("Not connected to robot")
            return False
        
        try:
            self.robot_mode = RobotMode.RUNNING
            result = self.robot.movej(target_joints, velocity=velocity, acceleration=acceleration)
            # Note: movej returns immediately after sending command
            # Motion completion should be tracked via state updates
            return result
            
        except Exception as e:
            self.get_logger().error(f"Motion failed: {e}")
            self.robot_mode = RobotMode.ERROR
            return False

    def move_linear(
        self,
        target_pose: List[float],
        velocity: float = 0.25,
        acceleration: float = 1.2,
    ) -> bool:
        """Move robot linearly to target TCP pose."""
        if self.protective_stop or self.emergency_stop:
            self.get_logger().error("Cannot move - robot in stop state")
            return False
        
        if self.simulation_mode:
            self.get_logger().info(f"[SIM] Linear move to: {target_pose}")
            self.robot_mode = RobotMode.RUNNING
            time.sleep(1.0)
            self.robot_mode = RobotMode.IDLE
            return True
        
        if not self.robot:
            self.get_logger().error("Not connected to robot")
            return False
        
        try:
            self.robot_mode = RobotMode.RUNNING
            result = self.robot.movel(target_pose, velocity=velocity, acceleration=acceleration)
            return result
            
        except Exception as e:
            self.get_logger().error(f"Motion failed: {e}")
            self.robot_mode = RobotMode.ERROR
            return False

    def stop_motion(self):
        """Stop current motion."""
        self.get_logger().warn("Stopping motion")
        
        if self.simulation_mode:
            self.robot_mode = RobotMode.IDLE
            return
        
        if self.robot:
            try:
                self.robot.stop()
            except Exception as e:
                self.get_logger().error(f"Stop failed: {e}")

    # === Service Callbacks ===
    
    def protective_stop_callback(self, request, response):
        """Handle protective stop request."""
        self.get_logger().warn("PROTECTIVE STOP triggered")
        
        self.protective_stop = True
        self.stop_motion()
        
        response.success = True
        response.message = "Protective stop activated"
        return response

    def emergency_stop_callback(self, request, response):
        """Handle emergency stop request."""
        self.get_logger().fatal("EMERGENCY STOP triggered")
        
        self.emergency_stop = True
        self.protective_stop = True
        self.stop_motion()
        
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
        
        if not self.simulation_mode and self.robot:
            try:
                self.robot.unlock_protective_stop()
                self.robot.close_safety_popup()
            except Exception as e:
                self.get_logger().warn(f"Reset warning: {e}")
        
        self.protective_stop = False
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
            response.success = True
            response.message = "Freedrive toggled (simulation)"
            return response
        
        if not self.robot:
            response.success = False
            response.message = "Not connected to robot"
            return response
        
        try:
            if self.robot_mode == RobotMode.FREEDRIVE:
                self.robot.freedrive(False)
                self.robot_mode = RobotMode.IDLE
                response.message = "Freedrive disabled"
            else:
                self.robot.freedrive(True)
                self.robot_mode = RobotMode.FREEDRIVE
                response.message = "Freedrive enabled"
            
            response.success = True
            
        except Exception as e:
            response.success = False
            response.message = f"Freedrive toggle failed: {e}"
        
        return response

    def power_on_callback(self, request, response):
        """Power on the robot."""
        self.get_logger().info("Power on requested")
        
        if self.simulation_mode:
            response.success = True
            response.message = "Power on (simulation)"
            return response
        
        if not self.robot:
            response.success = False
            response.message = "Not connected to robot"
            return response
        
        try:
            if self.robot.power_on():
                response.success = True
                response.message = "Robot powered on"
            else:
                response.success = False
                response.message = "Power on failed"
        except Exception as e:
            response.success = False
            response.message = f"Power on error: {e}"
        
        return response

    def power_off_callback(self, request, response):
        """Power off the robot."""
        self.get_logger().info("Power off requested")
        
        if self.simulation_mode:
            response.success = True
            response.message = "Power off (simulation)"
            return response
        
        if not self.robot:
            response.success = False
            response.message = "Not connected to robot"
            return response
        
        try:
            if self.robot.power_off():
                response.success = True
                response.message = "Robot powered off"
            else:
                response.success = False
                response.message = "Power off failed"
        except Exception as e:
            response.success = False
            response.message = f"Power off error: {e}"
        
        return response


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    node = RobotControlNodeV2()
    
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
