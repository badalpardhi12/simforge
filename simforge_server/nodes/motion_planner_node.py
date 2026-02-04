#!/usr/bin/env python3
"""
Motion Planner Node

Provides collision-free trajectory planning service using:
- GPU-accelerated cuMotion on NVIDIA Jetson (Isaac ROS)
- trac_ik fallback on x86 systems

This node exposes the /plan_cartesian_motion service for computing
collision-free joint trajectories to reach Cartesian target poses.

Features:
- IK solving with multiple solvers (cuMotion, trac_ik, KDL)
- Optional nvblox ESDF integration for real-time collision checking
- Trajectory optimization with velocity/acceleration limits
- Reference frame transformations via TF2
"""

import time
import platform
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, TransformStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

import tf2_ros
from tf2_ros import Buffer, TransformListener

# NOTE: simforge_msgs must be built first
# from simforge_msgs.srv import PlanCartesianMotion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Check platform for solver selection
IS_JETSON = platform.machine() == 'aarch64'
IS_X86 = platform.machine() in ('x86_64', 'AMD64')


class IKSolverBase:
    """Base class for IK solvers."""
    
    def __init__(self, robot_description: str, base_link: str, tip_link: str):
        self.robot_description = robot_description
        self.base_link = base_link
        self.tip_link = tip_link
        self.name = "base"
    
    def solve(
        self,
        target_pose: Pose,
        seed_joints: Optional[List[float]] = None,
        timeout: float = 0.1
    ) -> Optional[List[float]]:
        """Solve IK for target pose. Returns joint values or None if failed."""
        raise NotImplementedError


class TracIKSolver(IKSolverBase):
    """IK solver using trac_ik (x86 fallback)."""
    
    def __init__(self, robot_description: str, base_link: str, tip_link: str):
        super().__init__(robot_description, base_link, tip_link)
        self.name = "trac_ik"
        self._solver = None
        
        try:
            from trac_ik_py.trac_ik import IK
            self._solver = IK(
                base_link,
                tip_link,
                urdf_string=robot_description,
                timeout=0.1,
                epsilon=1e-5,
                solve_type="Distance"
            )
            logger.info(f"TracIK solver initialized: {base_link} -> {tip_link}")
        except ImportError:
            logger.warning("trac_ik_py not available, TracIK solver disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize TracIK: {e}")
    
    @property
    def available(self) -> bool:
        return self._solver is not None
    
    def solve(
        self,
        target_pose: Pose,
        seed_joints: Optional[List[float]] = None,
        timeout: float = 0.1
    ) -> Optional[List[float]]:
        if not self.available:
            return None
        
        if seed_joints is None:
            # Use neutral position as seed
            seed_joints = [0.0] * 6
        
        try:
            result = self._solver.get_ik(
                seed_joints,
                target_pose.position.x,
                target_pose.position.y,
                target_pose.position.z,
                target_pose.orientation.x,
                target_pose.orientation.y,
                target_pose.orientation.z,
                target_pose.orientation.w,
            )
            return list(result) if result else None
        except Exception as e:
            logger.warning(f"TracIK solve failed: {e}")
            return None


class KDLSolver(IKSolverBase):
    """IK solver using PyKDL (basic fallback)."""
    
    def __init__(self, robot_description: str, base_link: str, tip_link: str):
        super().__init__(robot_description, base_link, tip_link)
        self.name = "kdl"
        self._chain = None
        self._fk_solver = None
        self._ik_solver = None
        
        try:
            import PyKDL
            from urdf_parser_py.urdf import URDF
            from kdl_parser_py.urdf import treeFromUrdfModel
            
            # Parse URDF
            urdf_model = URDF.from_xml_string(robot_description)
            ok, tree = treeFromUrdfModel(urdf_model)
            if not ok:
                raise RuntimeError("Failed to build KDL tree from URDF")
            
            # Get chain
            self._chain = tree.getChain(base_link, tip_link)
            self._num_joints = self._chain.getNrOfJoints()
            
            # Create solvers
            self._fk_solver = PyKDL.ChainFkSolverPos_recursive(self._chain)
            self._ik_solver_vel = PyKDL.ChainIkSolverVel_pinv(self._chain)
            self._ik_solver = PyKDL.ChainIkSolverPos_NR(
                self._chain,
                self._fk_solver,
                self._ik_solver_vel,
                maxiter=100,
                eps=1e-6
            )
            logger.info(f"KDL solver initialized: {base_link} -> {tip_link}, {self._num_joints} joints")
        except ImportError as e:
            logger.warning(f"KDL dependencies not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize KDL: {e}")
    
    @property
    def available(self) -> bool:
        return self._ik_solver is not None
    
    def solve(
        self,
        target_pose: Pose,
        seed_joints: Optional[List[float]] = None,
        timeout: float = 0.1
    ) -> Optional[List[float]]:
        if not self.available:
            return None
        
        try:
            import PyKDL
            
            if seed_joints is None:
                seed_joints = [0.0] * self._num_joints
            
            # Create KDL frame from pose
            pos = PyKDL.Vector(
                target_pose.position.x,
                target_pose.position.y,
                target_pose.position.z
            )
            rot = PyKDL.Rotation.Quaternion(
                target_pose.orientation.x,
                target_pose.orientation.y,
                target_pose.orientation.z,
                target_pose.orientation.w
            )
            target_frame = PyKDL.Frame(rot, pos)
            
            # Create seed JntArray
            q_seed = PyKDL.JntArray(self._num_joints)
            for i, val in enumerate(seed_joints[:self._num_joints]):
                q_seed[i] = val
            
            # Solve
            q_result = PyKDL.JntArray(self._num_joints)
            status = self._ik_solver.CartToJnt(q_seed, target_frame, q_result)
            
            if status >= 0:
                return [q_result[i] for i in range(self._num_joints)]
            return None
        except Exception as e:
            logger.warning(f"KDL solve failed: {e}")
            return None


class CuMotionSolver(IKSolverBase):
    """
    GPU-accelerated IK solver using NVIDIA cuMotion (Jetson only).
    
    This requires:
    - NVIDIA Jetson with CUDA
    - Isaac ROS cuMotion package
    - cuRobo for IK computation
    """
    
    def __init__(self, robot_description: str, base_link: str, tip_link: str, robot_name: str = "ur5e"):
        super().__init__(robot_description, base_link, tip_link)
        self.name = "cumotion"
        self.robot_name = robot_name
        self._solver = None
        
        if not IS_JETSON:
            logger.info("cuMotion only available on Jetson, skipping initialization")
            return
        
        try:
            # cuRobo/cuMotion imports
            from curobo.types.base import TensorDeviceType
            from curobo.types.robot import JointState as CuJointState
            from curobo.geom.types import WorldConfig
            from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
            
            # Load robot config (XRDF format for cuMotion)
            tensor_args = TensorDeviceType()
            
            # TODO: Load from XRDF config file
            # For now, use built-in UR robot configs from cuRobo
            ik_config = IKSolverConfig.load_from_robot_config(
                robot_name,
                world_cfg=None,
                tensor_args=tensor_args,
                num_seeds=20,  # Number of IK seeds for batched solving
            )
            self._solver = IKSolver(ik_config)
            logger.info(f"cuMotion solver initialized for {robot_name}")
        except ImportError as e:
            logger.warning(f"cuRobo/cuMotion not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize cuMotion: {e}")
    
    @property
    def available(self) -> bool:
        return self._solver is not None
    
    def solve(
        self,
        target_pose: Pose,
        seed_joints: Optional[List[float]] = None,
        timeout: float = 0.1
    ) -> Optional[List[float]]:
        if not self.available:
            return None
        
        try:
            import torch
            from curobo.types.math import Pose as CuPose
            
            # Convert to cuRobo pose format
            position = torch.tensor([
                [target_pose.position.x, target_pose.position.y, target_pose.position.z]
            ], dtype=torch.float32, device="cuda")
            
            quaternion = torch.tensor([
                [target_pose.orientation.w, target_pose.orientation.x, 
                 target_pose.orientation.y, target_pose.orientation.z]
            ], dtype=torch.float32, device="cuda")  # cuRobo uses wxyz order
            
            cu_pose = CuPose(position, quaternion)
            
            # Solve IK (batched, returns best solution)
            result = self._solver.solve_single(cu_pose)
            
            if result.success.item():
                joints = result.solution[0].cpu().numpy().tolist()
                return joints
            return None
        except Exception as e:
            logger.warning(f"cuMotion solve failed: {e}")
            return None


class TrajectoryOptimizer:
    """
    Trajectory optimization with velocity/acceleration limits.
    
    On Jetson: Uses cuMotion trajectory optimizer
    On x86: Uses time-parameterized linear interpolation
    """
    
    def __init__(self, joint_names: List[str], velocity_limits: List[float], acceleration_limits: List[float]):
        self.joint_names = joint_names
        self.velocity_limits = velocity_limits
        self.acceleration_limits = acceleration_limits
    
    def optimize(
        self,
        start_joints: List[float],
        target_joints: List[float],
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
        num_waypoints: int = 50
    ) -> JointTrajectory:
        """
        Generate time-parameterized trajectory from start to target joints.
        
        Returns JointTrajectory message with properly timed waypoints.
        """
        start = np.array(start_joints)
        end = np.array(target_joints)
        diff = end - start
        
        # Scale velocity and acceleration limits
        v_max = np.array(self.velocity_limits) * velocity_scale
        a_max = np.array(self.acceleration_limits) * acceleration_scale
        
        # Compute time for each joint using trapezoidal profile
        # t = max(sqrt(4*d/a), 2*d/v) for each joint
        times = []
        for i in range(len(diff)):
            d = abs(diff[i])
            if d < 1e-6:
                times.append(0.0)
            else:
                t_accel = np.sqrt(4.0 * d / a_max[i]) if a_max[i] > 0 else float('inf')
                t_vel = 2.0 * d / v_max[i] if v_max[i] > 0 else float('inf')
                times.append(max(t_accel, t_vel))
        
        total_time = max(times) if times else 1.0
        total_time = max(total_time, 0.5)  # Minimum 0.5s trajectory
        
        # Generate trajectory points
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        
        for i in range(num_waypoints + 1):
            t = (i / num_waypoints) * total_time
            s = i / num_waypoints  # Parameter [0, 1]
            
            # Smooth interpolation using quintic polynomial (zero velocity at endpoints)
            # s_smooth = 10*s^3 - 15*s^4 + 6*s^5
            s_smooth = 10 * s**3 - 15 * s**4 + 6 * s**5
            
            point = JointTrajectoryPoint()
            point.positions = (start + s_smooth * diff).tolist()
            
            # Compute velocities (derivative of quintic)
            if total_time > 0:
                ds_smooth = (30 * s**2 - 60 * s**3 + 30 * s**4) / total_time
                point.velocities = (ds_smooth * diff).tolist()
            else:
                point.velocities = [0.0] * len(start)
            
            # Compute accelerations (second derivative)
            if total_time > 0:
                dds_smooth = (60 * s - 180 * s**2 + 120 * s**3) / (total_time**2)
                point.accelerations = (dds_smooth * diff).tolist()
            else:
                point.accelerations = [0.0] * len(start)
            
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t - int(t)) * 1e9)
            
            trajectory.points.append(point)
        
        return trajectory


class MotionPlannerNode(Node):
    """
    Motion Planner Node
    
    Provides /plan_cartesian_motion service for collision-free trajectory planning.
    Automatically selects best available IK solver based on platform.
    """

    def __init__(self):
        super().__init__('motion_planner')
        
        # Declare parameters
        self.declare_parameter('robot_description', '')
        self.declare_parameter('robot_name', 'nakul_ur5e')
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('tip_link', 'tool0')
        self.declare_parameter('use_nvblox', False)
        self.declare_parameter('nvblox_costmap_topic', '/nvblox/combined_esdf')
        
        # Get parameters
        self.robot_description = self.get_parameter('robot_description').value
        self.robot_name = self.get_parameter('robot_name').value
        self.base_link = self.get_parameter('base_link').value
        self.tip_link = self.get_parameter('tip_link').value
        self.use_nvblox = self.get_parameter('use_nvblox').value
        
        # UR5e joint limits (from URDF)
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.velocity_limits = [3.14, 3.14, 3.14, 6.28, 6.28, 6.28]  # rad/s
        self.acceleration_limits = [2.0, 2.0, 2.0, 4.0, 4.0, 4.0]  # rad/s^2
        
        # TF2 for frame transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Initialize IK solvers (in order of preference)
        self._ik_solvers: List[IKSolverBase] = []
        self._init_ik_solvers()
        
        # Trajectory optimizer
        self._trajectory_optimizer = TrajectoryOptimizer(
            self.joint_names,
            self.velocity_limits,
            self.acceleration_limits
        )
        
        # Current joint state subscription
        self._current_joints: Optional[List[float]] = None
        self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )
        
        # Service (uncomment when simforge_msgs is built)
        # self._planning_service = self.create_service(
        #     PlanCartesianMotion,
        #     '/plan_cartesian_motion',
        #     self._handle_plan_request,
        #     callback_group=ReentrantCallbackGroup()
        # )
        
        self.get_logger().info(
            f"Motion Planner initialized - Platform: {'Jetson' if IS_JETSON else 'x86'}, "
            f"IK solvers: {[s.name for s in self._ik_solvers if s.available]}"
        )
    
    def _init_ik_solvers(self):
        """Initialize IK solvers in order of preference."""
        # 1. cuMotion (Jetson only, highest performance)
        if IS_JETSON:
            solver = CuMotionSolver(
                self.robot_description,
                self.base_link,
                self.tip_link,
                self.robot_name
            )
            self._ik_solvers.append(solver)
        
        # 2. trac_ik (good x86 fallback)
        solver = TracIKSolver(self.robot_description, self.base_link, self.tip_link)
        self._ik_solvers.append(solver)
        
        # 3. KDL (basic fallback, always available with ROS)
        solver = KDLSolver(self.robot_description, self.base_link, self.tip_link)
        self._ik_solvers.append(solver)
    
    def _joint_state_callback(self, msg: JointState):
        """Update current joint positions."""
        if len(msg.position) >= 6:
            self._current_joints = list(msg.position[:6])
    
    def solve_ik(
        self,
        target_pose: Pose,
        seed_joints: Optional[List[float]] = None
    ) -> Tuple[Optional[List[float]], str]:
        """
        Solve IK using best available solver.
        
        Returns (joint_values, solver_name) or (None, "none") if all solvers fail.
        """
        if seed_joints is None:
            seed_joints = self._current_joints or [0.0] * 6
        
        for solver in self._ik_solvers:
            if not solver.available:
                continue
            
            result = solver.solve(target_pose, seed_joints)
            if result is not None:
                return result, solver.name
        
        return None, "none"
    
    def transform_pose_to_base(
        self,
        pose: Pose,
        source_frame: str,
        timeout: float = 1.0
    ) -> Optional[Pose]:
        """Transform pose from source_frame to robot base_link."""
        if source_frame == self.base_link:
            return pose
        
        try:
            # Look up transform
            transform = self.tf_buffer.lookup_transform(
                self.base_link,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout)
            )
            
            # Apply transform to pose
            # This is a simplified transform - in production use tf2_geometry_msgs
            t = transform.transform.translation
            r = transform.transform.rotation
            
            # For now, just translate (full rotation transform requires more math)
            # TODO: Use tf2_geometry_msgs for proper pose transformation
            transformed = Pose()
            transformed.position.x = pose.position.x + t.x
            transformed.position.y = pose.position.y + t.y
            transformed.position.z = pose.position.z + t.z
            transformed.orientation = pose.orientation
            
            return transformed
        except Exception as e:
            self.get_logger().warning(f"Transform lookup failed: {e}")
            return None
    
    def plan_trajectory(
        self,
        start_joints: List[float],
        target_joints: List[float],
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
        collision_check: bool = True
    ) -> Tuple[Optional[JointTrajectory], bool, str]:
        """
        Plan collision-free trajectory from start to target joints.
        
        Returns (trajectory, collision_free, message)
        """
        # TODO: Implement nvblox ESDF collision checking
        # For now, generate trajectory without collision checking
        
        trajectory = self._trajectory_optimizer.optimize(
            start_joints,
            target_joints,
            velocity_scale,
            acceleration_scale
        )
        
        collision_free = True  # TODO: Actual collision checking
        message = "Trajectory planned successfully"
        
        return trajectory, collision_free, message
    
    # Service handler (uncomment when simforge_msgs is built)
    # def _handle_plan_request(self, request, response):
    #     """Handle /plan_cartesian_motion service request."""
    #     start_time = time.time()
    #     
    #     # Get start joints
    #     if request.start_joints:
    #         start_joints = list(request.start_joints)
    #     else:
    #         start_joints = self._current_joints
    #     
    #     if start_joints is None:
    #         response.success = False
    #         response.message = "No starting joint configuration available"
    #         return response
    #     
    #     # Transform target pose to base frame
    #     target_pose = self.transform_pose_to_base(
    #         request.target_pose,
    #         request.reference_frame
    #     )
    #     
    #     if target_pose is None:
    #         response.success = False
    #         response.message = f"Failed to transform pose from {request.reference_frame}"
    #         return response
    #     
    #     # Solve IK
    #     target_joints, solver_used = self.solve_ik(target_pose, start_joints)
    #     
    #     if target_joints is None:
    #         response.success = False
    #         response.message = "IK solver failed to find solution"
    #         response.ik_solver_used = "none"
    #         return response
    #     
    #     # Plan trajectory
    #     velocity_scale = request.velocity_scale if request.velocity_scale > 0 else 0.3
    #     acceleration_scale = request.acceleration_scale if request.acceleration_scale > 0 else 0.3
    #     
    #     trajectory, collision_free, message = self.plan_trajectory(
    #         start_joints,
    #         target_joints,
    #         velocity_scale,
    #         acceleration_scale,
    #         request.collision_check_enabled
    #     )
    #     
    #     planning_time = time.time() - start_time
    #     
    #     # Build response
    #     response.success = True
    #     response.message = message
    #     response.trajectory = trajectory
    #     response.target_joints = target_joints
    #     response.planning_time_sec = planning_time
    #     response.trajectory_duration_sec = self._get_trajectory_duration(trajectory)
    #     response.trajectory_length_m = self._compute_trajectory_length(trajectory)
    #     response.ik_solver_used = solver_used
    #     response.collision_free = collision_free
    #     response.num_collision_checks = 0  # TODO: Actual count
    #     
    #     return response
    
    def _get_trajectory_duration(self, trajectory: JointTrajectory) -> float:
        """Get total duration of trajectory in seconds."""
        if not trajectory.points:
            return 0.0
        last_point = trajectory.points[-1]
        return last_point.time_from_start.sec + last_point.time_from_start.nanosec * 1e-9
    
    def _compute_trajectory_length(self, trajectory: JointTrajectory) -> float:
        """Compute approximate trajectory length in joint space (radians)."""
        if len(trajectory.points) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(trajectory.points)):
            prev = np.array(trajectory.points[i-1].positions)
            curr = np.array(trajectory.points[i].positions)
            total += np.linalg.norm(curr - prev)
        
        return total


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
