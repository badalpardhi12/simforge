import os
os.environ.setdefault("PYBULLET_SUPPRESS_URDF_WARNINGS", "1")
import pybullet as p
import contextlib
import time
import numpy as np
import warnings
from ompl import base as ob
from ompl import geometric as og
try:
    from ompl import util as ou
    # Reduce OMPL console chatter
    ou.setLogLevel(ou.LogLevel.LOG_WARN)
except Exception:
    pass

# Suppress PyBullet warnings about missing inertial data
warnings.filterwarnings("ignore", category=UserWarning, module="pybullet")
# Reserved for prior solvers; no ikpy used now

class PybulletPlanner:
    """A motion planner using OMPL with PyBullet for collision detection."""

    def __init__(self,
                 urdf_path: str,
                 fixed_base: bool = True,
                 min_clearance: float = 0.0,
                 base_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 base_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 joint_limits_by_name: dict | None = None):
        """
        Initializes the OMPL planner with PyBullet collision checking.

        Args:
            urdf_path (str): The file path to the robot's URDF.
            fixed_base (bool): Whether the robot's base is fixed.
        """
        self.client = p.connect(p.DIRECT)
        
        @contextlib.contextmanager
        def _silence_stdio():
            try:
                stdout_fd = os.dup(1)
                stderr_fd = os.dup(2)
                devnull_fd = os.open('/dev/null', os.O_WRONLY)
                os.dup2(devnull_fd, 1)
                os.dup2(devnull_fd, 2)
                yield
            finally:
                try:
                    os.dup2(stdout_fd, 1)
                    os.dup2(stderr_fd, 2)
                    os.close(devnull_fd)
                    os.close(stdout_fd)
                    os.close(stderr_fd)
                except Exception:
                    pass

        # Place planning robot at its configured base pose to keep obstacle relations consistent
        base_rpy_rad = tuple(np.deg2rad(v) for v in base_rpy_deg)
        base_quat = p.getQuaternionFromEuler(base_rpy_rad, physicsClientId=self.client)
        with _silence_stdio():
            # Enable self-collision in the planning model so contact queries
            # report robot-vs-robot (self) intersections. Exclude parent pairs
            # to reduce noise; we also filter in software as a safeguard.
            flags = getattr(p, 'URDF_USE_SELF_COLLISION_EXCLUDE_PARENT', 0) | getattr(p, 'URDF_USE_SELF_COLLISION', 0)
            try:
                self.robot_id = p.loadURDF(
                    urdf_path,
                    basePosition=base_position,
                    baseOrientation=base_quat,
                    useFixedBase=fixed_base,
                    flags=flags,
                    physicsClientId=self.client,
                )
            except TypeError:
                # Older pybullet may not support flags kwarg
                self.robot_id = p.loadURDF(
                    urdf_path,
                    basePosition=base_position,
                    baseOrientation=base_quat,
                    useFixedBase=fixed_base,
                    physicsClientId=self.client,
                )
        
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        self.joint_indices = []
        self.joint_limits = []
        # Build self-collision adjacency (parent-child link pairs) to ignore
        self._adjacent_self_pairs = set()
        
        # Get joint information and limits
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            joint_type = joint_info[2]
            parent_link = int(joint_info[16]) if len(joint_info) > 16 else -1
            child_link = i
            self._adjacent_self_pairs.add(frozenset((parent_link, child_link)))
            # Only consider revolute and prismatic joints
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_indices.append(i)
                # Prefer explicit URDF limits by name if provided
                lower_limit = None
                upper_limit = None
                if joint_limits_by_name is not None:
                    try:
                        jname = joint_info[1].decode('utf-8') if isinstance(joint_info[1], (bytes, bytearray)) else str(joint_info[1])
                        lims = joint_limits_by_name.get(jname)
                        if lims is not None:
                            lower_limit = float(lims[0]) if lims[0] is not None else None
                            upper_limit = float(lims[1]) if lims[1] is not None else None
                    except Exception:
                        pass
                if lower_limit is None or upper_limit is None:
                    # Fallback to PyBullet joint limits; if undefined, use generous ±2π
                    jl = joint_info[8]
                    ju = joint_info[9]
                    lower_limit = jl if jl > -1e6 else -2*np.pi
                    upper_limit = ju if ju < 1e6 else 2*np.pi
                self.joint_limits.append((lower_limit, upper_limit))
        
        # Set up OMPL
        self.space = ob.RealVectorStateSpace(len(self.joint_indices))
        bounds = ob.RealVectorBounds(len(self.joint_indices))
        
        for i, (lower, upper) in enumerate(self.joint_limits):
            bounds.setLow(i, lower)
            bounds.setHigh(i, upper)
        
        self.space.setBounds(bounds)
        
        # Create simple setup
        self.ss = og.SimpleSetup(self.space)
        
        # Set state validity checker
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))
        
        # Cache obstacles for collision checking
        self.obstacles = []
        # Minimum clearance distance (meters) for obstacle avoidance
        self.min_clearance = float(min_clearance)
        # Set of obstacle body IDs that should be EXEMPT from clearance checks
        # (e.g., ground plane) but still participate in hard contact checks.
        self._clearance_exempt_ids: set[int] = set()

    def set_clearance_exempt_ids(self, ids: list[int] | set[int]):
        """Mark obstacle body IDs that should not enforce min_clearance.

        Contacts with these IDs still invalidate states; only the clearance
        (getClosestPoints) test is skipped for them.
        """
        try:
            self._clearance_exempt_ids = set(int(x) for x in ids)
        except Exception:
            # Fallback to empty set on bad input
            self._clearance_exempt_ids = set()

    def close(self):
        """Disconnect the dedicated PyBullet client for this planner."""
        try:
            if getattr(self, 'client', None) is not None:
                p.disconnect(self.client)
        except Exception:
            pass
        finally:
            self.client = None

    def __del__(self):
        # Best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def isStateValid(self, state):
        """Check if a state is valid (collision-free)."""
        # Extract configuration from OMPL state
        config = []
        for i in range(len(self.joint_indices)):
            config.append(state[i])
        
        return self._is_collision_free(config, self.obstacles)

    def _is_collision_free(self, config: list, obstacles: list = []) -> bool:
        """Check if a configuration is collision-free."""
        # Set the robot to the given configuration
        for i, q in enumerate(config):
            if i < len(self.joint_indices):
                p.resetJointState(self.robot_id, self.joint_indices[i], q, physicsClientId=self.client)
        
        # Check for collisions: block robot-vs-obstacle and non-adjacent self-collisions
        contact_points = p.getContactPoints(bodyA=self.robot_id, physicsClientId=self.client)
        for contact in contact_points:
            # If it's a collision with an obstacle body, block
            if contact[2] in obstacles:
                return False
            # If it's a self-collision between non-adjacent links, block
            if contact[1] == self.robot_id and contact[2] == self.robot_id:
                link_a = contact[3]
                link_b = contact[4]
                pair = frozenset((int(link_a), int(link_b)))
                if pair not in self._adjacent_self_pairs:
                    return False
        
        # Clearance check against obstacles using closest points (conservative)
        if obstacles:
            try:
                for obs_id in obstacles:
                    # For exempt obstacles (e.g., ground), skip clearance distance checks entirely
                    # and rely solely on the contact test above to prevent penetration.
                    if int(obs_id) in self._clearance_exempt_ids:
                        continue
                    if self.min_clearance > 0.0:
                        cps = p.getClosestPoints(self.robot_id, obs_id, distance=self.min_clearance, physicsClientId=self.client)
                        if cps:
                            return False
            except Exception:
                # If closest points not available for some reason, ignore and rely on contact checks
                pass

        return True

    def plan_path(self, start_config: list, goal_config: list, obstacles: list = [], clearance_exempt_ids: list[int] | None = None) -> list | None:
        """
        Plans a collision-free path using OMPL.

        Args:
            start_config (list): The starting joint configuration.
            goal_config (list): The target joint configuration.
            obstacles (list): A list of obstacle IDs to consider for collision checking.

        Returns:
            list | None: A list of joint configurations representing the path, or None if no path is found.
        """
        # Store obstacles for collision checking
        self.obstacles = obstacles
        if clearance_exempt_ids is not None:
            self.set_clearance_exempt_ids(clearance_exempt_ids)
        
        # Ensure configurations match expected DOF
        if len(start_config) != len(self.joint_indices):
            return None
        if len(goal_config) != len(self.joint_indices):
            return None

        # Quick validity pre-checks using PyBullet (contacts + min_clearance)
        if not self._is_collision_free(start_config, obstacles):
            return None
        if not self._is_collision_free(goal_config, obstacles):
            return None
        
        # Create start state
        start_state = ob.State(self.space)
        for i, q in enumerate(start_config):
            start_state[i] = q
        
        # Create goal state
        goal_state = ob.State(self.space)
        for i, q in enumerate(goal_config):
            goal_state[i] = q
        
        # Set start and goal states
        self.ss.setStartAndGoalStates(start_state, goal_state)
        
        # Configure SpaceInformation and motion/state resolution
        si = self.ss.getSpaceInformation()
        try:
            # Finer resolution improves reliability in clutter
            si.setStateValidityCheckingResolution(0.01)
        except Exception:
            pass
        try:
            # Use discrete motion validator to enforce segment-wise checks
            mv = og.DiscreteMotionValidator(si)
            si.setMotionValidator(mv)
        except Exception:
            pass

        # Attempt to solve the problem within ~5 seconds using ParallelPlan if available.
        # Fallback to a sequential multi-planner strategy.
        @contextlib.contextmanager
        def _silence_stdio():
            try:
                stdout_fd = os.dup(1)
                stderr_fd = os.dup(2)
                devnull_fd = os.open('/dev/null', os.O_WRONLY)
                os.dup2(devnull_fd, 1)
                os.dup2(devnull_fd, 2)
                yield
            finally:
                try:
                    os.dup2(stdout_fd, 1)
                    os.dup2(stderr_fd, 2)
                    os.close(devnull_fd)
                    os.close(stdout_fd)
                    os.close(stderr_fd)
                except Exception:
                    pass

        solved = False
        with _silence_stdio():
            try:
                # Build a list of planners to try
                planners = []
                def _maybe(pl):
                    try:
                        planners.append(pl)
                    except Exception:
                        pass
                # Heuristic range based on joint spans
                spans = [abs(hi - lo) if np.isfinite(hi - lo) else np.pi for (lo, hi) in self.joint_limits]
                avg_span = float(np.mean(spans)) if spans else 0.5

                # RRTConnect
                try:
                    pr = og.RRTConnect(si)
                    try:
                        pr.setRange(max(0.05, 0.2 * avg_span))
                    except Exception:
                        pass
                    _maybe(pr)
                except Exception:
                    pass
                # KPIECE1
                try:
                    _maybe(og.KPIECE1(si))
                except Exception:
                    pass
                # BiEST
                try:
                    _maybe(og.BiEST(si))
                except Exception:
                    pass
                # BITstar (if built)
                try:
                    _maybe(og.BITstar(si))
                except Exception:
                    pass

                # Try ParallelPlan if bindings provide it
                used_parallel = False
                try:
                    if hasattr(og, 'ParallelPlan') and planners:
                        pp = og.ParallelPlan(self.ss.getProblemDefinition())
                        for pl in planners:
                            pp.addPlanner(pl)
                        solved = pp.solve(5.0)
                        used_parallel = True
                except Exception:
                    used_parallel = False
                    solved = False

                # Fallback: sequential attempts with split time budget
                if not used_parallel:
                    # Set first planner on SS for convenience but iterate all
                    for pl in planners:
                        try:
                            self.ss.setPlanner(pl)
                        except Exception:
                            pass
                        if self.ss.solve(2.0):
                            solved = True
                            break
                        # Clear between attempts
                        try:
                            self.ss.clear()
                            self.ss.setStartAndGoalStates(start_state, goal_state)
                        except Exception:
                            pass
            except Exception:
                solved = False

        if solved:
            # Get the solution path
            path = self.ss.getSolutionPath()
            
            # Simplify the path
            self.ss.simplifySolution()
            
            # Get simplified path
            simplified_path = self.ss.getSolutionPath()
            
            # Convert to list of configurations
            sparse_configs = []
            for i in range(simplified_path.getStateCount()):
                state = simplified_path.getState(i)
                config = []
                for j in range(len(self.joint_indices)):
                    config.append(state[j])
                sparse_configs.append(config)
            
            # Clear the planner for next use
            self.ss.clear()
            
            # ALWAYS densify paths for smooth motion - especially for RPY changes
            if len(sparse_configs) >= 2:
                # Calculate actual path distance for analysis
                total_distance = 0.0
                for i in range(1, len(sparse_configs)):
                    segment_dist = np.linalg.norm(np.array(sparse_configs[i]) - np.array(sparse_configs[i-1]))
                    total_distance += segment_dist
                
                # Suppress noisy debug print; keep densification logic
                
                # FORCE densification for ALL movements, including micro-movements from RPY changes
                target_waypoints = max(50, min(100, int(total_distance * 500))) if total_distance > 1e-6 else 100
                dense_configs = self._densify_path(sparse_configs, obstacles, target_waypoints)
                return dense_configs
            
            # Return original sparse path if only 1 waypoint
            return sparse_configs
        else:
            # Clear the planner for next use
            self.ss.clear()
            # No linear interpolation fallback: if OMPL fails, propagate failure
            return None
    
    def _densify_path(self, sparse_path: list, obstacles: list = [], target_waypoints: int = 100) -> list:
        """
        Densify a sparse path to create smooth, high-resolution trajectory.
        
        Args:
            sparse_path: List of sparse waypoints from OMPL
            obstacles: List of obstacle IDs for collision checking
            target_waypoints: Target number of waypoints in densified path
            
        Returns:
            Dense list of waypoints for smooth animation
        """
        if len(sparse_path) <= 1:
            return sparse_path
        
        # Calculate cumulative distances along the path
        distances = [0.0]
        total_distance = 0.0
        
        for i in range(1, len(sparse_path)):
            segment_distance = np.linalg.norm(np.array(sparse_path[i]) - np.array(sparse_path[i-1]))
            total_distance += segment_distance
            distances.append(total_distance)
        
        # DETAILED DEBUGGING FOR RPY ISSUE
        start_config = np.array(sparse_path[0])
        end_config = np.array(sparse_path[-1])
        joint_diffs = np.abs(end_config - start_config)
        max_joint_diff = np.max(joint_diffs)
        
        # Suppressed verbose debug prints; keep logic intact
        
        # MICRO-MOVEMENT FIX: For very small movements (orientation changes),
        # create artificial intermediate waypoints to ensure visible motion
        MIN_MOVEMENT_THRESHOLD = 1e-6  # Much smaller threshold for tiny RPY changes
        
        if total_distance < MIN_MOVEMENT_THRESHOLD:
            # Micro-movement: orientation-only changes
            # This is likely an orientation-only change with micro joint movements
            # Create artificial intermediate steps to ensure visible motion
            if len(sparse_path) == 2:
                
                # Force minimum visible movement by scaling up the differences
                if max_joint_diff > 0:
                    # Scale to ensure at least 0.05 radian movement on the most changing joint
                    scale_factor = max(1.0, 0.05 / max_joint_diff)
                    scaled_end = start_config + scale_factor * (end_config - start_config)
                else:
                    scaled_end = end_config
                
                # Create 50 intermediate steps for smooth motion
                artificial_path = []
                for i in range(51):  # 0 to 50 inclusive
                    t = i / 50.0
                    intermediate = start_config + t * (scaled_end - start_config)
                    artificial_path.append(intermediate.tolist())
                return artificial_path
            return sparse_path
        
        # For small but significant movements, use appropriate waypoint count
        if total_distance < 0.1:
            target_waypoints = min(target_waypoints, max(15, int(total_distance * 300)))
        
        # Generate dense waypoints with uniform spacing
        dense_path = []
        current_sparse_idx = 0
        
        for i in range(target_waypoints):
            # Parametric position along the path (0 to 1)
            t_global = i / (target_waypoints - 1)
            target_distance = t_global * total_distance
            
            # Find which segment this distance falls into
            while (current_sparse_idx < len(distances) - 1 and
                   distances[current_sparse_idx + 1] < target_distance):
                current_sparse_idx += 1
            
            if current_sparse_idx >= len(sparse_path) - 1:
                # At the end of the path
                dense_path.append(sparse_path[-1])
            else:
                # Interpolate within the current segment
                segment_start_dist = distances[current_sparse_idx]
                segment_end_dist = distances[current_sparse_idx + 1]
                segment_length = segment_end_dist - segment_start_dist
                
                if segment_length > 0:
                    t_segment = (target_distance - segment_start_dist) / segment_length
                else:
                    t_segment = 0.0
                
                # Linear interpolation between the two waypoints
                start_config = np.array(sparse_path[current_sparse_idx])
                end_config = np.array(sparse_path[current_sparse_idx + 1])
                interpolated_config = start_config + t_segment * (end_config - start_config)
                dense_path.append(interpolated_config.tolist())
        
        # ENHANCED COLLISION VALIDATION with debugging
        validated_path = []
        collision_count = 0
        
        # Optional validation against obstacles
        
        for i, config in enumerate(dense_path):
            if self._is_collision_free(config, obstacles):
                validated_path.append(config)
            else:
                collision_count += 1
                # Skip colliding waypoints completely
                continue

        # If too many waypoints were removed due to collisions, fall back to sparse path
        if len(validated_path) < max(5, target_waypoints * 0.3):
            # Too many collisions in dense path, use original sparse path which OMPL validated
            return sparse_path
        
        # Calculate final path distance for verification
        if len(validated_path) >= 2:
            path_distance = 0.0
            for i in range(1, len(validated_path)):
                path_distance += np.linalg.norm(np.array(validated_path[i]) - np.array(validated_path[i-1]))
            # For micro-movements, don't reject based on distance
            if total_distance < MIN_MOVEMENT_THRESHOLD:
                return validated_path
            
            # If validated path is too short for normal movements, return original sparse path
            if path_distance < 0.001:
                return sparse_path
        
        return validated_path
    

    def set_joint_states(self, joint_config: list):
        """
        Sets the joint states of the robot in the PyBullet simulation.

        Args:
            joint_config (list): The joint configuration to set.
        """
        for i, q in enumerate(joint_config):
            p.setJointMotorControl2(
                self.robot_id,
                self.joint_indices[i],
                p.POSITION_CONTROL,
                targetPosition=q,
                physicsClientId=self.client,
            )

    # NOTE: __del__ handled above to call close()
