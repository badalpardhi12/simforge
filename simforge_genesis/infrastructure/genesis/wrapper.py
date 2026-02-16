"""Genesis simulator wrapper that adapts the legacy simforge implementation."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Import from the original simforge package
import sys
original_simforge_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(original_simforge_path))

# Try to import legacy components
GENESIS_AVAILABLE = False
LEGACY_AVAILABLE = False

try:
    from simforge.genesis_renderer import GenesisRenderer
    from simforge.movement_controller import MovementController
    LEGACY_AVAILABLE = True
    logger.info("Legacy simforge components available")
except ImportError as e:
    logger.warning(f"Legacy components not available: {e}")

try:
    import genesis as gs
    GENESIS_AVAILABLE = True
    logger.info("Genesis package available")
except ImportError:
    logger.warning("Genesis package not available")


class GenesisWrapper:
    """
    Wrapper for Genesis simulator.
    Tries to use legacy simforge if available, otherwise direct Genesis.
    """
    
    def __init__(self, backend: str = "gpu", viewer_options: Dict = None):
        """Initialize the Genesis wrapper."""
        self.backend = backend
        self.viewer_options = viewer_options or {}
        self._renderer = None
        self._controller = None
        self._scene = None
        self._robots_data = []
        self._robot_entities = {}
        self._built = False
        
        # Determine which mode to use
        self.use_legacy = LEGACY_AVAILABLE
        self.use_direct = GENESIS_AVAILABLE and not LEGACY_AVAILABLE
        
        if not (LEGACY_AVAILABLE or GENESIS_AVAILABLE):
            logger.warning(
                "Neither legacy simforge nor Genesis package available. "
                "Running in mock mode."
            )
            self.use_mock = True
        else:
            self.use_mock = False
    
    def create_scene(self, dt: float = 0.01, gravity: List[float] = None):
        """Initialize Genesis scene configuration."""
        self._dt = dt
        self._gravity = gravity or [0.0, 0.0, -9.81]
        
        if self.use_direct:
            # Direct Genesis initialization
            gs.init(backend=self.backend)
            self._scene = gs.Scene(
                sim_options=gs.SimOptions(
                    dt=dt,
                    gravity=self._gravity
                ),
                viewer_options=gs.ViewerOptions(
                    camera_pos=self.viewer_options.get('camera_pos', [3.5, 0.0, 2.5]),
                    camera_lookat=self.viewer_options.get('camera_lookat', [0.0, 0.0, 0.0]),
                    camera_up=self.viewer_options.get('camera_up', [0.0, 0.0, 1.0]),
                    camera_fov=self.viewer_options.get('camera_fov', 40),
                    max_FPS=self.viewer_options.get('max_FPS', 60),
                ),
                show_viewer=True
            )
        elif self.use_legacy:
            # Prepare config for legacy renderer
            self._config_data = {
                'timestep': dt,
                'gravity': self._gravity,
                'viewer': self.viewer_options,
                'backend': self.backend,
                'robots': []
            }
        else:
            # Mock mode
            self._config_data = {
                'timestep': dt,
                'gravity': self._gravity
            }
    
    def add_robot(self, name: str, urdf_path: str, base_pos: List[float], base_quat: List[float]):
        """Add a robot to the scene."""
        robot_data = {
            'name': name,
            'urdf': str(urdf_path),
            'position': base_pos,
            'orientation': base_quat,
            'joint_positions': None
        }
        
        self._robots_data.append(robot_data)
        
        if self.use_direct and self._scene:
            # Add robot using direct Genesis
            robot = gs.morphs.URDF(
                file=str(urdf_path),
                pos=base_pos,
                quat=base_quat,
                fixed=False
            )
            self._scene.add_entity(robot, name=name)
            self._robot_entities[name] = robot
        elif self.use_legacy:
            # Add to config for legacy renderer
            if hasattr(self, '_config_data'):
                self._config_data['robots'].append(robot_data)
        
        return name
    
    def add_ground(self):
        """Add ground plane to the scene."""
        if self.use_direct and self._scene:
            ground = gs.morphs.Plane()
            self._scene.add_entity(ground, name="ground")
    
    def build(self):
        """Build the simulation scene."""
        if self._built:
            return
        
        if self.use_direct and self._scene:
            # Build direct Genesis scene
            self._scene.build()
            self._built = True
        elif self.use_legacy:
            # Build using legacy renderer
            try:
                self._renderer = GenesisRenderer(self._config_data)
                self._controller = MovementController(self._renderer)
                self._built = True
            except Exception as e:
                logger.error(f"Failed to build legacy scene: {e}")
                # Fall back to mock mode
                self.use_mock = True
                self.use_legacy = False
                self._built = True
        else:
            # Mock mode - just mark as built
            self._built = True
            logger.info("Running in mock mode")
    
    def set_robot_joints(self, robot_name: str, positions: np.ndarray):
        """Set joint positions for a robot."""
        if not self._built:
            raise RuntimeError("Scene not built. Call build() first.")
        
        # Update stored positions
        for robot_data in self._robots_data:
            if robot_data['name'] == robot_name:
                robot_data['joint_positions'] = positions.tolist()
                break
        
        if self.use_direct and robot_name in self._robot_entities:
            # Set using direct Genesis
            robot = self._robot_entities[robot_name]
            robot.set_dofs(positions)
        elif self.use_legacy and self._controller:
            # Set using legacy controller
            robot_idx = self._get_robot_index(robot_name)
            if robot_idx is not None:
                self._controller.set_joint_positions(robot_idx, positions.tolist())
    
    def get_robot_joints(self, robot_name: str) -> np.ndarray:
        """Get current joint positions for a robot."""
        if not self._built:
            return np.array([])
        
        # Return stored positions in mock mode
        for robot_data in self._robots_data:
            if robot_data['name'] == robot_name:
                if robot_data['joint_positions']:
                    return np.array(robot_data['joint_positions'])
                break
        
        if self.use_direct and robot_name in self._robot_entities:
            robot = self._robot_entities[robot_name]
            return robot.get_dofs()
        elif self.use_legacy and self._renderer:
            robot_idx = self._get_robot_index(robot_name)
            if robot_idx is not None:
                positions = self._renderer.get_joint_positions(robot_idx)
                return np.array(positions) if positions else np.array([])
        
        # Default return
        return np.zeros(6)  # Assume 6-DOF robot
    
    def _get_robot_index(self, robot_name: str) -> Optional[int]:
        """Get robot index by name."""
        for i, robot_data in enumerate(self._robots_data):
            if robot_data['name'] == robot_name:
                return i
        return None
    
    def step(self):
        """Step the simulation forward."""
        if self.use_direct and self._scene:
            self._scene.step()
        elif self.use_legacy and self._renderer:
            self._renderer.step()
        # Mock mode does nothing
    
    def close(self):
        """Close the simulator."""
        if self.use_direct and self._scene:
            # Genesis cleanup
            pass
        elif self.use_legacy and self._renderer:
            try:
                self._renderer.close()
            except:
                pass
        
        self._renderer = None
        self._controller = None
        self._scene = None
        self._built = False
    
    def is_running(self) -> bool:
        """Check if simulator is running."""
        return self._built
    
    def get_robot_count(self) -> int:
        """Get number of robots in the scene."""
        return len(self._robots_data)