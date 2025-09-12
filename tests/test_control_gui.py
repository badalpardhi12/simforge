import pytest
from unittest.mock import Mock, patch
from simforge.control_gui import ControlGUI, run_headless
from simforge.config_reader import SimforgeConfig, RobotConfig, SceneConfig
from simforge.movement_controller import ControlMode


def test_control_gui_initialization():
    """Test GUI initialization."""
    config = SimforgeConfig(
        scene=SceneConfig(),
        robots=[
            RobotConfig(
                name="test_robot",
                urdf="test.urdf",
                initial_joint_positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )
        ]
    )
    
    with patch('simforge.control_gui.MovementController'):
        gui = ControlGUI(config, debug=False)
        
        assert "test_robot" in gui.robot_states
        assert gui.robot_states["test_robot"]["mode"] == ControlMode.JOINT
        assert len(gui.robot_states["test_robot"]["joint_targets"]) == 6


def test_robot_operations():
    """Test robot control operations."""
    config = SimforgeConfig(
        robots=[RobotConfig(name="robot1", urdf="test.urdf")]
    )
    
    with patch('simforge.control_gui.MovementController') as mock_controller:
        gui = ControlGUI(config)
        
        # Test mode switching
        gui.set_robot_mode("robot1", ControlMode.CARTESIAN)
        mock_controller.return_value.switch_mode.assert_called_with("robot1", ControlMode.CARTESIAN)
        
        # Test joint movement
        gui.move_joint("robot1", 0, 45.0)
        mock_controller.return_value.set_joint_position.assert_called_with("robot1", 0, 45.0)
        
        # Test Cartesian movement
        gui.move_cartesian("robot1", 0.5, 0.0, 0.3, 0.0, 180.0, 0.0)
        mock_controller.return_value.move_cartesian.assert_called_with(
            "robot1", (0.5, 0.0, 0.3), (0.0, 180.0, 0.0)
        )


def test_run_headless():
    """Test headless execution."""
    config = SimforgeConfig(
        robots=[RobotConfig(name="robot1", urdf="test.urdf")]
    )
    
    with patch('simforge.control_gui.MovementController') as mock_controller:
        result = run_headless(config, debug=False)
        
        mock_controller.assert_called_once_with(config, debug=False)
        mock_controller.return_value.build_scene.assert_called_once()
        mock_controller.return_value.start.assert_called_once()
        assert hasattr(result, 'controller')
        assert hasattr(result, 'stop')


def test_get_available_robots():
    """Test getting list of available robots."""
    config = SimforgeConfig(
        robots=[
            RobotConfig(name="robot1", urdf="test1.urdf"),
            RobotConfig(name="robot2", urdf="test2.urdf")
        ]
    )
    
    with patch('simforge.control_gui.MovementController'):
        gui = ControlGUI(config)
        robots = gui.get_available_robots()
        
        assert len(robots) == 2
        assert "robot1" in robots
        assert "robot2" in robots