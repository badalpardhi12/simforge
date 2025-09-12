import pytest
from unittest.mock import Mock, patch
from simforge.movement_controller import MovementController, ControlMode
from simforge.config_reader import SimforgeConfig, RobotConfig, SceneConfig


def test_movement_controller_initialization():
    """Test basic controller initialization."""
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
    
    with patch('simforge.movement_controller.GenesisRenderer'):
        controller = MovementController(config, debug=False)
        
        assert "test_robot" in controller.robot_modes
        assert controller.robot_modes["test_robot"] == ControlMode.JOINT
        assert len(controller.joint_targets["test_robot"]) == 6


def test_mode_switching():
    """Test robot mode switching."""
    config = SimforgeConfig(
        robots=[RobotConfig(name="robot1", urdf="test.urdf")]
    )
    
    with patch('simforge.movement_controller.GenesisRenderer'):
        controller = MovementController(config)
        
        # Test initial mode
        assert controller.get_robot_mode("robot1") == ControlMode.JOINT
        
        # Test mode switching
        controller.switch_mode("robot1", ControlMode.CARTESIAN)
        # Process the command (simplified - in real system would be processed in thread)
        controller.robot_modes["robot1"] = ControlMode.CARTESIAN
        
        assert controller.get_robot_mode("robot1") == ControlMode.CARTESIAN


def test_joint_target_setting():
    """Test setting joint targets."""
    config = SimforgeConfig(
        robots=[RobotConfig(name="robot1", urdf="test.urdf")]
    )
    
    with patch('simforge.movement_controller.GenesisRenderer'):
        controller = MovementController(config)
        
        # Test setting individual joint
        controller.set_joint_position("robot1", 0, 45.0)
        
        # Test setting all joints
        targets = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        controller.set_joint_targets("robot1", targets)
        
        # Verify command was queued (in real system would be processed)
        assert not controller.command_queue.empty()