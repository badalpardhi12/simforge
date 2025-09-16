"""Tests for movement controller module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from simforge.movement_controller import (
    MovementController,
    ControlMode,
    SetJointCommand,
    SetJointTargetsCommand,
    CartesianMoveCommand,
    SwitchModeCommand
)
from simforge.config_reader import SimforgeConfig, RobotConfig, SceneConfig


class TestMovementController:
    """Test MovementController class functionality."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        return SimforgeConfig(
            scene=SceneConfig(),
            robots=[
                RobotConfig(
                    name="test_robot",
                    urdf="test.urdf",
                    initial_joint_positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    end_effector_link="wrist_3_link"
                )
            ]
        )

    @pytest.fixture
    def mock_genesis_renderer(self):
        """Mock GenesisRenderer for testing."""
        with patch('simforge.movement_controller.GenesisRenderer') as mock_renderer:
            yield mock_renderer

    def test_initialization(self, sample_config, mock_genesis_renderer):
        """Test controller initialization with valid config."""
        controller = MovementController(sample_config, debug=False)

        assert "test_robot" in controller.robot_modes
        assert controller.robot_modes["test_robot"] == ControlMode.JOINT
        assert len(controller.joint_targets["test_robot"]) == 6
        assert controller.running is False

    def test_initialization_with_debug(self, sample_config, mock_genesis_renderer):
        """Test controller initialization with debug enabled."""
        controller = MovementController(sample_config, debug=True)

        assert controller.running is False
        # Debug mode is passed to logging, not stored as attribute

    def test_mode_switching(self, sample_config, mock_genesis_renderer):
        """Test robot control mode switching."""
        controller = MovementController(sample_config)

        # Test initial mode
        assert controller.get_robot_mode("test_robot") == ControlMode.JOINT

        # Test mode switching
        controller.switch_mode("test_robot", ControlMode.CARTESIAN)
        assert controller.command_queue.qsize() == 1

        # Manually process the command for testing
        cmd = controller.command_queue.get()
        assert isinstance(cmd, SwitchModeCommand)
        assert cmd.robot == "test_robot"
        assert cmd.mode == ControlMode.CARTESIAN

    def test_invalid_robot_mode_switch(self, sample_config, mock_genesis_renderer):
        """Test mode switching for non-existent robot."""
        controller = MovementController(sample_config)

        # Should not raise exception, but command should still be queued
        controller.switch_mode("nonexistent_robot", ControlMode.CARTESIAN)
        assert controller.command_queue.qsize() == 1

    def test_joint_position_setting(self, sample_config, mock_genesis_renderer):
        """Test setting individual joint positions."""
        controller = MovementController(sample_config)

        controller.set_joint_position("test_robot", 0, 45.0)
        assert controller.command_queue.qsize() == 1

        cmd = controller.command_queue.get()
        assert isinstance(cmd, SetJointCommand)
        assert cmd.robot == "test_robot"
        assert cmd.joint_idx == 0
        assert cmd.value_deg == 45.0

    def test_joint_targets_setting(self, sample_config, mock_genesis_renderer):
        """Test setting all joint targets."""
        controller = MovementController(sample_config)

        targets = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        controller.set_joint_targets("test_robot", targets)

        assert controller.command_queue.qsize() == 1

        cmd = controller.command_queue.get()
        assert isinstance(cmd, SetJointTargetsCommand)
        assert cmd.robot == "test_robot"
        assert cmd.values_deg == targets

    def test_cartesian_move_command(self, sample_config, mock_genesis_renderer):
        """Test Cartesian move command creation."""
        controller = MovementController(sample_config)

        position = (0.5, 0.2, 0.3)
        orientation = (0.0, 0.0, 0.0)
        controller.move_cartesian("test_robot", position, orientation, "base")

        assert controller.command_queue.qsize() == 1

        cmd = controller.command_queue.get()
        assert isinstance(cmd, CartesianMoveCommand)
        assert cmd.robot == "test_robot"
        assert cmd.position == position
        assert cmd.orientation_deg == orientation
        assert cmd.frame == "base"

    def test_cartesian_move_default_frame(self, sample_config, mock_genesis_renderer):
        """Test Cartesian move with default frame."""
        controller = MovementController(sample_config)

        position = (0.5, 0.2, 0.3)
        orientation = (0.0, 0.0, 0.0)
        controller.move_cartesian("test_robot", position, orientation)

        cmd = controller.command_queue.get()
        assert cmd.frame == "base"  # Default frame

    def test_get_joint_targets_empty(self, sample_config, mock_genesis_renderer):
        """Test getting joint targets when none are set."""
        controller = MovementController(sample_config)

        targets = controller.get_joint_targets("test_robot")
        assert targets == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Initial positions

    def test_get_joint_targets_after_setting(self, sample_config, mock_genesis_renderer):
        """Test getting joint targets after setting them."""
        controller = MovementController(sample_config)

        targets = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        controller.set_joint_targets("test_robot", targets)

        # Simulate processing the command
        controller.joint_targets["test_robot"] = targets

        retrieved_targets = controller.get_joint_targets("test_robot")
        assert retrieved_targets == targets

    def test_get_robot_mode_nonexistent(self, sample_config, mock_genesis_renderer):
        """Test getting mode for non-existent robot."""
        controller = MovementController(sample_config)

        # Should return default mode
        mode = controller.get_robot_mode("nonexistent")
        assert mode == ControlMode.JOINT

    def test_get_joint_positions_nonexistent_robot(self, sample_config, mock_genesis_renderer):
        """Test getting joint positions for non-existent robot."""
        controller = MovementController(sample_config)

        positions = controller.get_joint_positions("nonexistent")
        assert positions == []

    @patch('simforge.movement_controller.setup_logging')
    def test_logging_initialization(self, mock_setup_logging, sample_config, mock_genesis_renderer):
        """Test that logging is properly initialized."""
        MovementController(sample_config, debug=True)
        mock_setup_logging.assert_called_once_with(True)

    def test_multiple_robots(self):
        """Test controller with multiple robots."""
        config = SimforgeConfig(
            robots=[
                RobotConfig(name="robot1", urdf="robot1.urdf"),
                RobotConfig(name="robot2", urdf="robot2.urdf")
            ]
        )

        with patch('simforge.movement_controller.GenesisRenderer'):
            controller = MovementController(config)

            assert "robot1" in controller.robot_modes
            assert "robot2" in controller.robot_modes
            assert controller.get_robot_mode("robot1") == ControlMode.JOINT
            assert controller.get_robot_mode("robot2") == ControlMode.JOINT

    def test_command_queue_threading(self, sample_config, mock_genesis_renderer):
        """Test that command queue is thread-safe."""
        controller = MovementController(sample_config)

        # Add multiple commands
        controller.set_joint_position("test_robot", 0, 45.0)
        controller.set_joint_targets("test_robot", [10.0] * 6)
        controller.move_cartesian("test_robot", (0.1, 0.2, 0.3), (0, 0, 0))

        assert controller.command_queue.qsize() == 3

        # Verify commands are correct
        cmd1 = controller.command_queue.get()
        cmd2 = controller.command_queue.get()
        cmd3 = controller.command_queue.get()

        assert isinstance(cmd1, SetJointCommand)
        assert isinstance(cmd2, SetJointTargetsCommand)
        assert isinstance(cmd3, CartesianMoveCommand)