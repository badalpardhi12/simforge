"""Tests for configuration reader module."""

import pytest
from simforge.config_reader import SimforgeConfig, RobotConfig, ControlConfig, SceneConfig


class TestSimforgeConfig:
    """Test SimforgeConfig class functionality."""

    def test_from_yaml_minimal_config(self, temp_config_file):
        """Test loading a minimal configuration."""
        config_content = """
        robots:
          - name: test_robot
            urdf: test.urdf
        """
        config_file = temp_config_file(config_content)
        cfg = SimforgeConfig.from_yaml(config_file)

        assert len(cfg.robots) == 1
        assert cfg.robots[0].name == "test_robot"
        assert cfg.robots[0].urdf == "test.urdf"
        assert cfg.scene.dt == 0.01  # default value

    def test_control_override(self, temp_config_file):
        """Test that robot-specific control settings override global ones."""
        config_content = """
        control:
          joint_speed_limit: 2.0
          cartesian_speed_limit: 0.2
        robots:
          - name: robot1
            urdf: test.urdf
            control:
              joint_speed_limit: 3.0
          - name: robot2
            urdf: test2.urdf
        """
        config_file = temp_config_file(config_content)
        cfg = SimforgeConfig.from_yaml(config_file)

        # Global control settings
        assert cfg.control.joint_speed_limit == 2.0
        assert cfg.control.cartesian_speed_limit == 0.2

        # Robot-specific override
        assert cfg.control_for("robot1").joint_speed_limit == 3.0
        assert cfg.control_for("robot1").cartesian_speed_limit == 0.2  # inherits global

        # Robot without override inherits all global
        assert cfg.control_for("robot2").joint_speed_limit == 2.0
        assert cfg.control_for("robot2").cartesian_speed_limit == 0.2

    def test_robot_with_pose(self, temp_config_file):
        """Test robot configuration with pose settings."""
        config_content = """
        robots:
          - name: robot1
            urdf: test.urdf
            pose:
              position: [1.0, 2.0, 3.0]
              rpy: [0.1, 0.2, 0.3]
        """
        config_file = temp_config_file(config_content)
        cfg = SimforgeConfig.from_yaml(config_file)

        robot = cfg.robots[0]
        assert robot.name == "robot1"
        assert robot.base_position == (1.0, 2.0, 3.0)
        assert robot.base_orientation == (0.1, 0.2, 0.3)

    def test_scene_config_defaults(self):
        """Test SceneConfig default values."""
        scene = SceneConfig()
        assert scene.dt == 0.01
        assert scene.gravity == (0.0, 0.0, -9.81)
        assert scene.backend == "gpu"
        assert scene.show_viewer is True
        assert scene.max_fps == 60

    def test_control_config_defaults(self):
        """Test ControlConfig default values."""
        control = ControlConfig()
        assert control.joint_speed_limit == 1.0
        assert control.cartesian_speed_limit == 0.1
        assert control.planner == "RRTConnect"
        assert control.collision_check is True
        assert control.self_collision_check is True

    def test_robot_config_defaults(self):
        """Test RobotConfig default values."""
        robot = RobotConfig(name="test", urdf="test.urdf")
        assert robot.name == "test"
        assert robot.urdf == "test.urdf"
        assert robot.base_position == (0.0, 0.0, 0.0)
        assert robot.base_orientation == (0.0, 0.0, 0.0)
        assert robot.fixed_base is True
        assert robot.initial_joint_positions is None
        assert robot.end_effector_link is None

    @pytest.mark.parametrize("backend", ["cpu", "gpu", "cuda"])
    def test_scene_backend_validation(self, backend):
        """Test that scene backend accepts valid values."""
        scene = SceneConfig(backend=backend)
        assert scene.backend == backend

    def test_yaml_include_robots(self, temp_config_file, tmp_path):
        """Test including robots from separate YAML files."""
        # Create included file
        included_content = """
        robots:
          - name: included_robot
            urdf: included.urdf
        """
        included_file = tmp_path / "included.yaml"
        included_file.write_text(included_content)

        # Create main config that includes the file
        main_content = f"""
        robots:
          - {included_file}
          - name: main_robot
            urdf: main.urdf
        """
        main_file = temp_config_file(main_content)
        cfg = SimforgeConfig.from_yaml(main_file)

        assert len(cfg.robots) == 2
        robot_names = [r.name for r in cfg.robots]
        assert "included_robot" in robot_names
        assert "main_robot" in robot_names

    def test_invalid_yaml_raises_error(self, tmp_path):
        """Test that invalid YAML raises an appropriate error."""
        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text("invalid: yaml: content: [")

        with pytest.raises(Exception):  # YAML parsing error
            SimforgeConfig.from_yaml(invalid_file)

    def test_empty_config_defaults(self, temp_config_file):
        """Test that empty config uses all defaults."""
        config_content = """
        robots: []
        """
        config_file = temp_config_file(config_content)
        cfg = SimforgeConfig.from_yaml(config_file)

        assert len(cfg.robots) == 0
        assert cfg.scene.dt == 0.01
        assert cfg.control.joint_speed_limit == 1.0
