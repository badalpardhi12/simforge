from simforge.config_reader import SimforgeConfig, RobotConfig, ControlConfig, SceneConfig
import tempfile
import textwrap


def test_load_minimal_config(tmp_path):
    cfg_yaml = tmp_path / "env.yaml"
    cfg_yaml.write_text(textwrap.dedent(
        """
        robots:
          - name: r1
            urdf: some.urdf
        """
    ))
    cfg = SimforgeConfig.from_yaml(cfg_yaml)
    assert cfg.robots[0].name == "r1"
    assert cfg.scene.dt == 0.01


def test_control_override(tmp_path):
    cfg_yaml = tmp_path / "env.yaml"
    cfg_yaml.write_text(textwrap.dedent(
        """
        control:
          joint_speed_limit: 2.0
        robots:
          - name: r1
            urdf: a.urdf
            control:
              joint_speed_limit: 3.0
        """
    ))
    cfg = SimforgeConfig.from_yaml(cfg_yaml)
    assert cfg.control.joint_speed_limit == 2.0
    assert cfg.control_for("r1").joint_speed_limit == 3.0
