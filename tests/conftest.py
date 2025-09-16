"""Pytest configuration and shared fixtures for Simforge tests."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from simforge.config_reader import SimforgeConfig


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""
    def _create_config(content: str, filename: str = "test_config.yaml") -> Path:
        config_file = tmp_path / filename
        config_file.write_text(content)
        return config_file
    return _create_config


@pytest.fixture
def minimal_config(temp_config_file):
    """Create a minimal valid configuration."""
    config_content = """
scene:
  dt: 0.01
  backend: cpu
  show_viewer: false

robots:
  - name: test_robot
    urdf: test.urdf
    base_position: [0.0, 0.0, 0.0]
    end_effector_link: ee_link

control:
  joint_speed_limit: 1.0
  cartesian_speed_limit: 0.1
"""
    return temp_config_file(config_content)


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock()
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


@pytest.fixture
def sample_simforge_config():
    """Create a sample SimforgeConfig object."""
    config = SimforgeConfig(
        scene={"dt": 0.01, "backend": "cpu", "show_viewer": False},
        robots=[{
            "name": "test_robot",
            "urdf": "test.urdf",
            "base_position": [0.0, 0.0, 0.0],
            "end_effector_link": "ee_link"
        }],
        control={"joint_speed_limit": 1.0, "cartesian_speed_limit": 0.1}
    )
    return config