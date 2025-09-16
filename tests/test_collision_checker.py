"""Tests for collision checker module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from simforge.collision_checker import CollisionChecker, HAS_FCL, HAS_TRIMESH


class TestCollisionChecker:
    """Test CollisionChecker class functionality."""

    def test_initialization_without_dependencies(self, mock_logger):
        """Test collision checker initialization when FCL/trimesh not available."""
        with patch('simforge.collision_checker.HAS_FCL', False):
            checker = CollisionChecker("test.urdf", mock_logger)
            assert not checker.available
            mock_logger.warning.assert_called_once()

    def test_initialization_with_dependencies(self, mock_logger):
        """Test collision checker initialization when dependencies are available."""
        with patch('simforge.collision_checker.HAS_FCL', True), \
             patch('simforge.collision_checker.HAS_TRIMESH', True):
            checker = CollisionChecker("test.urdf", mock_logger)
            assert checker.available
            assert checker.urdf_path.name == "test.urdf"
            assert checker.ground_plane_z == 0.0

    def test_base_transform_initialization(self, mock_logger):
        """Test that base transform is properly initialized."""
        with patch('simforge.collision_checker.HAS_FCL', True), \
             patch('simforge.collision_checker.HAS_TRIMESH', True):
            checker = CollisionChecker(
                "test.urdf",
                mock_logger,
                base_position=(1.0, 2.0, 3.0),
                base_orientation_rpy=(0.1, 0.2, 0.3)
            )
            assert checker._base_t.tolist() == [1.0, 2.0, 3.0]
            # Check that rotation matrix has correct shape
            assert checker._base_R.shape == (3, 3)

    def test_allowed_pairs_configuration(self, mock_logger):
        """Test configuration of allowed collision pairs."""
        allowed_pairs = [("link1", "link2"), ("link3", "link4")]
        with patch('simforge.collision_checker.HAS_FCL', True), \
             patch('simforge.collision_checker.HAS_TRIMESH', True):
            checker = CollisionChecker(
                "test.urdf",
                mock_logger,
                allowed_link_pairs=allowed_pairs
            )
            # Pairs are stored in sorted order
            assert ("link1", "link2") in checker.allowed_link_pairs
            assert ("link3", "link4") in checker.allowed_link_pairs
            # Should have exactly 2 pairs
            assert len(checker.allowed_link_pairs) == 2

    def test_world_allowed_pairs(self, mock_logger):
        """Test configuration of world allowed pairs."""
        world_pairs = [("robot:link1", "obj:table")]
        with patch('simforge.collision_checker.HAS_FCL', True), \
             patch('simforge.collision_checker.HAS_TRIMESH', True):
            checker = CollisionChecker(
                "test.urdf",
                mock_logger,
                world_allowed_pairs=world_pairs
            )
            assert ("robot:link1", "obj:table") in checker.allowed_world_pairs

    def test_ground_collision_detection(self, mock_logger):
        """Test ground plane collision detection."""
        checker = CollisionChecker("test.urdf", mock_logger)

        # Test poses above ground
        poses_above = {
            "link1": ((0, 0, 1), (1, 0, 0, 0)),
            "link2": ((1, 1, 0.5), (1, 0, 0, 0))
        }
        assert not checker.check_ground_collision(poses_above)

        # Test poses below ground
        poses_below = {
            "link1": ((0, 0, -0.1), (1, 0, 0, 0)),
            "link2": ((1, 1, 0.5), (1, 0, 0, 0))
        }
        assert checker.check_ground_collision(poses_below)

        # Test poses at ground level
        poses_at_ground = {
            "link1": ((0, 0, 0.0), (1, 0, 0, 0))
        }
        assert not checker.check_ground_collision(poses_at_ground)

    def test_ground_collision_with_custom_plane(self, mock_logger):
        """Test ground collision with custom ground plane Z."""
        checker = CollisionChecker("test.urdf", mock_logger, ground_plane_z=0.5)

        poses_above = {"link1": ((0, 0, 1.0), (1, 0, 0, 0))}
        poses_below = {"link1": ((0, 0, 0.3), (1, 0, 0, 0))}

        assert not checker.check_ground_collision(poses_above)
        assert checker.check_ground_collision(poses_below)

    @pytest.mark.skipif(not (HAS_FCL and HAS_TRIMESH), reason="FCL or trimesh not available")
    def test_urdf_loading_with_mock(self, mock_logger):
        """Test URDF loading with mocked dependencies."""
        mock_tree = MagicMock()
        mock_root = MagicMock()
        mock_tree.getroot.return_value = mock_root
        mock_root.findall.return_value = []  # No joints

        with patch('xml.etree.ElementTree.parse', return_value=mock_tree), \
             patch('simforge.collision_checker.Path') as mock_path_class:

            mock_path_instance = MagicMock()
            mock_path_instance.parent = MagicMock()
            mock_path_class.return_value = mock_path_instance

            checker = CollisionChecker("test.urdf", mock_logger)
            # Should not raise any exceptions

    def test_mesh_shrink_parameter(self, mock_logger):
        """Test collision mesh shrink parameter."""
        with patch('simforge.collision_checker.HAS_FCL', True), \
             patch('simforge.collision_checker.HAS_TRIMESH', True):
            checker = CollisionChecker("test.urdf", mock_logger, collision_mesh_shrink=0.8)
            assert checker._shrink == 0.8

    def test_world_objects_registration(self, mock_logger):
        """Test registration of world collision objects."""
        world_boxes = [
            ("table", (0.5, 0.5, 0.5), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        ]

        with patch('simforge.collision_checker.HAS_FCL', True), \
             patch('simforge.collision_checker.HAS_TRIMESH', True), \
             patch.object(CollisionChecker, '_load_urdf'):

            checker = CollisionChecker("test.urdf", mock_logger, world_boxes=world_boxes)
            assert "obj:table" in checker.env_objs  # Objects are prefixed with "obj:"

    def test_env_robot_registration(self, mock_logger):
        """Test registration of environment robots."""
        with patch('simforge.collision_checker.HAS_FCL', True), \
             patch('simforge.collision_checker.HAS_TRIMESH', True), \
             patch.object(CollisionChecker, '_load_urdf'), \
             patch.object(CollisionChecker, '_build_geoms_from_urdf') as mock_build:

            mock_build.return_value = {"link1": []}
            checker = CollisionChecker("test.urdf", mock_logger)
            checker.register_env_robot("robot1", "robot1.urdf")

            assert "robot1" in checker.env_robot_geoms
            mock_build.assert_called_once_with("robot1.urdf")