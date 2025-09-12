import pytest
from unittest.mock import Mock, patch
from simforge.collision_checker import CollisionChecker, HAS_FCL, HAS_TRIMESH


def test_collision_checker_initialization():
    """Test collision checker initialization."""
    logger = Mock()
    
    checker = CollisionChecker("test.urdf", logger)
    
    assert checker.urdf_path.name == "test.urdf"
    assert checker.ground_plane_z == 0.0


@pytest.mark.skipif(not (HAS_FCL and HAS_TRIMESH), reason="FCL or trimesh not available")
def test_collision_checker_with_dependencies():
    """Test collision checker when dependencies are available."""
    logger = Mock()
    
    with patch('simforge.collision_checker.Path') as mock_path, \
         patch('xml.etree.ElementTree.parse') as mock_parse:
        
        mock_path.return_value.exists.return_value = True
        mock_tree = Mock()
        mock_root = Mock()
        mock_tree.getroot.return_value = mock_root
        mock_root.findall.return_value = []
        mock_parse.return_value = mock_tree
        
        checker = CollisionChecker("test.urdf", logger)
        assert checker.available == True


def test_ground_collision_check():
    """Test ground plane collision detection."""
    logger = Mock()
    checker = CollisionChecker("test.urdf", logger)
    
    # Test poses
    poses_above = {"link1": ((0, 0, 1), (1, 0, 0, 0))}
    poses_below = {"link1": ((0, 0, -1), (1, 0, 0, 0))}
    
    assert not checker.check_ground_collision(poses_above)
    assert checker.check_ground_collision(poses_below)