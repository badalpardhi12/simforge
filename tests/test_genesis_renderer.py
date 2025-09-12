import pytest
from unittest.mock import Mock, patch
from simforge.genesis_renderer import GenesisRenderer


def test_genesis_renderer_initialization():
    """Test Genesis renderer initialization."""
    logger = Mock()
    
    with patch('simforge.genesis_renderer.GenesisRenderer._import_genesis') as mock_import:
        mock_gs = Mock()
        mock_import.return_value = mock_gs
        
        renderer = GenesisRenderer("gpu", logger)
        
        assert renderer.backend == "gpu"
        assert renderer.logger == logger
        mock_import.assert_called_once()


def test_backend_selection():
    """Test backend selection logic."""
    logger = Mock()
    
    with patch('simforge.genesis_renderer.GenesisRenderer._import_genesis') as mock_import:
        mock_gs = Mock()
        mock_gs.gpu = "gpu_backend"
        mock_gs.cpu = "cpu_backend" 
        mock_gs.cuda = "cuda_backend"
        mock_import.return_value = mock_gs
        
        # Test different backends
        for backend in ["gpu", "cpu", "cuda"]:
            renderer = GenesisRenderer(backend, logger)
            assert renderer.backend == backend


def test_scene_creation():
    """Test scene creation."""
    logger = Mock()
    
    with patch('simforge.genesis_renderer.GenesisRenderer._import_genesis') as mock_import:
        mock_gs = Mock()
        mock_scene = Mock()
        mock_gs.Scene.return_value = mock_scene
        mock_import.return_value = mock_gs
        
        renderer = GenesisRenderer("gpu", logger)
        scene = renderer.create_scene(
            dt=0.01, 
            gravity=(0, 0, -9.81), 
            show_viewer=True, 
            max_fps=60
        )
        
        assert scene == mock_scene
        mock_gs.Scene.assert_called_once()


def test_attribute_delegation():
    """Test attribute delegation to Genesis module."""
    logger = Mock()
    
    with patch('simforge.genesis_renderer.GenesisRenderer._import_genesis') as mock_import:
        mock_gs = Mock()
        mock_gs.some_attribute = "test_value"
        mock_import.return_value = mock_gs
        
        renderer = GenesisRenderer("gpu", logger)
        
        assert renderer.some_attribute == "test_value"