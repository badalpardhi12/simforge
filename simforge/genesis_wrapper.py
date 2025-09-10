from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from .config import SceneConfig


class GenesisWrapper:
    """
    A wrapper for the Genesis simulator backend to abstract away direct calls and
    handle different backend initializations (GPU, CPU, CUDA).
    """

    def __init__(self, backend_cfg: str, logger: logging.Logger):
        self.backend_cfg = backend_cfg.lower()
        self.logger = logger
        self.gs = self._import_genesis()
        self._init_backend()

    def _import_genesis(self) -> Any:
        """Import and return the Genesis module."""
        try:
            import genesis as gs
            return gs
        except ImportError as e:
            raise RuntimeError(
                "Genesis is not installed. Please `pip install genesis-world`"
            ) from e

    def _init_backend(self):
        """Initialize the Genesis backend based on the configuration."""
        backend_map = {
            "gpu": self.gs.gpu,
            "cuda": self.gs.cuda,
            "cpu": self.gs.cpu,
        }
        backend = backend_map.get(self.backend_cfg)
        if backend is None:
            self.logger.warning(
                f"Unknown backend '{self.backend_cfg}', defaulting to 'gpu'."
            )
            backend = self.gs.gpu

        self.gs.init(backend=backend)
        self.logger.info(f"Genesis initialized with backend: {self.backend_cfg}")

    def create_scene(self, scene_config: SceneConfig) -> Any:
        """Create a new Genesis scene."""
        return self.gs.Scene(
            sim_options=self.gs.options.SimOptions(
                dt=scene_config.dt, gravity=tuple(scene_config.gravity)
            ),
            viewer_options=self.gs.options.ViewerOptions(
                camera_pos=(3.0, 0.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                max_FPS=scene_config.max_fps,
            ),
            show_viewer=scene_config.show_viewer,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying Genesis module."""
        return getattr(self.gs, name)
