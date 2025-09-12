"""Genesis renderer wrapper.

Clean Genesis backend initialization without legacy patching.
"""
from __future__ import annotations

import logging
from typing import Any


class GenesisRenderer:
    """Minimal Genesis wrapper for scene creation."""
    
    def __init__(self, backend: str, logger: logging.Logger):
        self.backend = backend.lower()
        self.logger = logger
        self.gs = self._import_genesis()
        self._init_backend()

    def _import_genesis(self) -> Any:
        try:
            import genesis as gs
            return gs
        except ImportError as e:
            raise RuntimeError("Genesis not installed. Run: pip install genesis-world") from e

    def _init_backend(self):
        backend_map = {"gpu": self.gs.gpu, "cuda": self.gs.cuda, "cpu": self.gs.cpu}
        backend = backend_map.get(self.backend, self.gs.gpu)
        try:
            self.gs.init(backend=backend)
            self.logger.info(f"Genesis initialized with backend: {self.backend}")
        except Exception as e:
            if "already initialized" in str(e).lower():
                self.logger.debug("Genesis already initialized")
            else:
                raise

    def create_scene(self, dt: float, gravity: tuple, show_viewer: bool, max_fps: int) -> Any:
        scene = self.gs.Scene(
            sim_options=self.gs.options.SimOptions(dt=dt, gravity=gravity),
            viewer_options=self.gs.options.ViewerOptions(
                camera_pos=(3.0, 0.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                max_FPS=max_fps,
            ),
            show_viewer=show_viewer,
        )
        # Suppress on-screen profiling / FPS overlay if available
        try:
            if hasattr(scene, 'profiling_options') and hasattr(scene.profiling_options, 'show_FPS'):
                scene.profiling_options.show_FPS = False
        except Exception:
            pass
        return scene

    def __getattr__(self, name: str) -> Any:
        return getattr(self.gs, name)


__all__ = ["GenesisRenderer"]