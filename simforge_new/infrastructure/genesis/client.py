"""Genesis client wrapper used by the new control stack."""
from __future__ import annotations

import logging
from typing import Any, Optional

from ...core import Backend


class GenesisClient:
    """Thin wrapper around ``genesis-world`` initialisation and utilities."""

    def __init__(self, backend: Backend, logger: Optional[logging.Logger] = None) -> None:
        self.backend = backend
        self.logger = logger or logging.getLogger("simforge.genesis")
        self._gs = self._import_genesis()
        self._initialise_backend()
        self._destroyed = False

    @property
    def gs(self):
        return self._gs

    def _import_genesis(self) -> Any:
        try:
            import genesis as gs  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("Genesis not installed. Run `pip install genesis-world`." ) from exc
        return gs

    def _initialise_backend(self) -> None:
        backend_map = {
            Backend.GPU: self._gs.gpu,
            Backend.CUDA: self._gs.cuda,
            Backend.CPU: self._gs.cpu,
        }
        backend = backend_map.get(self.backend, self._gs.gpu)
        
        # Configure Genesis logging to reduce verbosity and avoid double logging
        genesis_logger = logging.getLogger("genesis")
        genesis_logger.setLevel(logging.WARNING)  # Suppress verbose info logs
        genesis_logger.propagate = False  # Prevent propagation to root logger
        
        try:
            self._gs.init(backend=backend)
            self.logger.info("Genesis initialised with backend: %s", self.backend.value)
        except Exception as exc:  # pragma: no cover - varies with GS internals
            if "already initialised" in str(exc).lower():
                self.logger.debug("Genesis already initialised")
            else:
                raise

    def destroy(self) -> None:
        """Shut down Genesis explicitly to avoid at-exit teardown issues."""
        if self._destroyed:
            return
        try:
            destroy = getattr(self._gs, "destroy", None)
            if callable(destroy):
                destroy()
                self.logger.info("Genesis shutdown complete")
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Genesis destroy reported: %s", exc)
        finally:
            self._destroyed = True

    def create_scene(
        self,
        dt: float,
        gravity: tuple[float, float, float],
        show_viewer: bool,
        max_fps: int,
    ) -> tuple[Any, bool]:
        """Create a Genesis scene.
        
        Returns:
            Tuple of (scene, viewer_active) where viewer_active indicates if
            the viewer was successfully initialized.
        """
        viewer_active = show_viewer
        
        scene = self._gs.Scene(
            sim_options=self._gs.options.SimOptions(dt=dt, gravity=gravity),
            viewer_options=self._gs.options.ViewerOptions(
                camera_pos=(3.0, 0.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                max_FPS=max_fps,
            ),
            show_viewer=show_viewer,
        )
        try:
            profiling = getattr(scene, "profiling_options", None)
            if profiling and hasattr(profiling, "show_FPS"):
                profiling.show_FPS = False
        except Exception:  # pragma: no cover
            pass
        return scene, viewer_active
    
    def check_viewer_health(self, scene: Any) -> bool:
        """Check if the viewer is still healthy and running.
        
        Returns True if viewer is active, False if closed or never started.
        """
        try:
            viewer = getattr(scene, "viewer", None)
            if viewer is None:
                return False
            # Check if viewer is running
            is_running = getattr(viewer, "is_running", None)
            if callable(is_running):
                return is_running()
            # Fallback: check for viewer attributes
            return hasattr(viewer, "_app") and viewer._app is not None
        except Exception:
            return False


__all__ = ["GenesisClient"]
