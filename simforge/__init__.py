__all__ = [
    "__version__",
]

__version__ = "0.1.0"
# Ensure PyBullet URDF warnings are suppressed before any submodule imports pybullet
import os as _os
_os.environ.setdefault("PYBULLET_SUPPRESS_URDF_WARNINGS", "1")

