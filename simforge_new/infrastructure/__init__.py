"""Infrastructure adapters for external dependencies."""

from .genesis import (
    GenesisClient,
    SceneBuildResult,
    build_scene,
    get_joint_positions,
    set_joint_positions,
)
from .pinocchio_cache import (
    PinocchioModelBundle,
    PinocchioModelCache,
    get_model_bundle,
    import_error as pinocchio_import_error,
    is_available as is_pinocchio_available,
)

__all__ = [
    "GenesisClient",
    "SceneBuildResult",
    "build_scene",
    "get_joint_positions",
    "set_joint_positions",
    "PinocchioModelBundle",
    "PinocchioModelCache",
    "get_model_bundle",
    "pinocchio_import_error",
    "is_pinocchio_available",
]
