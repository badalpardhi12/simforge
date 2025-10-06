"""Shared Pinocchio model cache used across Simforge services."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency import
    import pinocchio as _pin  # type: ignore
except Exception as exc:  # pragma: no cover - import failure path
    _pin = None  # type: ignore
    _PIN_AVAILABLE = False
    _PIN_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _PIN_AVAILABLE = True
    _PIN_IMPORT_ERROR = None


def _canonical_package_dirs(package_dirs: Optional[Sequence[str | Path]]) -> Tuple[str, ...]:
    if not package_dirs:
        return ()
    canonicalised = []
    for entry in package_dirs:
        canonicalised.append(str(Path(entry).expanduser().resolve()))
    return tuple(sorted(set(canonicalised)))


@dataclass(frozen=True)
class PinocchioModelBundle:
    """Thin wrapper around a Pinocchio model and metadata."""

    model: Any
    urdf_path: Path
    package_dirs: Tuple[str, ...]
    mtime_ns: int

    def create_data(self) -> Any:
        """Return a fresh :class:`pinocchio.pinocchio.Model` data instance."""
        return self.model.createData()

    @property
    def nq(self) -> int:
        return int(getattr(self.model, "nq", 0))


class PinocchioModelCache:
    """Thread-safe cache of Pinocchio models keyed by URDF path."""

    def __init__(self) -> None:
        self._entries: Dict[Tuple[str, Tuple[str, ...]], PinocchioModelBundle] = {}
        self._lock = threading.RLock()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def get(
        self,
        urdf_path: str | Path,
        *,
        package_dirs: Optional[Sequence[str | Path]] = None,
        force_reload: bool = False,
    ) -> Optional[PinocchioModelBundle]:
        if not _PIN_AVAILABLE or _pin is None:
            return None

        path = Path(urdf_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"URDF not found: {path}")

        key = (str(path), _canonical_package_dirs(package_dirs))
        mtime_ns = path.stat().st_mtime_ns

        with self._lock:
            bundle = self._entries.get(key)
            if bundle and not force_reload and bundle.mtime_ns == mtime_ns:
                return bundle

            # Pinocchio expects search paths to be configured via environment variables; for now
            # we rely on the fully-resolved URDF path that already incorporates any tooling merges.
            model = _pin.buildModelFromUrdf(str(path))
            bundle = PinocchioModelBundle(model=model, urdf_path=path, package_dirs=key[1], mtime_ns=mtime_ns)
            self._entries[key] = bundle
            return bundle


_CACHE = PinocchioModelCache()


def is_available() -> bool:
    """Return ``True`` when Pinocchio was imported successfully."""
    return _PIN_AVAILABLE


def import_error() -> Optional[BaseException]:
    """Return the exception raised during Pinocchio import, if any."""
    return _PIN_IMPORT_ERROR


def get_model_bundle(
    urdf_path: str | Path,
    *,
    package_dirs: Optional[Sequence[str | Path]] = None,
    force_reload: bool = False,
) -> Optional[PinocchioModelBundle]:
    """Convenience accessor for the shared cache."""
    return _CACHE.get(urdf_path, package_dirs=package_dirs, force_reload=force_reload)


__all__ = [
    "PinocchioModelBundle",
    "PinocchioModelCache",
    "get_model_bundle",
    "import_error",
    "is_available",
]
