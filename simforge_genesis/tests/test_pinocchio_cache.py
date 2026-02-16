from __future__ import annotations

from pathlib import Path

import pytest

from simforge_genesis.infrastructure.pinocchio_cache import (
    PinocchioModelCache,
    get_model_bundle,
    is_available,
)


def _sample_urdf() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "assets" / "ur5e" / "ur5e.urdf"


@pytest.mark.skipif(not is_available(), reason="Pinocchio not installed")
def test_shared_cache_returns_same_bundle():
    urdf = _sample_urdf()
    bundle1 = get_model_bundle(urdf)
    bundle2 = get_model_bundle(urdf)

    assert bundle1 is not None
    assert bundle2 is not None
    assert bundle1 is bundle2
    assert bundle1.model is bundle2.model

    data_first = bundle1.create_data()
    data_second = bundle1.create_data()
    assert data_first is not data_second


@pytest.mark.skipif(not is_available(), reason="Pinocchio not installed")
def test_independent_cache_force_reload(tmp_path):
    urdf = _sample_urdf()
    cache = PinocchioModelCache()

    bundle_initial = cache.get(urdf)
    bundle_reloaded = cache.get(urdf, force_reload=True)

    assert bundle_initial is not None
    assert bundle_reloaded is not None
    assert bundle_initial is not bundle_reloaded
    assert bundle_initial.model is not bundle_reloaded.model
