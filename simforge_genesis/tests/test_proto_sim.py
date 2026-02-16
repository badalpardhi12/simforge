"""Placeholder proto_sim tests.

The proto_sim subsystem depends on native Genesis components that trigger
teardown crashes in the current execution environment.  A real integration
suite should exercise ``generate_proto_poses`` and ``execute_proto_sim`` once
Genesis is available.  For now we skip the module to avoid spurious failures.
"""
from __future__ import annotations

import pytest

pytest.skip(
    "proto_sim integration tests require a stable Genesis runtime; skipping in this environment",
    allow_module_level=True,
)
