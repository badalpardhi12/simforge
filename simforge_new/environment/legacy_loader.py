"""Legacy loader removed.

This module previously offered helpers for transforming v1 environment
configurations. Legacy support has been discontinued; importing this module
now raises immediately to highlight the breaking change.
"""

raise ImportError(
    "simforge_new.environment.legacy_loader has been removed. "
    "Use simforge_new.environment.load_environment with v2 presets instead."
)
