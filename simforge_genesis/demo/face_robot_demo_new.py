#!/usr/bin/env python3
"""Compatibility wrapper for the packaged face robot demo."""

from __future__ import annotations

from simforge_genesis.demo import run_demo


def main() -> None:
    run_demo("face_robot")


if __name__ == "__main__":
    main()