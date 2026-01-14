#!/usr/bin/env python3
"""Launch the face robot preset using the new simforge_new control stack."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from simforge_new.interfaces.cli import main as cli_main


def _build_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the face robot demo via proto_sim")
    parser.add_argument(
        "--config",
        default=Path("simforge_new/environment/presets/face_robot.yaml"),
        type=Path,
        help="Environment preset to load",
    )
    parser.add_argument(
        "--backend",
        default="gpu",
        choices=["gpu", "cpu", "cuda"],
        help="Genesis backend to use",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_args(argv)
    cli_main([
        "proto_sim",
        "--config",
        str(args.config.resolve()),
        "--backend",
        args.backend,
    ])


if __name__ == "__main__":
    main()
