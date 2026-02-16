"""Module entry point so that ``python -m simforge_genesis`` dispatches to the CLI."""
from __future__ import annotations

from .interfaces.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
