"""Namespace package for the Simforge modules."""

from __future__ import annotations


def main() -> None:
	"""Entry point that defers importing the CLI until needed."""

	from .interfaces.cli import main as _cli_main

	_cli_main()


__all__ = ["main"]
