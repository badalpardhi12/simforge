"""Command line interface for Simforge."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from .gui import HAS_WX, run_gui, run_proto_sim_gui
from ..core import Backend
from ..demo import available_demos, run_demo
from ..logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
    prog="simforge_new",
    description="Headless control interface for the Simforge stack",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_cmd = sub.add_parser("init", help="Copy preset environments to the current directory")
    init_cmd.add_argument(
        "--destination",
        default="env_presets",
        help="Target directory for preset files (default: %(default)s)",
    )
    init_cmd.set_defaults(func=_cmd_init)

    run_cmd = sub.add_parser("run", help="Launch the interactive GUI controller")
    run_cmd.add_argument("--config", required=True, help="Path to an environment configuration")
    run_cmd.add_argument(
        "--backend",
        default="gpu",
        choices=[member.value for member in Backend],
        help="Genesis backend to use",
    )
    run_cmd.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: %(default)s)",
    )
    run_cmd.add_argument(
        "--debug",
        action="store_true",
        help="Enable extra GUI logging and diagnostics",
    )
    run_cmd.set_defaults(func=_cmd_run)

    demo_cmd = sub.add_parser("demo", help="Run a packaged Simforge demo")
    demo_cmd.add_argument("name", nargs="?", help="Name of the demo to run")
    demo_cmd.add_argument(
        "--list",
        action="store_true",
        help="List available demos and exit",
    )
    demo_cmd.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level for demo execution (default: %(default)s)",
    )
    demo_cmd.set_defaults(func=_cmd_demo)

    proto_cmd = sub.add_parser("proto_sim", help="Launch the prototype simulation GUI")
    proto_cmd.add_argument("--config", required=True, help="Path to an environment configuration")
    proto_cmd.add_argument(
        "--backend",
        default="gpu",
        choices=[member.value for member in Backend],
        help="Genesis backend to use",
    )
    proto_cmd.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: %(default)s)",
    )
    proto_cmd.add_argument(
        "--debug",
        action="store_true",
        help="Enable extra GUI logging and diagnostics",
    )
    proto_cmd.set_defaults(func=_cmd_proto_sim)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _cmd_init(args) -> int:
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    presets_dir = Path(__file__).resolve().parent.parent / "environment" / "presets"
    copied = 0
    for preset in presets_dir.glob("*.yaml"):
        target = destination / preset.name
        if not target.exists():
            target.write_text(preset.read_text())
            copied += 1
    print(f"Copied {copied} preset(s) to {destination}")
    return 0


def _cmd_run(args) -> int:
    logger = setup_logging(args.log_level, debug=args.debug).getChild("cli")

    if not HAS_WX:
        logger.error("wxPython is required for the GUI. Install it with 'pip install wxpython'.")
        return 2

    try:
        logger.info(
            "Launching GUI backend=%s config=%s",
            args.backend,
            args.config,
        )
        run_gui(
            args.config,
            backend=Backend(args.backend),
            log_level=args.log_level,
            debug=args.debug,
            logger=logger.getChild("gui"),
        )
    except Exception as exc:
        logger.exception("Session terminated with error: %s", exc)
        return 1
    return 0


def _cmd_demo(args) -> int:
    demos = available_demos()
    if args.list:
        if demos:
            print("Available demos:")
            for name, desc in sorted(demos.items()):
                print(f"  {name:<12} {desc}")
        else:
            print("No demos are currently registered.")
        return 0

    if not args.name:
        print("Please provide a demo name. Use '--list' to see options.")
        return 2

    if args.name not in demos:
        print(
            f"Unknown demo '{args.name}'. Available demos: {', '.join(sorted(demos))}."
        )
        return 2

    logger = setup_logging(args.log_level).getChild("demo")
    try:
        logger.info("Starting demo '%s'", args.name)
        # run_demo handles its own asyncio loop
        run_demo(args.name)
        logger.info("Demo '%s' finished", args.name)
    except KeyboardInterrupt:  # pragma: no cover - user abort
        logger.info("Demo '%s' interrupted", args.name)
        return 130
    except Exception as exc:  # pragma: no cover - demo error reporting
        logger.exception("Demo '%s' failed: %s", args.name, exc)
        return 1
    return 0


def _cmd_proto_sim(args) -> int:
    logger = setup_logging(args.log_level, debug=args.debug).getChild("cli")

    if not HAS_WX:
        logger.error("wxPython is required for the GUI. Install it with 'pip install wxpython'.")
        return 2

    try:
        logger.info(
            "Launching proto_sim GUI backend=%s config=%s",
            args.backend,
            args.config,
        )
        run_proto_sim_gui(
            args.config,
            backend=Backend(args.backend),
            log_level=args.log_level,
            debug=args.debug,
            logger=logger.getChild("proto_gui"),
        )
    except Exception as exc:
        logger.exception("ProtoSim session terminated with error: %s", exc)
        return 1
    return 0


__all__ = ["build_parser", "main"]
