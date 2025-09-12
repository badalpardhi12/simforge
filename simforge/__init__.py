"""Simforge: Genesis-powered robot simulator with joint and Cartesian control."""

from __future__ import annotations

import argparse
import signal
import atexit
import shutil
from pathlib import Path
import logging

from .config_reader import SimforgeConfig
from .control_gui import run_gui
from .logging_utils import setup_logging


__all__ = [
    "__version__",
    "main",
    "SimforgeConfig",
    "run_gui",
]

__version__ = "0.1.0"


# Global references for cleanup
_current_simulator = None
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) and SIGTERM gracefully."""
    global _shutdown_requested, _current_simulator
    if not _shutdown_requested:
        _shutdown_requested = True
        print("\nShutting down gracefully... (Ctrl+C again to force quit)")
        if _current_simulator:
            try:
                _current_simulator.stop()
            except Exception as e:
                print(f"Error during shutdown: {e}")
        return


def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)


def cleanup():
    """Cleanup function registered with atexit."""
    global _current_simulator
    if _current_simulator:
        try:
            _current_simulator.stop()
        except Exception:
            pass


atexit.register(cleanup)


def cmd_init():
    """Copy template environment config files to current directory."""
    here = Path(__file__).resolve().parent.parent
    src = here / "env_configs"
    dst = Path.cwd() / "env_configs"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.yaml"):
        out = dst / f.name
        if not out.exists():
            shutil.copy(f, out)
    print(f"Templates copied to {dst}")


def cmd_run(config_path: str, debug: bool) -> None:
    """Run simulator with Genesis viewer and control GUI."""
    global _current_simulator
    logger = setup_logging(debug)
    cfg = SimforgeConfig.from_yaml(config_path)
    
    try:
        run_gui(cfg, debug=debug)
    except KeyboardInterrupt:
        logger.info("Interrupt received, stopping...")
    except Exception as e:
        logger.error(f"Error running simulation: {e}")
        raise


def main() -> None:
    """Main entry point for the simforge CLI."""
    setup_signal_handlers()
    parser = argparse.ArgumentParser(
        prog="simforge", 
        description="Genesis-powered robot simulator with GUI"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Copy template env config files")
    p_init.set_defaults(func=lambda args: cmd_init())

    p_run = sub.add_parser(
        "run", 
        help="Run simulator with Genesis viewer and control GUI"
    )
    p_run.add_argument(
        "--config", 
        required=True, 
        help="Path to YAML env config"
    )
    p_run.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    p_run.set_defaults(func=lambda args: cmd_run(args.config, args.debug))

    args = parser.parse_args()
    args.func(args)

    # Ensure Genesis logger stays quiet
    g = logging.getLogger("genesis")
    g.setLevel(logging.WARNING)
    g.propagate = False

