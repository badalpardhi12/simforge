from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .config import SimforgeConfig
from .gui import run_app
from .logging_utils import setup_logging
from .simulator import Simulator


def cmd_init():
    here = Path(__file__).resolve().parent.parent
    src = here / "env_configs"
    dst = Path.cwd() / "env_configs"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.yaml"):
        out = dst / f.name
        if not out.exists():
            shutil.copy(f, out)
    print(f"Templates copied to {dst}")


def cmd_run(config_path: str, headless: bool, debug: bool):
    logger = setup_logging(debug)
    cfg = SimforgeConfig.from_yaml(config_path)
    if headless:
        sim = Simulator(cfg, debug=debug)
        sim.start()  # <-- don't call build_scene() here
        logger.info("Running headless. Press Ctrl+C to stop.")
        try:
            while True:
                pass
        except KeyboardInterrupt:
            sim.stop()
    else:
        run_app(cfg, debug=debug)


def main():
    parser = argparse.ArgumentParser(prog="simforge", description="Genesis-powered robot simulator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Copy template env config files")
    p_init.set_defaults(func=lambda args: cmd_init())

    p_run = sub.add_parser("run", help="Run simulator (GUI by default)")
    p_run.add_argument("--config", required=True, help="Path to YAML env config")
    p_run.add_argument("--headless", action="store_true", help="Run without GUI viewer")
    p_run.add_argument("--debug", action="store_true", help="Enable debug logging")
    p_run.set_defaults(func=lambda args: cmd_run(args.config, args.headless, args.debug))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

