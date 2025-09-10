from __future__ import annotations

import argparse
import atexit
import shutil
import signal
import sys
from pathlib import Path

from .config import SimforgeConfig
from .gui import run_app
from .logging_utils import setup_logging
from .simulator import Simulator


# Global references for cleanup
_current_simulator = None
_shutdown_requested = False


def _safe_genesis_destroy():
    """Safe wrapper for Genesis destroy that handles CUDA context errors."""
    try:
        # Try to import genesis and call destroy
        import genesis as gs
        if hasattr(gs, 'destroy'):
            gs.destroy()
    except Exception as e:
        # Suppress CUDA context errors during shutdown
        if "CUDA_ERROR_INVALID_CONTEXT" in str(e) or "invalid device context" in str(e):
            print("Note: CUDA cleanup completed (some GPU resources may have been cleaned already)")
        else:
            print(f"Warning: Genesis cleanup error (non-critical): {e}")


def _patch_genesis_cleanup():
    """Patch Genesis cleanup to handle CUDA errors gracefully."""
    try:
        import genesis as gs
        import atexit

        # Use a more aggressive approach - replace the finalize function
        if hasattr(gs, 'logger') and hasattr(gs.logger, 'info'):
            original_info = gs.logger.info
            gs.logger.info = lambda *args: None  # Suppress logging during cleanup

        # Try to patch Taichi cleanup which often causes the CUDA issues
        try:
            import taichi as ti
            if hasattr(ti, 'reset'):
                original_ti_reset = ti.reset
                def safe_ti_reset():
                    try:
                        original_ti_reset()
                    except Exception:
                        pass  # Suppress all Taichi cleanup errors
                ti.reset = safe_ti_reset
        except ImportError:
            pass

        # Patch Genesis destroy function
        if hasattr(gs, 'destroy'):
            original_destroy = gs.destroy
            def wrapped_destroy():
                # Completely suppress stderr during cleanup to hide CUDA error messages and prevent core dumps
                import os
                try:
                    # Save original stderr and redirect to /dev/null to suppress ALL CUDA/graphics errors
                    stderr_fd = os.dup(2)  # Duplicate original stderr file descriptor
                    devnull_fd = os.open('/dev/null', os.O_WRONLY)
                    os.dup2(devnull_fd, 2)  # Make stderr (fd 2) point to /dev/null

                    try:
                        original_destroy()
                    except Exception:
                        # Suppress ALL cleanup errors during the cleanup phase
                        pass
                    finally:
                        # Restore stderr properly even if cleanup fails
                        try:
                            os.dup2(stderr_fd, 2)     # Restore original stderr
                            os.close(devnull_fd)      # Close our devnull file descriptor
                            os.close(stderr_fd)       # Close our saved stderr copy
                        except (OSError, AttributeError):
                            # If file descriptor operations fail, ignore and continue
                            pass
                except Exception:
                    pass  # Suppress ALL cleanup errors even at the redirection level

            gs.destroy = wrapped_destroy

    except Exception as e:
        print(f"Warning: Genesis cleanup patching failed: {e}")


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
        # Let main function handle exit to ensure proper cleanup order
        return


def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    # Ignore SIGPIPE (broken pipe) - common in headless mode
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)


@atexit.register
def cleanup():
    """Final cleanup function called when the process exits."""
    global _current_simulator, _shutdown_requested
    if _current_simulator and not _shutdown_requested:
        try:
            print("Performing final cleanup...")
            _current_simulator.stop()
        except Exception as e:
            # Always catch exceptions in atexit handlers
            print(f"Error during final cleanup: {e}")
            import traceback
            traceback.print_exc()


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
    global _current_simulator
    logger = setup_logging(debug)
    cfg = SimforgeConfig.from_yaml(config_path)
    if headless:
        sim = Simulator(cfg, debug=debug)
        _current_simulator = sim  # Set global reference for cleanup
        sim.start()  # <-- don't call build_scene() here
        logger.info("Running headless. Press Ctrl+C to stop.")
        try:
            while not _shutdown_requested:
                import time
                time.sleep(0.1)  # More efficient than busy loop
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            sim.stop()
        except Exception as e:
            logger.error(f"Unexpected error in headless mode: {e}")
            sim.stop()
        finally:
            _current_simulator = None
    else:
        run_app(cfg, debug=debug)


def main():
    # Set up signal handlers for graceful shutdown
    setup_signal_handlers()

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

