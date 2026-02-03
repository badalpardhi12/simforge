#!/usr/bin/env python3
"""
Simforge Client Entry Point

Launch the Simforge client GUI on macOS for protocol simulation and robot control.

Usage:
    python -m simforge_client --server 192.168.1.12 --port 8766
    
Or run directly:
    python simforge_client/run_client.py --server 192.168.1.12
"""

import argparse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Simforge Client - Robot Control UI for macOS"
    )
    parser.add_argument(
        "--server",
        default="192.168.1.12",
        help="Server IP address (AI Workstation or Jetson Thor)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help="Command gateway WebSocket port",
    )
    parser.add_argument(
        "--foxglove-port",
        type=int,
        default=9090,
        help="Foxglove bridge WebSocket port",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting Simforge Client")
    logger.info(f"  Server: {args.server}")
    logger.info(f"  Command Port: {args.port}")
    logger.info(f"  Foxglove Port: {args.foxglove_port}")
    
    # Check for wxPython
    try:
        import wx
    except ImportError:
        logger.error("wxPython is required for the GUI.")
        logger.error("Install with: pip install wxPython")
        logger.error("")
        logger.error("On macOS, you may need:")
        logger.error("  brew install python-tk")
        logger.error("  pip install wxPython")
        sys.exit(1)
    
    # Import and run the GUI
    try:
        from simforge_client.gui.proto_sim_client import run_proto_sim_client
        
        logger.info("Launching Proto-Sim Client GUI...")
        logger.info("")
        logger.info("For visualization, open Foxglove Studio and connect to:")
        logger.info(f"  ws://{args.server}:{args.foxglove_port}")
        logger.info("")
        
        run_proto_sim_client(
            server_ip=args.server,
            command_port=args.port,
        )
        
    except Exception as e:
        logger.exception(f"Failed to start GUI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
