"""
Simforge Client - Main Entry Point

Run with: python -m simforge_client --server 192.168.1.12
"""
#!/usr/bin/env python3
"""
Entry point for running simforge_client as a module.

Usage:
    python -m simforge_client --server 192.168.1.12
"""
import sys
import os

# Ensure parent directory is in path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from simforge_client.run_client import main

if __name__ == "__main__":
    main()
else:
    # Also run main when imported as module (python -m simforge_client)
    main()
