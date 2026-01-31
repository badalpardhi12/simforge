"""
Simforge Client Package

Cross-platform client for controlling robots via WebSocket connection to the 
Simforge Server running on AI Workstation or Jetson Thor.

This client does NOT require ROS 2 - it uses pure WebSocket communication.

Designed for macOS (Apple Silicon) but works on any platform with Python 3.10+.
"""

__version__ = "1.0.0"
__author__ = "Badal"

from simforge_client.command_client import SimforgeClient
from simforge_client.safety_monitor import SafetyMonitor

__all__ = [
    "SimforgeClient",
    "SafetyMonitor",
]
