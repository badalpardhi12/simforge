"""
Simforge Client GUI Package

wxPython-based UI for protocol definition and robot control on macOS.
Communicates with the server via WebSocket for:
- Visualization (via Foxglove Studio)
- Robot commands (via Command Gateway)
- Safety monitoring (via Heartbeat)
"""

try:
    from simforge_client.gui.proto_sim_client import ProtoSimClientFrame, run_proto_sim_client
    __all__ = [
        "ProtoSimClientFrame",
        "run_proto_sim_client",
    ]
except ImportError:
    # Allow import to succeed even if wx is not installed
    __all__ = []
