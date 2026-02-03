"""
Proto-Sim Client UI

wxPython interface for protocol simulation that works with the distributed architecture.
Similar to simforge_new/interfaces/gui/proto_sim.py but communicates via WebSocket
instead of directly controlling Genesis.

Features:
- Protocol parameter input (pose sampling)
- Robot selection
- Object/reference frame selection
- Simulation vs Real Robot toggle
- Progress visualization
- Results logging
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass

try:
    import wx
    HAS_WX = True
except ImportError:
    HAS_WX = False
    wx = None

from simforge_client.command_client import SimforgeClient, MoveResult, MoveFeedback, Pose


LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


@dataclass
class ProtoSimParameters:
    """Parameters for protocol simulation pose generation."""
    # Distance from face (meters)
    distances: List[float]
    # Horizontal angles (degrees) - azimuth
    horizontal_angles: List[float]
    # Vertical angles (degrees) - elevation
    vertical_angles: List[float]
    # Idle time at each pose (seconds)
    idle_time: float = 2.0
    # Randomize order
    randomize: bool = False


@dataclass
class ProtoSimProgress:
    """Progress update during protocol simulation."""
    current_pose_index: int
    total_poses: int
    current_pose_name: str
    status: str
    progress_percent: float


class ParameterInputControl:
    """Helper for mode toggle + value entry widgets."""
    
    MODES = ("List", "Range")
    
    def __init__(
        self,
        parent: wx.Window,
        *,
        name: str,
        label: str,
        unit: str,
        default_mode: str,
        default_text: str,
    ) -> None:
        self.name = name
        self.label = label
        self.unit = unit
        self.mode_choice = wx.Choice(parent, choices=list(self.MODES))
        default_index = self.MODES.index(default_mode.title()) if default_mode.title() in self.MODES else 0
        self.mode_choice.SetSelection(default_index)
        self.text_ctrl = wx.TextCtrl(parent, value=default_text, size=(200, -1))
    
    def enable(self, enabled: bool) -> None:
        self.mode_choice.Enable(enabled)
        self.text_ctrl.Enable(enabled)
    
    def get_values(self) -> List[float]:
        mode = self.mode_choice.GetStringSelection()
        text = self.text_ctrl.GetValue().strip()
        if not text:
            raise ValueError(f"{self.label}: please provide a value")
        if mode == "List":
            return self._parse_list(text)
        if mode == "Range":
            return self._parse_range(text)
        raise ValueError(f"Unknown mode '{mode}' for {self.label}")
    
    @staticmethod
    def _parse_list(text: str) -> List[float]:
        values: List[float] = []
        for fragment in text.split(","):
            frag = fragment.strip()
            if not frag:
                continue
            try:
                values.append(float(frag))
            except ValueError as exc:
                raise ValueError(f"Invalid number '{frag}'") from exc
        if not values:
            raise ValueError("Please provide at least one numeric value")
        return values
    
    @staticmethod
    def _parse_range(text: str) -> List[float]:
        parts = [frag.strip() for frag in text.split(",") if frag.strip()]
        if len(parts) != 3:
            raise ValueError("Range format must be 'min,max,step'")
        try:
            min_val, max_val, step = (float(part) for part in parts)
        except ValueError as exc:
            raise ValueError("Range entries must be numeric") from exc
        if step == 0.0:
            raise ValueError("Range step must be non-zero")
        if max_val < min_val and step > 0.0:
            step = -abs(step)
        elif max_val > min_val and step < 0.0:
            step = abs(step)
        values: List[float] = []
        current = min_val
        limit = max_val + (1e-9 if step > 0 else -1e-9)
        safety = 0
        while (step > 0 and current <= limit) or (step < 0 and current >= limit):
            values.append(round(current, 6))
            current += step
            safety += 1
            if safety > 1000:
                raise ValueError("Range produces too many values (>1000)")
        return values


class ProtoSimClientFrame(wx.Frame):
    """
    Main wxPython frame for protocol simulation client.
    
    This runs on macOS and communicates with the server via WebSocket.
    """
    
    def __init__(
        self,
        parent: Optional[wx.Window],
        *,
        server_ip: str = "192.168.1.12",
        command_port: int = 8766,
        title: str = "Simforge Protocol Simulation",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not HAS_WX:
            raise ImportError("wxPython is required for the GUI. Install with: pip install wxPython")
        
        super().__init__(parent, title=title, size=(800, 700))
        
        self.logger = logger or logging.getLogger(__name__)
        self.server_ip = server_ip
        self.command_port = command_port
        
        # Async event loop in background thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        # Client instance
        self._client: Optional[SimforgeClient] = None
        self._connected = False
        self._running_simulation = False
        
        # Available robots and objects (fetched from server)
        self._robots: List[str] = []
        self._objects: List[str] = []
        
        self._build_ui()
        self._bind_events()
        
        # Auto-connect on startup
        wx.CallAfter(self._on_connect)
    
    def _run_loop(self) -> None:
        """Background thread running the asyncio event loop."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
    
    def _run_async(self, coro) -> asyncio.Future:
        """Schedule a coroutine on the background loop."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)
    
    def _build_ui(self) -> None:
        """Build the UI components."""
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # === Connection Section ===
        conn_box = wx.StaticBox(panel, label="Server Connection")
        conn_sizer = wx.StaticBoxSizer(conn_box, wx.HORIZONTAL)
        
        conn_sizer.Add(wx.StaticText(panel, label="Server IP:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.server_ip_ctrl = wx.TextCtrl(panel, value=self.server_ip, size=(150, -1))
        conn_sizer.Add(self.server_ip_ctrl, 0, wx.ALL, 5)
        
        conn_sizer.Add(wx.StaticText(panel, label="Port:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.port_ctrl = wx.TextCtrl(panel, value=str(self.command_port), size=(60, -1))
        conn_sizer.Add(self.port_ctrl, 0, wx.ALL, 5)
        
        self.connect_btn = wx.Button(panel, label="Connect")
        conn_sizer.Add(self.connect_btn, 0, wx.ALL, 5)
        
        self.status_label = wx.StaticText(panel, label="Disconnected")
        self.status_label.SetForegroundColour(wx.Colour(200, 0, 0))
        conn_sizer.Add(self.status_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        
        main_sizer.Add(conn_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # === Robot Selection ===
        robot_box = wx.StaticBox(panel, label="Robot Selection")
        robot_sizer = wx.StaticBoxSizer(robot_box, wx.HORIZONTAL)
        
        robot_sizer.Add(wx.StaticText(panel, label="Robot:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.robot_choice = wx.Choice(panel, choices=[])
        robot_sizer.Add(self.robot_choice, 1, wx.ALL, 5)
        
        # Execution mode
        self.sim_mode_radio = wx.RadioButton(panel, label="Simulation Only", style=wx.RB_GROUP)
        self.real_mode_radio = wx.RadioButton(panel, label="Real Robot")
        self.both_mode_radio = wx.RadioButton(panel, label="Both (Sim + Real)")
        
        robot_sizer.Add(self.sim_mode_radio, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        robot_sizer.Add(self.real_mode_radio, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        robot_sizer.Add(self.both_mode_radio, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        
        main_sizer.Add(robot_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # === Target Object Selection ===
        object_box = wx.StaticBox(panel, label="Target Object")
        object_sizer = wx.StaticBoxSizer(object_box, wx.HORIZONTAL)
        
        object_sizer.Add(wx.StaticText(panel, label="Object:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.object_choice = wx.Choice(panel, choices=[])
        object_sizer.Add(self.object_choice, 1, wx.ALL, 5)
        
        main_sizer.Add(object_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # === Protocol Parameters ===
        params_box = wx.StaticBox(panel, label="Protocol Parameters")
        params_sizer = wx.StaticBoxSizer(params_box, wx.VERTICAL)
        
        # Parameter grid
        param_grid = wx.FlexGridSizer(4, 4, 5, 10)
        param_grid.AddGrowableCol(2)
        
        # Headers
        param_grid.Add(wx.StaticText(panel, label="Parameter"), 0, wx.ALIGN_CENTER)
        param_grid.Add(wx.StaticText(panel, label="Mode"), 0, wx.ALIGN_CENTER)
        param_grid.Add(wx.StaticText(panel, label="Values"), 0, wx.ALIGN_CENTER)
        param_grid.Add(wx.StaticText(panel, label="Unit"), 0, wx.ALIGN_CENTER)
        
        # Distance
        self.distance_input = ParameterInputControl(
            panel,
            name="distance",
            label="Distance",
            unit="m",
            default_mode="List",
            default_text="0.3, 0.4, 0.5",
        )
        param_grid.Add(wx.StaticText(panel, label="Distance:"), 0, wx.ALIGN_CENTER_VERTICAL)
        param_grid.Add(self.distance_input.mode_choice, 0)
        param_grid.Add(self.distance_input.text_ctrl, 1, wx.EXPAND)
        param_grid.Add(wx.StaticText(panel, label="m"), 0, wx.ALIGN_CENTER_VERTICAL)
        
        # Horizontal angle
        self.horizontal_input = ParameterInputControl(
            panel,
            name="horizontal",
            label="Horizontal Angle",
            unit="°",
            default_mode="Range",
            default_text="-30, 30, 15",
        )
        param_grid.Add(wx.StaticText(panel, label="Horizontal:"), 0, wx.ALIGN_CENTER_VERTICAL)
        param_grid.Add(self.horizontal_input.mode_choice, 0)
        param_grid.Add(self.horizontal_input.text_ctrl, 1, wx.EXPAND)
        param_grid.Add(wx.StaticText(panel, label="°"), 0, wx.ALIGN_CENTER_VERTICAL)
        
        # Vertical angle
        self.vertical_input = ParameterInputControl(
            panel,
            name="vertical",
            label="Vertical Angle",
            unit="°",
            default_mode="Range",
            default_text="-15, 15, 15",
        )
        param_grid.Add(wx.StaticText(panel, label="Vertical:"), 0, wx.ALIGN_CENTER_VERTICAL)
        param_grid.Add(self.vertical_input.mode_choice, 0)
        param_grid.Add(self.vertical_input.text_ctrl, 1, wx.EXPAND)
        param_grid.Add(wx.StaticText(panel, label="°"), 0, wx.ALIGN_CENTER_VERTICAL)
        
        params_sizer.Add(param_grid, 0, wx.EXPAND | wx.ALL, 5)
        
        # Additional options
        options_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        options_sizer.Add(wx.StaticText(panel, label="Idle Time:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.idle_time_ctrl = wx.SpinCtrlDouble(panel, min=0.5, max=30.0, initial=2.0, inc=0.5)
        options_sizer.Add(self.idle_time_ctrl, 0, wx.ALL, 5)
        options_sizer.Add(wx.StaticText(panel, label="sec"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        
        options_sizer.AddStretchSpacer()
        
        self.randomize_check = wx.CheckBox(panel, label="Randomize Order")
        options_sizer.Add(self.randomize_check, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        
        params_sizer.Add(options_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        main_sizer.Add(params_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # === Control Buttons ===
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.start_btn = wx.Button(panel, label="Start Protocol")
        self.start_btn.SetBackgroundColour(wx.Colour(100, 200, 100))
        button_sizer.Add(self.start_btn, 0, wx.ALL, 5)
        
        self.stop_btn = wx.Button(panel, label="Stop")
        self.stop_btn.SetBackgroundColour(wx.Colour(200, 100, 100))
        self.stop_btn.Enable(False)
        button_sizer.Add(self.stop_btn, 0, wx.ALL, 5)
        
        button_sizer.AddStretchSpacer()
        
        self.estop_btn = wx.Button(panel, label="E-STOP", size=(100, 40))
        self.estop_btn.SetBackgroundColour(wx.Colour(255, 0, 0))
        self.estop_btn.SetForegroundColour(wx.Colour(255, 255, 255))
        button_sizer.Add(self.estop_btn, 0, wx.ALL, 5)
        
        main_sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # === Progress ===
        progress_box = wx.StaticBox(panel, label="Progress")
        progress_sizer = wx.StaticBoxSizer(progress_box, wx.VERTICAL)
        
        self.progress_bar = wx.Gauge(panel, range=100)
        progress_sizer.Add(self.progress_bar, 0, wx.EXPAND | wx.ALL, 5)
        
        self.progress_label = wx.StaticText(panel, label="Ready")
        progress_sizer.Add(self.progress_label, 0, wx.ALL, 5)
        
        main_sizer.Add(progress_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # === Log Output ===
        log_box = wx.StaticBox(panel, label="Log")
        log_sizer = wx.StaticBoxSizer(log_box, wx.VERTICAL)
        
        self.log_ctrl = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL,
            size=(-1, 150),
        )
        self.log_ctrl.SetFont(wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        log_sizer.Add(self.log_ctrl, 1, wx.EXPAND | wx.ALL, 5)
        
        main_sizer.Add(log_sizer, 1, wx.EXPAND | wx.ALL, 10)
        
        panel.SetSizer(main_sizer)
        
        # Menu bar
        self._create_menu_bar()
    
    def _create_menu_bar(self) -> None:
        """Create the menu bar."""
        menubar = wx.MenuBar()
        
        # File menu
        file_menu = wx.Menu()
        file_menu.Append(wx.ID_SAVE, "Save Log...\tCtrl+S")
        file_menu.AppendSeparator()
        file_menu.Append(wx.ID_EXIT, "Quit\tCtrl+Q")
        menubar.Append(file_menu, "File")
        
        # Help menu
        help_menu = wx.Menu()
        help_menu.Append(wx.ID_ABOUT, "About")
        menubar.Append(help_menu, "Help")
        
        self.SetMenuBar(menubar)
    
    def _bind_events(self) -> None:
        """Bind event handlers."""
        self.connect_btn.Bind(wx.EVT_BUTTON, lambda e: self._on_connect())
        self.start_btn.Bind(wx.EVT_BUTTON, lambda e: self._on_start())
        self.stop_btn.Bind(wx.EVT_BUTTON, lambda e: self._on_stop())
        self.estop_btn.Bind(wx.EVT_BUTTON, lambda e: self._on_estop())
        
        self.Bind(wx.EVT_MENU, lambda e: self._on_save_log(), id=wx.ID_SAVE)
        self.Bind(wx.EVT_MENU, lambda e: self.Close(), id=wx.ID_EXIT)
        self.Bind(wx.EVT_MENU, lambda e: self._on_about(), id=wx.ID_ABOUT)
        self.Bind(wx.EVT_CLOSE, self._on_close)
    
    def _log(self, message: str) -> None:
        """Add message to log control."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        wx.CallAfter(self.log_ctrl.AppendText, f"[{timestamp}] {message}\n")
        self.logger.info(message)
    
    def _update_connection_status(self, connected: bool) -> None:
        """Update UI based on connection status."""
        self._connected = connected
        if connected:
            wx.CallAfter(self.status_label.SetLabel, "Connected")
            wx.CallAfter(self.status_label.SetForegroundColour, wx.Colour(0, 150, 0))
            wx.CallAfter(self.connect_btn.SetLabel, "Disconnect")
        else:
            wx.CallAfter(self.status_label.SetLabel, "Disconnected")
            wx.CallAfter(self.status_label.SetForegroundColour, wx.Colour(200, 0, 0))
            wx.CallAfter(self.connect_btn.SetLabel, "Connect")
    
    def _on_connect(self) -> None:
        """Handle connect/disconnect button."""
        if self._connected:
            self._run_async(self._disconnect())
        else:
            server_ip = self.server_ip_ctrl.GetValue().strip()
            try:
                port = int(self.port_ctrl.GetValue().strip())
            except ValueError:
                self._log("Invalid port number")
                return
            self._run_async(self._connect(server_ip, port))
    
    async def _connect(self, server_ip: str, port: int) -> None:
        """Connect to the server."""
        self._log(f"Connecting to {server_ip}:{port}...")
        try:
            self._client = SimforgeClient(
                server_ip=server_ip,
                command_port=port,
                client_id="proto_sim_gui",
            )
            await self._client.connect()
            self._update_connection_status(True)
            self._log("Connected to server")
            
            # Fetch available robots and objects
            await self._fetch_environment_info()
            
        except Exception as e:
            self._log(f"Connection failed: {e}")
            self._update_connection_status(False)
    
    async def _disconnect(self) -> None:
        """Disconnect from the server."""
        if self._client:
            await self._client.disconnect()
            self._client = None
        self._update_connection_status(False)
        self._log("Disconnected")
    
    async def _fetch_environment_info(self) -> None:
        """Fetch available robots and objects from server."""
        if not self._client:
            return
        
        try:
            # Get environment info via custom RPC call
            response = await self._client.call_rpc("get_environment_info", {})
            
            if response.get("success"):
                self._robots = response.get("robots", [])
                self._objects = response.get("objects", [])
                
                wx.CallAfter(self._update_robot_choices)
                wx.CallAfter(self._update_object_choices)
                
                self._log(f"Found {len(self._robots)} robots, {len(self._objects)} objects")
            else:
                self._log(f"Failed to get environment info: {response.get('error', 'Unknown error')}")
                
        except Exception as e:
            self._log(f"Error fetching environment info: {e}")
    
    def _update_robot_choices(self) -> None:
        """Update robot dropdown."""
        self.robot_choice.Clear()
        for robot in self._robots:
            self.robot_choice.Append(robot)
        if self._robots:
            self.robot_choice.SetSelection(0)
    
    def _update_object_choices(self) -> None:
        """Update object dropdown."""
        self.object_choice.Clear()
        for obj in self._objects:
            self.object_choice.Append(obj)
        if self._objects:
            self.object_choice.SetSelection(0)
    
    def _on_start(self) -> None:
        """Start protocol simulation."""
        if not self._connected:
            self._log("Not connected to server")
            return
        
        # Gather parameters
        try:
            params = ProtoSimParameters(
                distances=self.distance_input.get_values(),
                horizontal_angles=self.horizontal_input.get_values(),
                vertical_angles=self.vertical_input.get_values(),
                idle_time=self.idle_time_ctrl.GetValue(),
                randomize=self.randomize_check.GetValue(),
            )
        except ValueError as e:
            self._log(f"Parameter error: {e}")
            return
        
        robot = self.robot_choice.GetStringSelection()
        target_object = self.object_choice.GetStringSelection()
        
        if not robot:
            self._log("Please select a robot")
            return
        if not target_object:
            self._log("Please select a target object")
            return
        
        # Determine execution mode
        if self.sim_mode_radio.GetValue():
            mode = "simulation"
        elif self.real_mode_radio.GetValue():
            mode = "real"
        else:
            mode = "both"
        
        total_poses = len(params.distances) * len(params.horizontal_angles) * len(params.vertical_angles)
        self._log(f"Starting protocol: {total_poses} poses, mode={mode}")
        
        # Update UI
        self.start_btn.Enable(False)
        self.stop_btn.Enable(True)
        self._running_simulation = True
        
        # Run the protocol
        self._run_async(self._run_protocol(robot, target_object, params, mode))
    
    async def _run_protocol(
        self,
        robot: str,
        target_object: str,
        params: ProtoSimParameters,
        mode: str,
    ) -> None:
        """Execute the protocol simulation."""
        try:
            response = await self._client.call_rpc("run_proto_sim", {
                "robot_name": robot,
                "target_object": target_object,
                "distances": params.distances,
                "horizontal_angles": params.horizontal_angles,
                "vertical_angles": params.vertical_angles,
                "idle_time": params.idle_time,
                "randomize": params.randomize,
                "mode": mode,
            })
            
            if response.get("success"):
                self._log(f"Protocol completed: {response.get('message', 'OK')}")
            else:
                self._log(f"Protocol failed: {response.get('error', 'Unknown error')}")
                
        except Exception as e:
            self._log(f"Protocol error: {e}")
        finally:
            self._running_simulation = False
            wx.CallAfter(self.start_btn.Enable, True)
            wx.CallAfter(self.stop_btn.Enable, False)
            wx.CallAfter(self.progress_bar.SetValue, 0)
    
    def _on_stop(self) -> None:
        """Stop the current simulation."""
        self._log("Stopping...")
        if self._client:
            self._run_async(self._client.call_rpc("stop_proto_sim", {}))
    
    def _on_estop(self) -> None:
        """Trigger emergency stop."""
        self._log("E-STOP TRIGGERED!")
        if self._client:
            self._run_async(self._client.emergency_stop())
    
    def _on_save_log(self) -> None:
        """Save log to file."""
        with wx.FileDialog(
            self,
            "Save Log",
            defaultDir=str(LOG_DIR),
            defaultFile=f"proto_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            wildcard="Log files (*.log)|*.log",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                with open(path, "w") as f:
                    f.write(self.log_ctrl.GetValue())
                self._log(f"Log saved to {path}")
    
    def _on_about(self) -> None:
        """Show about dialog."""
        wx.MessageBox(
            "Simforge Protocol Simulation Client\n\n"
            "A distributed robot control interface for\n"
            "protocol definition and execution.\n\n"
            "Part of the Simforge distributed architecture.",
            "About",
            wx.OK | wx.ICON_INFORMATION,
        )
    
    def _on_close(self, event: wx.CloseEvent) -> None:
        """Handle window close."""
        if self._running_simulation:
            if wx.MessageBox(
                "A simulation is running. Stop and exit?",
                "Confirm Exit",
                wx.YES_NO | wx.ICON_WARNING,
            ) != wx.YES:
                event.Veto()
                return
        
        # Cleanup
        if self._client:
            self._run_async(self._disconnect())
        
        self._loop.call_soon_threadsafe(self._loop.stop)
        event.Skip()


def run_proto_sim_client(
    server_ip: str = "192.168.1.12",
    command_port: int = 8766,
) -> None:
    """Launch the Proto-Sim client GUI."""
    if not HAS_WX:
        raise ImportError("wxPython is required. Install with: pip install wxPython")
    
    app = wx.App()
    frame = ProtoSimClientFrame(
        None,
        server_ip=server_ip,
        command_port=command_port,
    )
    frame.Show()
    app.MainLoop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simforge Proto-Sim Client")
    parser.add_argument("--server", default="192.168.1.12", help="Server IP address")
    parser.add_argument("--port", type=int, default=8766, help="Command port")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    run_proto_sim_client(server_ip=args.server, command_port=args.port)
