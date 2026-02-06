"""
Proto-Sim Client UI

wxPython interface for protocol simulation that works with the distributed architecture.
Generates poses locally on macOS and sends them to the server in base_link frame.

Architecture:
- Client fetches environment geometry (TF data) from server
- Client generates poses locally using spherical coordinate sampling
- Client transforms poses to robot's base_link frame
- Client sends pre-computed poses to server
- Server executes poses with collision checking (rejects unsafe poses)

Features:
- Protocol parameter input (pose sampling) matching simforge_new exactly
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
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass

try:
    import wx
    HAS_WX = True
except ImportError:
    HAS_WX = False
    wx = None

from simforge_client.command_client import SimforgeClient, MoveResult, MoveFeedback, Pose
from simforge_client.utils.pose_generation import (
    ProtoSimParameters as PoseGenParams,
    generate_world_poses,
    count_poses,
)


LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


@dataclass
class ProtoSimParameters:
    """
    Parameters for protocol simulation pose generation.
    
    Matches simforge_new/control/proto_simulation.py ProtoSimParameters exactly.
    """
    # Horizontal shift from object center (mm)
    horiz: List[float]
    # Vertical shift from object center (mm)
    vert: List[float]
    # Distance from object (mm)
    distance: List[float]
    # Roll angle (degrees)
    roll: List[float]
    # Pitch angle (degrees) - vertical angle to look up/down
    pitch: List[float]
    # Yaw angle (degrees) - horizontal angle to look left/right
    yaw: List[float]
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
    Parameters match simforge_new/interfaces/gui/proto_sim.py exactly.
    """
    
    # Parameter definitions matching simforge_new
    PARAMETERS = (
        {"name": "horiz", "label": "Horiz Shift", "unit": "mm", "mode": "List", "default": "0"},
        {"name": "vert", "label": "Vert Shift", "unit": "mm", "mode": "List", "default": "0"},
        {"name": "distance", "label": "Distance", "unit": "mm", "mode": "List", "default": "250,350,450,550"},
        {"name": "roll", "label": "Roll", "unit": "°", "mode": "List", "default": "-90"},
        {"name": "pitch", "label": "Pitch", "unit": "°", "mode": "List", "default": "-45,-30,0,15"},
        {"name": "yaw", "label": "Yaw", "unit": "°", "mode": "Range", "default": "-30,30,30"},
    )
    
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
        
        super().__init__(parent, title=title, size=(850, 850))
        
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
        # Object transforms in base_link frame (for local pose generation)
        self._object_transforms: Dict[str, Optional[Dict]] = {}
        
        # Parameter controls
        self.parameter_controls: Dict[str, ParameterInputControl] = {}
        
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
        
        # Execution mode — always enabled, server validates at execution time
        self.sim_mode_radio = wx.RadioButton(panel, label="Simulation Only", style=wx.RB_GROUP)
        self.real_mode_radio = wx.RadioButton(panel, label="Real Robot")
        self.both_mode_radio = wx.RadioButton(panel, label="Both (Sim + Real)")
        
        robot_sizer.Add(self.sim_mode_radio, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        robot_sizer.Add(self.real_mode_radio, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        robot_sizer.Add(self.both_mode_radio, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        
        # Robot connection status indicator (informational, does NOT gate mode selection)
        self.robot_status_indicator = wx.StaticText(panel, label="\u25CF")
        self.robot_status_indicator.SetForegroundColour(wx.Colour(150, 150, 150))  # grey = unknown
        self.robot_status_indicator.SetToolTip("Robot connection status (click Refresh)")
        robot_sizer.Add(self.robot_status_indicator, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 2)
        
        self.robot_status_label = wx.StaticText(panel, label="")
        self.robot_status_label.SetFont(wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_ITALIC, wx.FONTWEIGHT_NORMAL))
        robot_sizer.Add(self.robot_status_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        
        # Refresh button to check robot status
        self.refresh_status_btn = wx.Button(panel, label="↻ Refresh")
        self.refresh_status_btn.SetToolTip("Check real robot connection status")
        robot_sizer.Add(self.refresh_status_btn, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        
        main_sizer.Add(robot_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # === Target Object Selection ===
        object_box = wx.StaticBox(panel, label="Target Object")
        object_sizer = wx.StaticBoxSizer(object_box, wx.HORIZONTAL)
        
        object_sizer.Add(wx.StaticText(panel, label="Object:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.object_choice = wx.Choice(panel, choices=[])
        object_sizer.Add(self.object_choice, 1, wx.ALL, 5)
        
        main_sizer.Add(object_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # === Protocol Parameters (matching simforge_new) ===
        params_box = wx.StaticBox(panel, label="Movement Parameters")
        params_sizer = wx.StaticBoxSizer(params_box, wx.VERTICAL)
        
        # Parameter grid
        param_grid = wx.FlexGridSizer(rows=len(self.PARAMETERS), cols=3, hgap=10, vgap=8)
        param_grid.AddGrowableCol(2, 1)
        
        for item in self.PARAMETERS:
            label_text = f"{item['label']} ({item['unit']})"
            label = wx.StaticText(panel, label=label_text)
            control = ParameterInputControl(
                panel,
                name=item["name"],
                label=item["label"],
                unit=item["unit"],
                default_mode=item["mode"],
                default_text=item["default"],
            )
            self.parameter_controls[item["name"]] = control
            param_grid.Add(label, 0, wx.ALIGN_CENTER_VERTICAL)
            param_grid.Add(control.mode_choice, 0, wx.EXPAND)
            param_grid.Add(control.text_ctrl, 1, wx.EXPAND)
        
        params_sizer.Add(param_grid, 0, wx.EXPAND | wx.ALL, 8)
        
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
        
        # Pose count preview
        self.pose_count_label = wx.StaticText(panel, label="Total poses: --")
        self.pose_count_label.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_ITALIC, wx.FONTWEIGHT_NORMAL))
        params_sizer.Add(self.pose_count_label, 0, wx.ALL, 8)
        
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
        self.refresh_status_btn.Bind(wx.EVT_BUTTON, lambda e: self._on_refresh_status())
        
        # Bind parameter changes to update pose count
        for control in self.parameter_controls.values():
            control.text_ctrl.Bind(wx.EVT_TEXT, lambda e: self._update_pose_count())
            control.mode_choice.Bind(wx.EVT_CHOICE, lambda e: self._update_pose_count())
        
        self.Bind(wx.EVT_MENU, lambda e: self._on_save_log(), id=wx.ID_SAVE)
        self.Bind(wx.EVT_MENU, lambda e: self.Close(), id=wx.ID_EXIT)
        self.Bind(wx.EVT_MENU, lambda e: self._on_about(), id=wx.ID_ABOUT)
        self.Bind(wx.EVT_CLOSE, self._on_close)
    
    def _update_pose_count(self) -> None:
        """Update the pose count label based on current parameters."""
        try:
            params = self._collect_parameters()
            count = (
                len(params.horiz) * 
                len(params.vert) * 
                len(params.distance) * 
                len(params.roll) * 
                len(params.pitch) * 
                len(params.yaw)
            )
            self.pose_count_label.SetLabel(f"Total poses: {count}")
        except ValueError:
            self.pose_count_label.SetLabel("Total poses: (invalid parameters)")
    
    def _collect_parameters(self) -> ProtoSimParameters:
        """Collect parameters from UI controls."""
        return ProtoSimParameters(
            horiz=self.parameter_controls["horiz"].get_values(),
            vert=self.parameter_controls["vert"].get_values(),
            distance=self.parameter_controls["distance"].get_values(),
            roll=self.parameter_controls["roll"].get_values(),
            pitch=self.parameter_controls["pitch"].get_values(),
            yaw=self.parameter_controls["yaw"].get_values(),
            idle_time=self.idle_time_ctrl.GetValue(),
            randomize=self.randomize_check.GetValue(),
        )
    
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
    
    def _on_refresh_status(self) -> None:
        """Handle refresh status button click."""
        if not self._connected:
            self._log("Not connected to server")
            return
        self._log("Refreshing robot status...")
        self._run_async(self._fetch_robot_status())
    
    async def _fetch_environment_info(self) -> None:
        """Fetch available robots, objects, and their transforms from server."""
        if not self._client:
            return
        
        try:
            # Get environment info via custom RPC call
            response = await self._client.call_rpc("get_environment_info", {})
            
            if response.get("success"):
                self._robots = response.get("robots", [])
                self._objects = response.get("objects", [])
                # Store object transforms for local pose generation
                self._object_transforms = response.get("object_transforms", {})
                
                wx.CallAfter(self._update_robot_choices)
                wx.CallAfter(self._update_object_choices)
                
                self._log(f"Found {len(self._robots)} robots, {len(self._objects)} objects")
                
                # Log available transforms
                for obj, tf in self._object_transforms.items():
                    if tf:
                        pos = tf.get('position', [0, 0, 0])
                        self._log(f"  {obj}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            else:
                self._log(f"Failed to get environment info: {response.get('error', 'Unknown error')}")
            
            # Also fetch robot connection status to enable/disable mode buttons
            await self._fetch_robot_status()
                
        except Exception as e:
            self._log(f"Error fetching environment info: {e}")
    
    async def _fetch_robot_status(self) -> None:
        """Fetch real robot connection status and update mode button availability."""
        if not self._client:
            return
        
        try:
            response = await self._client.call_rpc("get_robot_status", {})
            
            if response.get("success"):
                real_robot_available = response.get("real_robot_available", False)
                available_modes = response.get("available_modes", ["simulation"])
                connection_details = response.get("connection_details", {})
                
                # Log connection status
                if real_robot_available:
                    self._log("✓ Real robot available - all modes enabled")
                else:
                    self._log("⚠ Real robot not connected - simulation only mode")
                    for key, value in connection_details.items():
                        self._log(f"  {key}: {value}")
                
                # Update mode button availability
                wx.CallAfter(self._update_mode_availability, available_modes)
            else:
                self._log(f"Could not get robot status: {response.get('error', 'Unknown')}")
                wx.CallAfter(self._update_mode_availability, ["simulation"])
                
        except Exception as e:
            self._log(f"Error fetching robot status: {e}")
            wx.CallAfter(self._update_mode_availability, ["simulation"])
    
    def _update_robot_status_indicator(
        self,
        robot_available: bool,
        program_running: bool,
        connection_details: dict,
    ) -> None:
        """Update the robot connection status indicator (dot + label).
        
        This is purely informational — mode radio buttons stay enabled.
        """
        if robot_available and program_running:
            self.robot_status_indicator.SetForegroundColour(wx.Colour(0, 180, 0))   # green
            self.robot_status_indicator.SetLabel("\u25CF")
            self.robot_status_label.SetLabel("Robot ready")
            self.robot_status_label.SetForegroundColour(wx.Colour(0, 140, 0))
        elif robot_available:
            self.robot_status_indicator.SetForegroundColour(wx.Colour(220, 180, 0)) # amber
            self.robot_status_indicator.SetLabel("\u25CF")
            self.robot_status_label.SetLabel("Robot reachable")
            self.robot_status_label.SetForegroundColour(wx.Colour(180, 140, 0))
        else:
            self.robot_status_indicator.SetForegroundColour(wx.Colour(180, 0, 0))   # red
            self.robot_status_indicator.SetLabel("\u25CF")
            self.robot_status_label.SetLabel("Robot offline")
            self.robot_status_label.SetForegroundColour(wx.Colour(150, 0, 0))
        
        tip_lines = []
        for k, v in connection_details.items():
            tip_lines.append(f"{k}: {v}")
        self.robot_status_indicator.SetToolTip("\n".join(tip_lines) if tip_lines else "No details")
        
        self.robot_status_indicator.Refresh()
        self.robot_status_label.Refresh()
    
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
            params = self._collect_parameters()
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
        
        total_poses = (
            len(params.horiz) * 
            len(params.vert) * 
            len(params.distance) * 
            len(params.roll) * 
            len(params.pitch) * 
            len(params.yaw)
        )
        self._log(f"Starting protocol: {total_poses} poses, mode={mode}")
        
        # For real / both modes, run a pre-flight check with a progress dialog
        if mode in ("real", "both"):
            self.start_btn.Enable(False)
            self._log("Checking robot readiness...")
            self._run_async(self._prepare_and_run(robot, target_object, params, mode, total_poses))
        else:
            # Simulation — run immediately
            self.start_btn.Enable(False)
            self.stop_btn.Enable(True)
            self._running_simulation = True
            self._run_async(self._run_protocol(robot, target_object, params, mode))
    
    async def _prepare_and_run(
        self,
        robot: str,
        target_object: str,
        params: ProtoSimParameters,
        mode: str,
        total_poses: int,
    ) -> None:
        """Pre-flight check for real/both modes, then run the protocol.
        
        Calls the server's prepare_mode RPC to validate that the real robot
        is reachable and the program is running. Shows a brief freeze/wait
        while the server checks. If the check fails the user gets a dialog
        with the option to retry, switch to simulation, or cancel.
        """
        try:
            wx.CallAfter(self.progress_label.SetLabel, "Checking robot readiness…")
            
            response = await self._client.call_rpc("prepare_mode", {"mode": mode}, timeout=30.0)
            
            if response.get("ready"):
                self._log(f"✓ Robot ready for {mode} mode")
                self._log(response.get("message", ""))
                # Proceed with execution
                self._running_simulation = True
                wx.CallAfter(self.stop_btn.Enable, True)
                await self._run_protocol(robot, target_object, params, mode)
            else:
                msg = response.get("message", "Robot not ready")
                self._log(f"✗ {msg}")
                
                can_retry = response.get("can_retry", False)
                
                # Show dialog on the GUI thread and wait for user decision
                user_choice = await self._show_mode_check_dialog(mode, msg, can_retry)
                
                if user_choice == "retry":
                    self._log("Retrying robot readiness check...")
                    await self._prepare_and_run(robot, target_object, params, mode, total_poses)
                    return
                elif user_choice == "simulation":
                    self._log("Switching to Simulation Only mode")
                    wx.CallAfter(self.sim_mode_radio.SetValue, True)
                    self._running_simulation = True
                    wx.CallAfter(self.stop_btn.Enable, True)
                    await self._run_protocol(robot, target_object, params, "simulation")
                else:
                    self._log("Cancelled by user")
                    wx.CallAfter(self.start_btn.Enable, True)
                    wx.CallAfter(self.progress_label.SetLabel, "Ready")
                    
        except Exception as e:
            self._log(f"Pre-flight check error: {e}")
            # Offer to fall back to simulation
            user_choice = await self._show_mode_check_dialog(
                mode,
                f"Could not reach server for pre-flight check:\n{e}",
                can_retry=True,
            )
            if user_choice == "simulation":
                self._log("Falling back to Simulation Only mode")
                wx.CallAfter(self.sim_mode_radio.SetValue, True)
                self._running_simulation = True
                wx.CallAfter(self.stop_btn.Enable, True)
                await self._run_protocol(robot, target_object, params, "simulation")
            elif user_choice == "retry":
                await self._prepare_and_run(robot, target_object, params, mode, total_poses)
            else:
                wx.CallAfter(self.start_btn.Enable, True)
                wx.CallAfter(self.progress_label.SetLabel, "Ready")
    
    async def _show_mode_check_dialog(
        self, mode: str, message: str, can_retry: bool
    ) -> str:
        """Show a dialog on the GUI thread and return the user's choice.
        
        Returns one of: 'retry', 'simulation', 'cancel'
        """
        future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        
        def _show():
            dlg = wx.Dialog(self, title=f"{mode.title()} Mode — Pre-flight Check", size=(480, 280))
            sizer = wx.BoxSizer(wx.VERTICAL)
            
            msg_ctrl = wx.TextCtrl(
                dlg, value=message,
                style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_NO_VSCROLL,
                size=(-1, 120),
            )
            msg_ctrl.SetFont(wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            sizer.Add(msg_ctrl, 1, wx.EXPAND | wx.ALL, 10)
            
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
            if can_retry:
                retry_btn = wx.Button(dlg, label="Retry")
                retry_btn.Bind(wx.EVT_BUTTON, lambda e: (dlg.EndModal(1)))
                btn_sizer.Add(retry_btn, 0, wx.ALL, 5)
            
            sim_btn = wx.Button(dlg, label="Use Simulation Only")
            sim_btn.Bind(wx.EVT_BUTTON, lambda e: (dlg.EndModal(2)))
            btn_sizer.Add(sim_btn, 0, wx.ALL, 5)
            
            cancel_btn = wx.Button(dlg, label="Cancel")
            cancel_btn.Bind(wx.EVT_BUTTON, lambda e: (dlg.EndModal(0)))
            btn_sizer.Add(cancel_btn, 0, wx.ALL, 5)
            
            sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)
            dlg.SetSizer(sizer)
            
            result = dlg.ShowModal()
            dlg.Destroy()
            
            choice = {0: "cancel", 1: "retry", 2: "simulation"}.get(result, "cancel")
            # Resolve the future from the async loop thread
            self._loop.call_soon_threadsafe(future.set_result, choice)
        
        wx.CallAfter(_show)
        return await future

    async def _run_protocol(
        self,
        robot: str,
        target_object: str,
        params: ProtoSimParameters,
        mode: str,
    ) -> None:
        """Execute the protocol simulation with client-side pose generation."""
        try:
            # Get target object transform for local pose generation
            target_tf = self._object_transforms.get(target_object)
            if not target_tf:
                self._log(f"Error: No transform available for {target_object}")
                self._log("Refreshing environment info...")
                await self._fetch_environment_info()
                target_tf = self._object_transforms.get(target_object)
                if not target_tf:
                    self._log(f"Error: Still no transform for {target_object}. Aborting.")
                    return
            
            target_position = tuple(target_tf['position'])
            target_orientation = tuple(target_tf['orientation'])
            
            self._log(f"Target {target_object} at: ({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})")
            
            # Generate poses locally using pose_generation module
            pose_params = PoseGenParams(
                horiz=params.horiz,
                vert=params.vert,
                distance=params.distance,
                roll=params.roll,
                pitch=params.pitch,
                yaw=params.yaw,
            )
            
            world_poses = generate_world_poses(
                pose_params,
                target_position,
                target_orientation,
                randomize=params.randomize,
            )
            
            total_poses = len(world_poses)
            self._log(f"Generated {total_poses} poses locally in base_link frame")
            
            # Convert WorldPose objects to dicts for JSON serialization
            poses_data = [pose.to_dict() for pose in world_poses]
            
            # Calculate timeout
            timeout_per_pose = params.idle_time + 3.0
            timeout = max(120.0, total_poses * timeout_per_pose + 60.0)
            
            self._log(f"Sending poses to server (timeout: {timeout:.0f}s)")
            
            # Send pre-computed poses to server
            response = await self._client.call_rpc("run_proto_sim", {
                "robot_name": robot,
                "poses": poses_data,  # NEW: Pre-computed poses in base_link frame
                "idle_time": params.idle_time,
                "mode": mode,
            }, timeout=timeout)
            
            if response.get("success"):
                completed = response.get('completed', 0)
                collision_rejected = response.get('collision_rejected', 0)
                ik_failed = response.get('ik_failed', 0)
                self._log(f"Protocol completed: {completed}/{total_poses} poses")
                if collision_rejected > 0:
                    self._log(f"  Collision rejected: {collision_rejected}")
                if ik_failed > 0:
                    self._log(f"  IK failed: {ik_failed}")
            else:
                self._log(f"Protocol failed: {response.get('error', 'Unknown error')}")
                
        except Exception as e:
            self._log(f"Protocol error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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
            "Part of the Simforge distributed architecture.\n\n"
            "Parameters match simforge_new/interfaces/gui/proto_sim.py",
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
