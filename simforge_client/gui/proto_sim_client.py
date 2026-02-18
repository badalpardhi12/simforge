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
- Protocol parameter input (pose sampling) matching simforge_genesis exactly
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
import time
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
    
    Matches simforge_genesis/control/proto_simulation.py ProtoSimParameters exactly.
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
    # Velocity scaling for real robot (0.01–1.0)
    move_speed: float = 0.25


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
    Parameters match simforge_genesis/interfaces/gui/proto_sim.py exactly.
    """
    
    # Parameter definitions matching simforge_genesis
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
        
        # Mode-switch tracking
        self._server_mode: Optional[str] = None   # last confirmed server mode
        self._mode_ready: bool = False             # True once prepare_mode succeeded
        self._mode_switching: bool = False          # True while RPC is in-flight
        
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
        
        # Execution mode — two options:
        #   "Simulation Only" → mode="simulation"
        #   "Real Robot"      → mode="both" (sim + real combined)
        self.sim_mode_radio = wx.RadioButton(panel, label="Simulation Only", style=wx.RB_GROUP)
        self.real_mode_radio = wx.RadioButton(panel, label="Real Robot")
        
        robot_sizer.Add(self.sim_mode_radio, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        robot_sizer.Add(self.real_mode_radio, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        
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
        
        # === Protocol Parameters (matching simforge_genesis) ===
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

        options_sizer.AddSpacer(15)

        options_sizer.Add(wx.StaticText(panel, label="Speed:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.speed_ctrl = wx.SpinCtrlDouble(panel, min=0.01, max=1.0, initial=0.25, inc=0.05)
        self.speed_ctrl.SetDigits(2)
        options_sizer.Add(self.speed_ctrl, 0, wx.ALL, 5)
        self.speed_pct_label = wx.StaticText(panel, label="(25 %)")
        options_sizer.Add(self.speed_pct_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        
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
        
        # Mode radio buttons → trigger mode switch immediately
        self.sim_mode_radio.Bind(wx.EVT_RADIOBUTTON, lambda e: self._on_mode_changed())
        self.real_mode_radio.Bind(wx.EVT_RADIOBUTTON, lambda e: self._on_mode_changed())

        # Speed control → update percentage label
        self.speed_ctrl.Bind(
            wx.EVT_SPINCTRLDOUBLE,
            lambda e: self.speed_pct_label.SetLabel(
                f"({int(self.speed_ctrl.GetValue() * 100)} %)"
            ),
        )
        
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
            move_speed=self.speed_ctrl.GetValue(),
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

    # ─────────────────────────────────────────────────────────────────
    # Immediate mode switching on radio-button change
    # ─────────────────────────────────────────────────────────────────

    def _on_mode_changed(self) -> None:
        """Called when the user clicks the Simulation / Real radio button.

        Immediately triggers a server-side mode switch and freezes the UI
        with a progress dialog until the switch completes (or fails).
        """
        if not self._connected:
            return  # Can't switch if not connected

        if getattr(self, '_mode_switching', False):
            return  # Already switching — ignore duplicate clicks

        mode = "simulation" if self.sim_mode_radio.GetValue() else "both"

        # If we're already in the right mode, nothing to do
        if getattr(self, '_server_mode', None) == mode:
            return

        self._mode_switching = True
        self._mode_ready = False
        self._log(f"Switching server to {mode} mode…")
        self._run_async(self._switch_mode(mode))

    async def _switch_mode(self, mode: str) -> None:
        """Perform the mode switch RPC with a blocking progress dialog.

        The server returns phased progress information so the user can
        see exactly what is happening (teardown → health check → joint
        states → MoveIt → robot programs).  If the switch fails, a
        MessageDialog shows the phase that failed and the error detail.
        """
        dlg_ref: list = []  # mutable container so the callback can store the dialog
        start_time = time.time()

        def _show_progress():
            dlg = wx.ProgressDialog(
                f"Switching to {mode.title()} Mode",
                f"Requesting mode switch to {mode}…\n"
                "Waiting for server to restart the ROS2 stack.\n"
                "This may take 30–60 seconds.",
                maximum=100,
                parent=self,
                style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT,
            )
            dlg.Pulse()
            dlg_ref.append(dlg)

        wx.CallAfter(_show_progress)
        # Give wx a moment to display the dialog
        await asyncio.sleep(0.3)

        # Periodically pulse the dialog to keep it alive and show elapsed time
        async def _pulse_loop():
            while getattr(self, '_mode_switching', False):
                elapsed = int(time.time() - start_time)
                def _update(e=elapsed):
                    if dlg_ref and dlg_ref[0]:
                        dlg_ref[0].Pulse(
                            f"Switching to {mode} mode… ({e}s elapsed)\n"
                            "Server is verifying all services are ready.\n"
                            "Please wait — do not close this window."
                        )
                wx.CallAfter(_update)
                await asyncio.sleep(1.0)

        pulse_task = asyncio.ensure_future(_pulse_loop())

        try:
            response = await self._client.call_rpc(
                "prepare_mode", {"mode": mode}, timeout=120.0
            )

            elapsed = int(time.time() - start_time)

            if response.get("ready"):
                self._server_mode = mode
                self._mode_ready = True
                msg = response.get("message", "")
                self._log(f"✓ {mode} mode ready ({elapsed}s)")
                if msg:
                    self._log(f"  {msg}")

                # Log phase details if provided
                phases = response.get("phases", [])
                for p in phases:
                    self._log(f"  {p.get('phase', '')}: {p.get('detail', '')}")

                # Refresh robot status indicator after mode change
                await self._fetch_robot_status()
            else:
                self._mode_ready = False
                err = response.get("message", "Unknown error")
                self._log(f"✗ Mode switch failed ({elapsed}s): {err}")

                # Log phase details
                phases = response.get("phases", [])
                for p in phases:
                    self._log(f"  {p.get('phase', '')}: {p.get('detail', '')}")

                # Show error dialog with details
                def _show_error(error_msg=err, phase_list=phases):
                    detail_lines = [f"Mode switch to {mode} failed:\n"]
                    detail_lines.append(f"Error: {error_msg}\n")
                    if phase_list:
                        detail_lines.append("Progress phases:")
                        for p in phase_list:
                            detail_lines.append(
                                f"  {p.get('phase', '')}: {p.get('detail', '')}"
                            )
                    wx.MessageDialog(
                        self,
                        "\n".join(detail_lines),
                        "Mode Switch Failed",
                        wx.OK | wx.ICON_ERROR,
                    ).ShowModal()
                wx.CallAfter(_show_error)

                # Revert the radio button to match actual server state
                wx.CallAfter(self._revert_radio_to_server_mode)

        except asyncio.TimeoutError:
            elapsed = int(time.time() - start_time)
            self._mode_ready = False
            self._log(f"✗ Mode switch timed out after {elapsed}s")
            def _show_timeout():
                wx.MessageDialog(
                    self,
                    f"Mode switch to {mode} timed out after {elapsed}s.\n\n"
                    "The server may still be starting up.\n"
                    "Check the server logs and try again.",
                    "Mode Switch Timeout",
                    wx.OK | wx.ICON_WARNING,
                ).ShowModal()
            wx.CallAfter(_show_timeout)
            wx.CallAfter(self._revert_radio_to_server_mode)

        except Exception as e:
            elapsed = int(time.time() - start_time)
            self._mode_ready = False
            self._log(f"✗ Mode switch error ({elapsed}s): {e}")
            def _show_exc(exc=e):
                wx.MessageDialog(
                    self,
                    f"Mode switch to {mode} failed:\n\n{exc}\n\n"
                    "Check the server connection and try again.",
                    "Mode Switch Error",
                    wx.OK | wx.ICON_ERROR,
                ).ShowModal()
            wx.CallAfter(_show_exc)
            wx.CallAfter(self._revert_radio_to_server_mode)

        finally:
            self._mode_switching = False
            pulse_task.cancel()
            # Dismiss the progress dialog
            def _close_dlg():
                if dlg_ref:
                    try:
                        dlg_ref[0].Destroy()
                    except Exception:
                        pass
            wx.CallAfter(_close_dlg)

    def _revert_radio_to_server_mode(self) -> None:
        """Set the radio buttons back to match the last-known server mode."""
        current = getattr(self, '_server_mode', 'simulation')
        if current == 'simulation':
            self.sim_mode_radio.SetValue(True)
        else:
            self.real_mode_radio.SetValue(True)
    
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
        """Fetch real robot connection status and update status indicator.
        
        This no longer disables mode radio buttons — all modes are always
        selectable. The server validates at execution time via prepare_mode.
        """
        if not self._client:
            return
        
        try:
            response = await self._client.call_rpc("get_robot_status", {})
            
            if response.get("success"):
                real_robot_available = response.get("real_robot_available", False)
                connection_details = response.get("connection_details", {})
                robot_program = connection_details.get("robot_program_running", False)
                
                # Update connection status indicator
                wx.CallAfter(
                    self._update_robot_status_indicator,
                    real_robot_available,
                    robot_program,
                    connection_details,
                )
                
                # Log connection status
                if real_robot_available and robot_program:
                    self._log("✓ Real robot connected and program running")
                elif real_robot_available:
                    self._log("⚠ Real robot reachable but program not running")
                else:
                    self._log("⚠ Real robot not connected (will be checked before execution)")
                    for key, value in connection_details.items():
                        if isinstance(value, str):
                            self._log(f"  {key}: {value}")
            else:
                self._log(f"Could not get robot status: {response.get('error', 'Unknown')}")
                wx.CallAfter(self._update_robot_status_indicator, False, False, {})
                
        except Exception as e:
            self._log(f"Error fetching robot status: {e}")
            wx.CallAfter(self._update_robot_status_indicator, False, False, {})
    
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
        """Start protocol simulation.

        The mode switch has *already* happened when the user clicked the
        radio button.  Here we just validate that the mode is ready and
        launch the protocol.
        """
        if not self._connected:
            self._log("Not connected to server")
            return

        if getattr(self, '_mode_switching', False):
            self._log("Mode switch in progress — please wait")
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

        # Determine execution mode from radio buttons
        if self.sim_mode_radio.GetValue():
            mode = "simulation"
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

        self.start_btn.Enable(False)
        self._run_async(self._prepare_and_run(robot, target_object, params, mode, total_poses))
    
    async def _prepare_and_run(
        self,
        robot: str,
        target_object: str,
        params: ProtoSimParameters,
        mode: str,
        total_poses: int,
    ) -> None:
        """Ensure the server is in the right mode, then run the protocol.

        If the mode was already switched via the radio-button handler the
        ``prepare_mode`` RPC is skipped.  Otherwise it is called as a
        fallback (e.g. first run where the user never toggled the radio).
        """
        try:
            # Check if we need to call prepare_mode
            need_prepare = (
                getattr(self, '_server_mode', None) != mode
                or not getattr(self, '_mode_ready', False)
            )

            if need_prepare:
                wx.CallAfter(
                    self.progress_label.SetLabel,
                    f"Switching to {mode} mode…"
                )
                self._log(f"Switching to {mode} mode…")

                response = await self._client.call_rpc(
                    "prepare_mode", {"mode": mode}, timeout=120.0
                )

                if response.get("ready"):
                    self._server_mode = mode
                    self._mode_ready = True
                    self._log(f"✓ {mode} mode ready")
                    self._log(response.get("message", ""))
                    for p in response.get("phases", []):
                        self._log(f"  {p.get('phase','')}: {p.get('detail','')}")
                else:
                    msg = response.get("message", "Robot not ready")
                    self._log(f"✗ {msg}")
                    for p in response.get("phases", []):
                        self._log(f"  {p.get('phase','')}: {p.get('detail','')}")
                    can_retry = response.get("can_retry", False)
                    user_choice = await self._show_mode_check_dialog(mode, msg, can_retry)
                    if user_choice == "retry":
                        self._log("Retrying mode switch...")
                        await self._prepare_and_run(robot, target_object, params, mode, total_poses)
                        return
                    elif user_choice == "simulation":
                        self._log("Switching to Simulation Only mode")
                        wx.CallAfter(self.sim_mode_radio.SetValue, True)
                        self._server_mode = "simulation"
                        self._mode_ready = True
                        await self._prepare_and_run(
                            robot, target_object, params, "simulation", total_poses
                        )
                        return
                    else:
                        self._log("Cancelled by user")
                        wx.CallAfter(self.start_btn.Enable, True)
                        wx.CallAfter(self.progress_label.SetLabel, "Ready")
                        return

            # Mode is ready — proceed with protocol execution
            wx.CallAfter(self.progress_label.SetLabel, "Running protocol…")
            self._running_simulation = True
            wx.CallAfter(self.stop_btn.Enable, True)
            await self._run_protocol(robot, target_object, params, mode)

        except Exception as e:
            self._log(f"Mode switch error: {e}")
            user_choice = await self._show_mode_check_dialog(
                mode,
                f"Could not reach server for mode switch:\n{e}",
                can_retry=True,
            )
            if user_choice == "simulation":
                self._log("Falling back to Simulation Only mode")
                wx.CallAfter(self.sim_mode_radio.SetValue, True)
                self._server_mode = "simulation"
                self._mode_ready = True
                await self._prepare_and_run(
                    robot, target_object, params, "simulation", total_poses
                )
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
            
            # Use activity-based inactivity timeout instead of a fixed
            # total budget.  The server sends periodic rpc_feedback
            # messages during planning and execution; as long as those
            # keep arriving within the inactivity window the call stays
            # alive — no matter how long cuRobo planning or trajectory
            # execution takes.  The timeout only fires if the server
            # goes completely silent (crash, network partition, etc.).
            INACTIVITY_TIMEOUT = 120.0   # seconds of server silence
            
            move_speed = getattr(params, 'move_speed', 0.25)
            
            self._log(
                f"Sending {total_poses} poses to server "
                f"(speed={move_speed}, inactivity_timeout={INACTIVITY_TIMEOUT:.0f}s)"
            )

            # Feedback callback — updates progress bar & label in real time
            def _on_feedback(msg):
                status = msg.get("status", "")
                pct = msg.get("progress_percent", 0)
                pose_idx = msg.get("current_pose_index", 0)
                pose_name = msg.get("current_pose_name", "")
                message = msg.get("message", "")

                if status == "paused":
                    label = message or "⏸ Safeguard stop — waiting…"
                elif status == "fault_paused":
                    label = message or "⚠ Hardware fault — waiting for recovery…"
                elif status == "resuming":
                    label = message or "▶ Resuming…"
                elif status == "planning":
                    label = message or f"Planning {pose_idx+1}/{total_poses}"
                elif status == "moving":
                    label = f"Moving to {pose_name or pose_idx+1}"
                elif status == "reached":
                    label = f"Reached {pose_name or pose_idx+1}"
                elif status == "error":
                    label = message or "Error"
                else:
                    label = message or status or "Working…"

                wx.CallAfter(self.progress_bar.SetValue, int(pct))
                wx.CallAfter(self.progress_label.SetLabel, label)

                # Log significant events
                if status in ("paused", "fault_paused", "resuming", "error"):
                    self._log(label)
            
            # ── Decide: inline send vs chunked upload ────────────
            # For small protocols (≤500 poses, ≈140 KiB) send inline
            # in a single RPC.  For large protocols use the chunked
            # upload flow to stay well within WebSocket frame limits
            # and to show upload progress to the user.
            CHUNK_THRESHOLD = 500   # poses
            CHUNK_SIZE      = 500   # poses per chunk

            common_params = {
                "robot_name": robot,
                "idle_time": params.idle_time,
                "mode": mode,
                "move_speed": move_speed,
                "go_home_before": True,
                "go_home_after": True,
            }

            if total_poses <= CHUNK_THRESHOLD:
                # ── Small protocol: single RPC ───────────────────
                response = await self._client.call_rpc("run_proto_sim", {
                    **common_params,
                    "poses": poses_data,
                }, inactivity_timeout=INACTIVITY_TIMEOUT,
                   feedback_callback=_on_feedback)
            else:
                # ── Large protocol: chunked upload ───────────────
                self._log(
                    f"Large protocol ({total_poses} poses) — "
                    f"using chunked upload ({CHUNK_SIZE} poses/chunk)"
                )
                wx.CallAfter(
                    self.progress_label.SetLabel,
                    f"Uploading poses… 0/{total_poses}",
                )

                # 1. Initialise upload session
                init_resp = await self._client.call_rpc(
                    "proto_upload_init",
                    {**common_params, "total_poses": total_poses},
                    timeout=30.0,
                )
                if not init_resp.get("success"):
                    raise RuntimeError(
                        f"Upload init failed: "
                        f"{init_resp.get('error', init_resp.get('message', '?'))}"
                    )
                upload_id = init_resp["upload_id"]
                self._log(f"Upload session {upload_id} created")

                # 2. Send pose chunks
                for chunk_start in range(0, total_poses, CHUNK_SIZE):
                    chunk_end = min(chunk_start + CHUNK_SIZE, total_poses)
                    chunk = poses_data[chunk_start:chunk_end]

                    chunk_resp = await self._client.call_rpc(
                        "proto_upload_chunk",
                        {"upload_id": upload_id, "poses": chunk},
                        timeout=30.0,
                    )
                    if not chunk_resp.get("success"):
                        raise RuntimeError(
                            f"Upload chunk failed: "
                            f"{chunk_resp.get('error', '?')}"
                        )
                    received = chunk_resp.get("received", chunk_end)
                    pct = chunk_resp.get("progress_percent", 0)
                    wx.CallAfter(
                        self.progress_label.SetLabel,
                        f"Uploading poses… {received}/{total_poses} "
                        f"({pct:.0f}%)",
                    )

                self._log(
                    f"All {total_poses} poses uploaded — "
                    f"starting execution"
                )
                wx.CallAfter(
                    self.progress_label.SetLabel,
                    "Upload complete — starting execution…",
                )

                # 3. Execute the uploaded protocol
                response = await self._client.call_rpc(
                    "proto_upload_execute",
                    {"upload_id": upload_id},
                    inactivity_timeout=INACTIVITY_TIMEOUT,
                    feedback_callback=_on_feedback,
                )
            
            if response.get("success"):
                completed = response.get('completed', 0)
                collision_rejected = response.get('collision_rejected', 0)
                ik_failed = response.get('ik_failed', 0)
                real_failed = response.get('real_failed', 0)
                self._log(f"Protocol completed: {completed}/{total_poses} poses")
                if collision_rejected > 0:
                    self._log(f"  Collision rejected: {collision_rejected}")
                if ik_failed > 0:
                    self._log(f"  IK failed: {ik_failed}")
                if real_failed > 0:
                    self._log(f"  Execution failed: {real_failed}")
            else:
                self._log(f"Protocol failed: {response.get('error', response.get('message', 'Unknown error'))}")

            # Surface any hardware issues reported by the server
            hw_issues = response.get("hardware_issues", [])
            hw_abort = response.get("hardware_abort", False)

            # Classify issues: safeguard stops are recoverable
            safeguard_only = (
                hw_issues
                and all(iss.endswith(":SAFEGUARD_STOP") for iss in hw_issues)
            )

            if hw_issues:
                if safeguard_only and not hw_abort:
                    # Safeguard stops occurred but were auto-resolved
                    self._log(
                        f"  ℹ Safeguard stop(s) occurred and auto-cleared: "
                        f"{', '.join(hw_issues)}"
                    )
                else:
                    self._log(f"  ⚠ Hardware issues: {', '.join(hw_issues)}")

            if hw_abort:
                if safeguard_only:
                    # Safeguard stop did NOT clear in time
                    self._log(
                        "  ⏸ Protocol ABORTED — safeguard stop "
                        "did not clear within timeout"
                    )
                    def _show_sg_error(issues=hw_issues, msg=response.get("message", "")):
                        detail = (
                            "The robot paused due to a safeguard stop "
                            "(laser scanner / safety zone) but the area "
                            "was not cleared within the timeout.\n\n"
                            "Safeguard events:\n"
                            + "\n".join(f"  • {iss}" for iss in issues)
                            + "\n\n"
                            f"Server message:\n  {msg}\n\n"
                            "Clear the safety zone around the robot and "
                            "try again. The robot will resume automatically "
                            "once the area is clear."
                        )
                        wx.MessageDialog(
                            self, detail,
                            "Safeguard Stop — Protocol Paused",
                            wx.OK | wx.ICON_WARNING,
                        ).ShowModal()
                    wx.CallAfter(_show_sg_error)
                else:
                    # Fatal fault — protocol either recovered and
                    # continued, or was aborted after the user
                    # pressed Stop during the recovery wait.
                    completed = response.get('completed', 0)
                    total_p = response.get('total', total_poses)
                    stopped = response.get('stopped', False)

                    if stopped:
                        self._log(
                            f"  ⛔ Protocol STOPPED after hardware "
                            f"fault at {completed}/{total_p} poses"
                        )
                    else:
                        self._log(
                            f"  ⚠ Protocol completed with hardware "
                            f"fault recovery: {completed}/{total_p}"
                        )

                    def _show_hw_error(
                        issues=hw_issues,
                        msg=response.get("message", ""),
                        _completed=completed,
                        _total=total_p,
                        _stopped=stopped,
                    ):
                        if _stopped:
                            title = "Robot Hardware Fault — Protocol Stopped"
                            icon = wx.ICON_ERROR
                            action = (
                                f"The protocol was stopped at pose "
                                f"{_completed}/{_total}.\n\n"
                                "To resume from where you left off, "
                                "reset the robot, put it back in remote "
                                "control, and re-run the protocol.\n\n"
                                "The server will automatically pause and "
                                "wait for recovery if a fault occurs "
                                "during execution."
                            )
                        else:
                            title = "Hardware Fault Recovered"
                            icon = wx.ICON_WARNING
                            action = (
                                f"The robot recovered automatically "
                                f"and completed {_completed}/{_total} "
                                f"poses."
                            )
                        detail = (
                            "A hardware fault occurred during protocol "
                            "execution.\n\n"
                            "Hardware faults:\n"
                            + "\n".join(f"  • {iss}" for iss in issues)
                            + "\n\n"
                            f"Server message:\n  {msg}\n\n"
                            + action
                        )
                        wx.MessageDialog(
                            self, detail, title,
                            wx.OK | icon,
                        ).ShowModal()
                    wx.CallAfter(_show_hw_error)
            elif response.get("stopped"):
                self._log("  Protocol was stopped by user")
                
        except Exception as e:
            error_str = str(e) if str(e) else repr(e)
            self._log(f"Protocol error: {error_str}")
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
            "Parameters match simforge_genesis/interfaces/gui/proto_sim.py",
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
