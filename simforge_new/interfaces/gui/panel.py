"""wxPython-based GUI for the Simforge control stack."""
from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import suppress
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Any

try:  # GUI dependency is optional
    import wx

    HAS_WX = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_WX = False
    wx = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing only
    import wx as wx_types

    SliderType = wx_types.Slider
    TextCtrlType = wx_types.TextCtrl
    ChoiceType = wx_types.Choice
    EventType = wx_types.Event
else:  # Fallback types when wx is unavailable at runtime (e.g., during linting)
    SliderType = Any
    TextCtrlType = Any
    ChoiceType = Any
    EventType = Any

import numpy as np

from ...core import Backend
from ...core.commands import CartesianMoveCommand, JointTargetsCommand
from ...control.session import SimulationSession
from ...logging import setup_logging

JOINT_SLIDER_MIN = -180
JOINT_SLIDER_MAX = 180
DEFAULT_CART_POSITION = (0.4, 0.4, 0.4)
DEFAULT_CART_ORIENTATION = (0.0, 0.0, 0.0)
UPDATE_INTERVAL_MS = 300


class SessionController:
    """Threaded wrapper around :class:`SimulationSession` for GUI usage."""

    def __init__(self, config_path: str, backend: Backend, logger: logging.Logger) -> None:
        self._config_path = str(config_path)
        self._backend = backend
        self._logger = logger
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="SimforgeSessionLoop", daemon=True)
        self._session: Optional[SimulationSession] = None
        self._stopped = False
        self._thread.start()
        future = self._run(self._create_session())
        self._session = future.result()
        self.spec = self._session.spec
        self._profiles = {robot.name: robot for robot in self.spec.robots}

    # ------------------------------------------------------------------
    # Loop helpers
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:  # graceful shutdown
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                with suppress(asyncio.CancelledError):
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()

    def _run(self, coro: asyncio.Future) -> "asyncio.Future[object]":
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _create_session(self) -> SimulationSession:
        return await SimulationSession.create(self._config_path, backend=self._backend, logger=self._logger)

    # ------------------------------------------------------------------
    # Public API consumed by the GUI
    # ------------------------------------------------------------------
    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def robot_names(self) -> List[str]:
        return list(self._profiles.keys())

    def joint_count(self, robot_name: str) -> int:
        return self._profiles[robot_name].joint_count

    def home_joints(self, robot_name: str) -> List[float]:
        profile = self._profiles[robot_name]
        if profile.initial_joint_positions_deg:
            return list(profile.initial_joint_positions_deg)
        return [0.0] * profile.joint_count

    def get_reference_frames(self, robot_name: str) -> List[Tuple[str, str]]:  # noqa: ARG002 - robot for future use
        frames: List[Tuple[str, str]] = [("base", "Robot Base")]
        if self._session is None:
            return frames
        mapping = self._session.reference_frames()
        if "world" in mapping:
            frames.append(("world", "World"))
        for key in sorted(mapping):
            if key == "world":
                continue
            label = key.split(":", 1)[1] if key.startswith("obj:") else key
            frames.append((key, label))
        return frames

    def get_joint_positions(self, robot_name: str) -> List[float]:
        future = self._run(self._fetch_state(robot_name))
        try:
            state = future.result()
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.debug("Failed to fetch state for %s: %s", robot_name, exc)
            return []
        if state is None:
            return []
        return list(np.rad2deg(state.joint_positions))

    def set_joint_position(self, robot_name: str, joint_idx: int, value_deg: float, duration: float = 0.5) -> None:
        future = self._run(self._set_joint_position(robot_name, joint_idx, value_deg, duration))
        with suppress(Exception):  # non-blocking UI; errors already logged
            future.result()

    def move_cartesian(
        self,
        robot_name: str,
        position: Tuple[float, float, float],
        orientation_deg: Tuple[float, float, float],
        frame: str,
        duration: float = 4.0,
    ) -> None:
        future = self._run(self._move_cartesian(robot_name, position, orientation_deg, frame, duration))
        with suppress(Exception):  # errors already logged
            future.result()

    def shutdown(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._session is not None:
            future = self._run(self._session.close())
            with suppress(Exception):
                future.result(timeout=15)
            self._session = None
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=15)

    # ------------------------------------------------------------------
    # Async helpers
    # ------------------------------------------------------------------
    async def _fetch_state(self, robot_name: str):
        if self._session is None:
            return None
        state = self._session.get_latest_state(robot_name)
        if state is not None:
            return state
        with suppress(TimeoutError):
            return await self._session.wait_for_robot_state(robot_name, timeout=0.5)
        return None

    async def _set_joint_position(self, robot_name: str, joint_idx: int, value_deg: float, duration: float) -> None:
        state = await self._fetch_state(robot_name)
        if state is None:
            self._logger.debug("No state available for %s; joint command skipped", robot_name)
            return
        values = list(np.rad2deg(state.joint_positions))
        if joint_idx >= len(values):
            self._logger.warning("Joint index %s out of range for %s", joint_idx, robot_name)
            return
        values[joint_idx] = value_deg
        command = JointTargetsCommand(
            robot_name=robot_name,
            values_deg=values,
            duration=float(max(duration, 0.05)),
            metadata={
                "source": "gui_slider",
                "direct_joint_set": True,
                "queue_mode": "interrupt",
            },
        )
        await self._session.send_command(command)

    async def _move_cartesian(
        self,
        robot_name: str,
        position: Tuple[float, float, float],
        orientation_deg: Tuple[float, float, float],
        frame: str,
        duration: float,
    ) -> None:
        if self._session is None:
            return
        command = CartesianMoveCommand(
            robot_name=robot_name,
            position_m=[float(v) for v in position],
            orientation_deg=[float(v) for v in orientation_deg],
            duration=float(max(duration, 0.5)),
            reference_frame=frame,
            use_collision_checking=True,
            metadata={
                "source": "gui_cartesian",
                "queue_mode": "interrupt",
            },
        )
        await self._session.send_command(command)


class RobotControlFrame(wx.Frame):
    """Main wxPython frame presenting joint and Cartesian controls."""

    def __init__(self, controller: SessionController, debug: bool = False):
        super().__init__(
            parent=None,
            title="Simforge Robot Control",
            size=(600, 700),
            style=wx.DEFAULT_FRAME_STYLE & ~(wx.RESIZE_BORDER | wx.MAXIMIZE_BOX),
        )
        self.controller = controller
        self.logger = controller.logger
        self.debug = debug
        self.joint_sliders: Dict[str, List[SliderType]] = {}
        self.cart_fields: Dict[str, List[TextCtrlType]] = {}
        self.cart_frame_choice: Dict[str, ChoiceType] = {}
        self.cart_frame_map: Dict[str, Dict[str, str]] = {}
        self._building = True
        self._running = True

        self._setup_gui()
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_timer, self.timer)
        self.timer.Start(UPDATE_INTERVAL_MS)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self._building = False
        self.logger.info("Simforge GUI initialised")

    # ------------------------------------------------------------------
    # GUI assembly helpers
    # ------------------------------------------------------------------
    def _setup_gui(self) -> None:
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.status_bar = self.CreateStatusBar()
        self.status_bar.SetStatusText("Ready")

        self.notebook = wx.Notebook(panel)
        vbox.Add(self.notebook, 1, wx.EXPAND | wx.ALL, 5)

        for robot_name in self.controller.robot_names():
            self._create_robot_tab(robot_name)

        panel.SetSizer(vbox)
        self.Show()
        self.Raise()

    def _create_robot_tab(self, robot_name: str) -> None:
        tab = wx.Panel(self.notebook)
        tab_sizer = wx.BoxSizer(wx.VERTICAL)

        joint_box = wx.StaticBoxSizer(wx.VERTICAL, tab, "Joint Control (degrees)")
        sliders: List[SliderType] = []
        joint_count = self.controller.joint_count(robot_name)
        home = self.controller.home_joints(robot_name)

        for joint_idx in range(joint_count):
            row = wx.BoxSizer(wx.HORIZONTAL)
            label = wx.StaticText(tab, label=f"J{joint_idx + 1}:")
            label.SetMinSize((30, -1))
            row.Add(label, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)

            slider = wx.Slider(
                tab,
                value=int(home[joint_idx]) if joint_idx < len(home) else 0,
                minValue=JOINT_SLIDER_MIN,
                maxValue=JOINT_SLIDER_MAX,
                style=wx.SL_HORIZONTAL | wx.SL_VALUE_LABEL,
                size=(300, -1),
            )
            slider.Bind(
                wx.EVT_SLIDER,
                lambda evt, rn=robot_name, idx=joint_idx: self._on_joint_slider(rn, idx, evt),
            )
            row.Add(slider, 1, wx.EXPAND | wx.RIGHT, 10)

            value_text = wx.StaticText(tab, label=f"{slider.GetValue()}°")
            value_text.SetMinSize((50, -1))
            row.Add(value_text, 0, wx.ALIGN_CENTER_VERTICAL)

            joint_box.Add(row, 0, wx.EXPAND | wx.ALL, 3)
            sliders.append(slider)

        self.joint_sliders[robot_name] = sliders
        tab_sizer.Add(joint_box, 0, wx.EXPAND | wx.ALL, 5)

        cart_box = wx.StaticBoxSizer(wx.VERTICAL, tab, "Cartesian Control")
        cart_grid = wx.FlexGridSizer(rows=3, cols=4, hgap=15, vgap=12)
        cart_grid.AddGrowableCol(1, 1)
        cart_grid.AddGrowableCol(3, 1)

        pos_fields: List[TextCtrlType] = []
        ori_fields: List[TextCtrlType] = []

        for idx, (pos_label, ori_label, default_pos, default_ori) in enumerate(
            zip(
                ("X (m):", "Y (m):", "Z (m):"),
                ("Roll (°):", "Pitch (°):", "Yaw (°):"),
                DEFAULT_CART_POSITION,
                DEFAULT_CART_ORIENTATION,
            )
        ):
            pos_lbl = wx.StaticText(tab, label=pos_label)
            pos_lbl.SetMinSize((60, -1))
            pos_field = wx.TextCtrl(tab, value=f"{default_pos:.3f}", size=(100, -1))

            ori_lbl = wx.StaticText(tab, label=ori_label)
            ori_lbl.SetMinSize((70, -1))
            ori_field = wx.TextCtrl(tab, value=f"{default_ori:.1f}", size=(100, -1))

            cart_grid.Add(pos_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
            cart_grid.Add(pos_field, 1, wx.EXPAND)
            cart_grid.Add(ori_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
            cart_grid.Add(ori_field, 1, wx.EXPAND)

            pos_fields.append(pos_field)
            ori_fields.append(ori_field)

        cart_box.Add(cart_grid, 0, wx.EXPAND | wx.ALL, 15)

        frame_sizer = wx.BoxSizer(wx.HORIZONTAL)
        frame_label = wx.StaticText(tab, label="Reference Frame:")
        frame_label.SetMinSize((120, -1))
        frames = self.controller.get_reference_frames(robot_name)
        frame_choice = wx.Choice(tab, choices=[label for _, label in frames] or ["Robot Base"])
        if frame_choice.GetCount() > 0:
            frame_choice.SetSelection(0)
        self.cart_frame_choice[robot_name] = frame_choice
        self.cart_frame_map[robot_name] = {label: key for key, label in frames} if frames else {"Robot Base": "base"}
        frame_sizer.Add(frame_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        frame_sizer.Add(frame_choice, 0, wx.ALIGN_CENTER_VERTICAL)
        cart_box.Add(frame_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)

        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        move_btn = wx.Button(tab, label="🎯 Move to Pose", size=(150, 35))
        move_btn.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        move_btn.SetBackgroundColour(wx.Colour(70, 130, 180))
        move_btn.SetForegroundColour(wx.Colour(255, 255, 255))
        move_btn.Bind(wx.EVT_BUTTON, lambda evt, rn=robot_name: self._on_move_cartesian(rn))
        button_sizer.AddStretchSpacer()
        button_sizer.Add(move_btn, 0, wx.ALIGN_CENTER)
        button_sizer.AddStretchSpacer()
        cart_box.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 10)

        self.cart_fields[robot_name] = pos_fields + ori_fields
        tab_sizer.Add(cart_box, 0, wx.EXPAND | wx.ALL, 5)

        tab.SetSizer(tab_sizer)
        self.notebook.AddPage(tab, robot_name)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_joint_slider(self, robot_name: str, joint_idx: int, event: EventType) -> None:
        if self._building:
            return
        value_deg = float(event.GetInt())
        self.status_bar.SetStatusText(f"{robot_name} J{joint_idx + 1}: {value_deg:.1f}°")
        threading.Thread(
            target=self.controller.set_joint_position,
            args=(robot_name, joint_idx, value_deg),
            kwargs={"duration": 0.05},
            daemon=True,
        ).start()

    def _on_move_cartesian(self, robot_name: str) -> None:
        fields = self.cart_fields[robot_name]
        try:
            values = []
            for idx, field in enumerate(fields):
                raw = field.GetValue().strip()
                if not raw:
                    default_vals = (*DEFAULT_CART_POSITION, *DEFAULT_CART_ORIENTATION)
                    values.append(default_vals[idx])
                else:
                    values.append(float(raw))
        except ValueError:
            wx.MessageBox("Invalid Cartesian input", "Input Error", wx.OK | wx.ICON_ERROR)
            return

        position = tuple(values[:3])
        orientation = tuple(values[3:])

        frame_key = "base"
        frame_choice = self.cart_frame_choice.get(robot_name)
        frame_map = self.cart_frame_map.get(robot_name, {})
        if frame_choice and frame_choice.GetCount() > 0:
            if frame_choice.GetSelection() == wx.NOT_FOUND:
                frame_choice.SetSelection(0)
            selected_label = frame_choice.GetStringSelection()
            frame_key = frame_map.get(selected_label, frame_key)

        self.status_bar.SetStatusText(
            f"{robot_name} moving to {position} {orientation} (frame={frame_key})"
        )
        threading.Thread(
            target=self.controller.move_cartesian,
            args=(robot_name, position, orientation, frame_key),
            daemon=True,
        ).start()

    def _on_timer(self, event: EventType) -> None:
        if not self._running:
            return
        for robot_name, sliders in self.joint_sliders.items():
            positions = self.controller.get_joint_positions(robot_name)
            if not positions:
                continue
            for idx, slider in enumerate(sliders):
                if idx >= len(positions):
                    continue
                new_val = int(round(positions[idx]))
                if slider.GetValue() != new_val:
                    slider.SetValue(new_val)

    def _on_close(self, event: EventType) -> None:
        self._running = False
        try:
            if hasattr(self, "timer") and self.timer.IsRunning():
                self.timer.Stop()
        finally:
            self.Destroy()


def run_gui(
    config_path: str,
    *,
    backend: Backend = Backend.GPU,
    log_level: str = "INFO",
    debug: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Launch the wxPython GUI for Simforge."""
    if not HAS_WX:
        raise RuntimeError("wxPython is required for the GUI. Install with 'pip install wxpython'.")

    level_value = getattr(logging, log_level.upper(), logging.INFO)
    if logger is None:
        logger = setup_logging(level_value, debug=debug).getChild("gui")
    else:
        logger.setLevel(level_value)

    controller = SessionController(config_path, backend, logger)
    try:
        app = wx.App(False)
        frame = RobotControlFrame(controller, debug=debug)
        app.MainLoop()
    finally:
        controller.shutdown()


__all__ = ["run_gui", "HAS_WX", "SessionController", "RobotControlFrame"]
