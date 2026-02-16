"""wxPython interface for the ``proto_sim`` workflow."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .panel import HAS_WX, SessionController, wx
from ...core import Backend
from ...logging import setup_logging
from ...control.proto_simulation import (
    ProtoSimParameters,
    ProtoSimProgress,
    ProtoSimRunResult,
    format_summary_lines,
)


LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


class ParameterInputControl:
    """Helper wrapper around the mode toggle + value entry widgets."""

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
            except ValueError as exc:  # pragma: no cover - GUI validation
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
        except ValueError as exc:  # pragma: no cover - GUI validation
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
        if not values:
            raise ValueError("Range did not produce any samples")
        return values


class ProtoSimFrame(wx.Frame):
    """Main frame for the proto_sim workflow."""

    PARAMETERS = (
        {"name": "horiz", "label": "Horiz Shift", "unit": "mm", "mode": "List", "default": "0"},
        {"name": "vert", "label": "Vert Shift", "unit": "mm", "mode": "List", "default": "0"},
        {"name": "distance", "label": "Distance", "unit": "mm", "mode": "List", "default": "250,350,450,550"},
        {"name": "roll", "label": "Roll", "unit": "°", "mode": "List", "default": "-90"},
        {"name": "pitch", "label": "Pitch", "unit": "°", "mode": "List", "default": "-45,-30,0,15"},
        {"name": "yaw", "label": "Yaw", "unit": "°", "mode": "Range", "default": "-30,30,30"},
    )

    def __init__(self, controller: SessionController, logger: logging.Logger, debug: bool = False) -> None:
        super().__init__(
            parent=None,
            title="Simforge Proto Simulation",
            size=(720, 1000),
            style=wx.DEFAULT_FRAME_STYLE & ~(wx.RESIZE_BORDER | wx.MAXIMIZE_BOX),
        )
        self.controller = controller
        self.logger = logger
        self.debug = debug
        self.parameter_controls: Dict[str, ParameterInputControl] = {}
        self._current_future = None
        self._last_result: Optional[ProtoSimRunResult] = None

        self._build_ui()
        self.Centre()
        self.Show()
        self.Raise()

    # ------------------------------------------------------------------
    # UI setup helpers
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.status_bar = self.CreateStatusBar()
        self.status_bar.SetStatusText("Ready")

        vbox.Add(self._build_robot_selector(panel), 0, wx.EXPAND | wx.ALL, 10)
        vbox.Add(self._build_parameter_section(panel), 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        vbox.Add(self._build_object_section(panel), 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        vbox.Add(self._build_action_section(panel), 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        vbox.Add(self._build_log_section(panel), 1, wx.EXPAND | wx.ALL, 10)

        panel.SetSizer(vbox)

    def _build_robot_selector(self, parent: wx.Window) -> wx.Sizer:
        box = wx.StaticBoxSizer(wx.VERTICAL, parent, "Robot")
        robots = self.controller.robot_names()
        self.robot_choice = wx.Choice(parent, choices=robots or ["<none>"])
        if robots:
            self.robot_choice.SetSelection(0)
        else:
            self.robot_choice.Enable(False)
        box.Add(self.robot_choice, 0, wx.EXPAND | wx.ALL, 8)
        return box

    def _build_parameter_section(self, parent: wx.Window) -> wx.Sizer:
        box = wx.StaticBoxSizer(wx.VERTICAL, parent, "Movement Parameters")
        grid = wx.FlexGridSizer(rows=len(self.PARAMETERS), cols=3, hgap=10, vgap=10)
        grid.AddGrowableCol(2, 1)
        for item in self.PARAMETERS:
            label_text = f"{item['label']} ({item['unit']})"
            label = wx.StaticText(parent, label=label_text)
            control = ParameterInputControl(
                parent,
                name=item["name"],
                label=item["label"],
                unit=item["unit"],
                default_mode=item["mode"],
                default_text=item["default"],
            )
            self.parameter_controls[item["name"]] = control
            grid.Add(label, 0, wx.ALIGN_CENTER_VERTICAL)
            grid.Add(control.mode_choice, 0, wx.EXPAND)
            grid.Add(control.text_ctrl, 1, wx.EXPAND)
        box.Add(grid, 1, wx.EXPAND | wx.ALL, 8)
        return box

    def _build_object_section(self, parent: wx.Window) -> wx.Sizer:
        box = wx.StaticBoxSizer(wx.VERTICAL, parent, "Objects")
        objects = self.controller.environment_objects()
        self.object_list = wx.CheckListBox(parent, choices=objects)
        self.object_list.SetMinSize((660, 160))
        box.Add(self.object_list, 1, wx.EXPAND | wx.ALL, 8)
        hint = wx.StaticText(parent, label="Select one or more objects to simulate.")
        hint.SetForegroundColour(wx.Colour(90, 90, 90))
        box.Add(hint, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        return box

    def _build_action_section(self, parent: wx.Window) -> wx.Sizer:
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.simulate_button = wx.Button(parent, label="Simulate Movement", size=(200, 40))
        self.simulate_button.Bind(wx.EVT_BUTTON, self._on_simulate_clicked)
        self.export_button = wx.Button(parent, label="Export Plan…", size=(150, 35))
        self.export_button.Bind(wx.EVT_BUTTON, self._on_export_clicked)
        self.export_button.Enable(False)
        self.deploy_button = wx.Button(parent, label="Deploy to Robot", size=(150, 35))
        self.deploy_button.Bind(wx.EVT_BUTTON, self._on_deploy_clicked)
        self.deploy_button.Enable(False)
        self.deploy_button.SetBackgroundColour(wx.Colour(200, 230, 200))  # Light green
        hbox.AddStretchSpacer()
        hbox.Add(self.simulate_button, 0, wx.RIGHT, 10)
        hbox.Add(self.export_button, 0, wx.RIGHT, 10)
        hbox.Add(self.deploy_button, 0)
        hbox.AddStretchSpacer()
        return hbox

    def _build_log_section(self, parent: wx.Window) -> wx.Sizer:
        box = wx.StaticBoxSizer(wx.VERTICAL, parent, "Progress Log")
        self.log_output = wx.TextCtrl(
            parent,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2,
            size=(-1, 220),
        )
        self.log_output.SetMinSize((660, 220))
        box.Add(self.log_output, 1, wx.EXPAND | wx.ALL, 8)
        return box

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_simulate_clicked(self, event):  # noqa: D401, ANN001 - wx signature
        if not self.controller.robot_names():
            wx.MessageBox("No robots available in the environment.", "Configuration", wx.OK | wx.ICON_WARNING)
            return
        try:
            robot_name = self.robot_choice.GetStringSelection()
        except Exception:  # pragma: no cover - defensive
            wx.MessageBox("Select a robot before starting the simulation.", "Input Error", wx.OK | wx.ICON_ERROR)
            return

        selected_objects = [self.object_list.GetString(i) for i in range(self.object_list.GetCount()) if self.object_list.IsChecked(i)]
        if not selected_objects:
            wx.MessageBox("Please select at least one object.", "Input Error", wx.OK | wx.ICON_ERROR)
            return

        try:
            params = self._collect_parameters()
        except ValueError as exc:
            wx.MessageBox(str(exc), "Input Error", wx.OK | wx.ICON_ERROR)
            return

        self._prepare_run_ui()
        self.status_bar.SetStatusText("Simulating …")
        self.logger.info("Starting proto_sim run: robot=%s objects=%s", robot_name, selected_objects)

        def progress_callback(progress: ProtoSimProgress) -> None:
            wx.CallAfter(self._handle_progress, progress)

        try:
            future = self.controller.execute_proto_sim(
                robot_name,
                params,
                selected_objects,
                idle_timeout=20.0,
                progress=progress_callback,
            )
        except Exception as exc:  # pragma: no cover - early validation
            wx.MessageBox(str(exc), "Simulation Error", wx.OK | wx.ICON_ERROR)
            self._finalise_run_ui(success=False)
            return

        self._current_future = future
        future.add_done_callback(self._handle_completion)

    def _on_export_clicked(self, event):  # noqa: D401, ANN001 - wx signature
        if not self._last_result:
            wx.MessageBox("No plan available to export.", "Export", wx.OK | wx.ICON_INFORMATION)
            return
        plan = self._last_result.build_plan_export()
        if not plan:
            wx.MessageBox("No successful poses to export.", "Export", wx.OK | wx.ICON_INFORMATION)
            return

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        default_name = f"proto_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with wx.FileDialog(
            self,
            message="Export plan",
            defaultDir=str(LOG_DIR),
            defaultFile=default_name,
            wildcard="JSON files (*.json)|*.json",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as dialog:
            if dialog.ShowModal() != wx.ID_OK:
                return
            target_path = Path(dialog.GetPath())

        try:
            with target_path.open("w", encoding="utf-8") as fh:
                json.dump(plan, fh, indent=2)
            self.logger.info("Exported proto plan to %s", target_path)
            wx.MessageBox(f"Plan exported to {target_path}", "Export", wx.OK | wx.ICON_INFORMATION)
        except Exception as exc:  # pragma: no cover - filesystem errors
            self.logger.exception("Failed to export plan: %s", exc)
            wx.MessageBox(f"Failed to export plan: {exc}", "Export Error", wx.OK | wx.ICON_ERROR)

    # ------------------------------------------------------------------
    # Progress + completion handling
    # ------------------------------------------------------------------

    def _handle_progress(self, progress: ProtoSimProgress) -> None:
        execution = progress.execution
        outcome = "SUCCESS" if execution.success else f"FAILED ({execution.failure_reason or 'unknown'})"
        message = (
            f"[{progress.object_name}] Pose {progress.pose_index}/{progress.total_poses}: {outcome}"
            f" | params={execution.pose.parameters}"
        )
        self.log_output.AppendText(message + "\n")

    def _handle_completion(self, future) -> None:  # noqa: D401 - background callback
        def _finalise():
            self._current_future = None
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.exception("proto_sim run failed: %s", exc)
                wx.MessageBox(f"Simulation failed: {exc}", "Simulation Error", wx.OK | wx.ICON_ERROR)
                self._finalise_run_ui(success=False)
                return

            self._last_result = result
            self._write_summary_log(result)
            summary_lines = format_summary_lines(result)
            self.log_output.AppendText("\n" + "\n".join(summary_lines) + "\n")
            successes = result.total_successes()
            failures = result.total_failures()
            self.status_bar.SetStatusText(f"Simulation complete: {successes} success / {failures} failure")
            self.logger.info("proto_sim run complete: %s success, %s failure", successes, failures)
            self._finalise_run_ui(bool(successes > 0 or result.home_trajectory))

        wx.CallAfter(_finalise)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_parameters(self) -> ProtoSimParameters:
        values: Dict[str, List[float]] = {}
        for name, control in self.parameter_controls.items():
            values[name] = control.get_values()
        return ProtoSimParameters(
            horiz=values["horiz"],
            vert=values["vert"],
            distance=values["distance"],
            roll=values["roll"],
            pitch=values["pitch"],
            yaw=values["yaw"],
        )

    def _prepare_run_ui(self) -> None:
        self.log_output.Clear()
        self._last_result = None
        self.export_button.Enable(False)
        self.deploy_button.Enable(False)
        self.simulate_button.Enable(False)
        for control in self.parameter_controls.values():
            control.enable(False)
        self.object_list.Enable(False)
        self.robot_choice.Enable(False)

    def _finalise_run_ui(self, success: bool) -> None:
        self.simulate_button.Enable(True)
        for control in self.parameter_controls.values():
            control.enable(True)
        self.object_list.Enable(True)
        self.robot_choice.Enable(True)
        self.export_button.Enable(bool(success))
        # Enable deploy button only if we have successful poses AND robot has real_robot config
        can_deploy = bool(success) and self._can_deploy_to_robot()
        self.deploy_button.Enable(can_deploy)

    def _write_summary_log(self, result: ProtoSimRunResult) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"proto_sim_{timestamp}.log"
        lines = format_summary_lines(result)
        try:
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.logger.info("Wrote proto_sim summary log to %s", log_path)
        except Exception as exc:  # pragma: no cover - filesystem errors
            self.logger.exception("Failed to write summary log: %s", exc)

    def _can_deploy_to_robot(self) -> bool:
        """Check if the selected robot has real_robot configuration."""
        try:
            robot_name = self.robot_choice.GetStringSelection()
            profile = self.controller._profiles.get(robot_name)
            return profile is not None and profile.real_robot is not None
        except Exception:
            return False

    def _on_deploy_clicked(self, event):  # noqa: D401, ANN001 - wx signature
        """Handle deploy button click - execute plan on real robot."""
        if not self._last_result:
            wx.MessageBox("No plan available to deploy.", "Deploy", wx.OK | wx.ICON_INFORMATION)
            return

        robot_name = self.robot_choice.GetStringSelection()
        profile = self.controller._profiles.get(robot_name)
        if profile is None or profile.real_robot is None:
            wx.MessageBox(
                "No real robot configuration found for this robot.\n"
                "Add 'real_robot' section to the robot config in the YAML file.",
                "Deploy Error",
                wx.OK | wx.ICON_ERROR,
            )
            return

        # Build plan from result
        plan = self._last_result.build_plan_export()
        if not plan:
            wx.MessageBox("No successful poses to deploy.", "Deploy", wx.OK | wx.ICON_INFORMATION)
            return

        # Count total poses
        total_poses = 0
        for obj_name, obj_data in plan.items():
            for key in obj_data:
                if key.startswith("pose_") and key != "pose_in_robot_frame":
                    total_poses += 1

        # Confirm deployment
        real_robot_cfg = profile.real_robot
        confirm_msg = (
            f"Deploy plan to real robot?\n\n"
            f"Robot: {robot_name}\n"
            f"IP: {real_robot_cfg.ip}\n"
            f"Total poses: {total_poses}\n"
            f"Velocity: {real_robot_cfg.default_velocity:.2f} rad/s\n"
            f"Pause between poses: 1 second\n\n"
            f"The robot WILL MOVE. Ensure the workspace is clear."
        )
        result = wx.MessageBox(
            confirm_msg,
            "Confirm Deployment",
            wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING,
        )
        if result != wx.YES:
            return

        # Start deployment in background thread
        self._start_deployment(plan, profile)

    def _start_deployment(self, plan: Dict, profile) -> None:
        """Start the deployment in a background thread."""
        import threading

        self._prepare_deploy_ui()
        self.status_bar.SetStatusText("Deploying to robot…")
        self.log_output.AppendText("\n" + "=" * 60 + "\n")
        self.log_output.AppendText("DEPLOYING TO REAL ROBOT\n")
        self.log_output.AppendText("=" * 60 + "\n")

        def deploy_thread():
            try:
                self._execute_deployment(plan, profile)
            except Exception as exc:
                wx.CallAfter(self._handle_deploy_error, str(exc))

        thread = threading.Thread(target=deploy_thread, daemon=True)
        thread.start()

    def _execute_deployment(self, plan: Dict, profile) -> None:
        """Execute the deployment on the real robot (runs in background thread)."""
        import time

        try:
            from ...services.real_robot import URRobotDriver, RobotDriverConfig, Trajectory
        except ImportError as exc:
            wx.CallAfter(
                self._handle_deploy_error,
                f"Failed to import real_robot driver: {exc}\n"
                "Make sure ur_rtde is installed: pip install ur_rtde"
            )
            return

        real_robot_cfg = profile.real_robot
        
        # Create driver config with all parameters from YAML
        driver_config = RobotDriverConfig(
            robot_ip=real_robot_cfg.ip,
            robot_name=profile.name,
            max_joint_velocity=real_robot_cfg.default_velocity,
            max_joint_acceleration=real_robot_cfg.default_acceleration,
            velocity_scaling=1.0,  # Full speed since we set low default velocity
            blend_radius=real_robot_cfg.blend_radius,
            settling_time=real_robot_cfg.settling_time,
        )

        wx.CallAfter(self.log_output.AppendText, f"Connecting to robot at {real_robot_cfg.ip}...\n")

        try:
            driver = URRobotDriver(driver_config, self.logger.getChild("deploy"))
            if not driver.connect():
                wx.CallAfter(self._handle_deploy_error, f"Failed to connect to robot at {real_robot_cfg.ip}")
                return

            wx.CallAfter(self.log_output.AppendText, "✅ Connected to robot\n\n")

            # Get current position as home
            home_joints = driver.get_joint_positions(degrees=False)
            wx.CallAfter(self.log_output.AppendText, f"Home position recorded\n")

            pose_count = 0
            total_poses = sum(
                1 for obj_data in plan.values()
                for key in obj_data
                if key.startswith("pose_") and key != "pose_in_robot_frame"
            )

            # Execute each object's poses
            for obj_name, obj_data in plan.items():
                wx.CallAfter(self.log_output.AppendText, f"\n--- Object: {obj_name} ---\n")

                # Sort pose keys to execute in order
                pose_keys = sorted(
                    [k for k in obj_data if k.startswith("pose_") and k != "pose_in_robot_frame"],
                    key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0
                )

                # the pose_home key needs to be executed last if present
                if "pose_home" in pose_keys:
                    pose_keys.remove("pose_home")
                    pose_keys.append("pose_home")

                for pose_key in pose_keys:
                    pose_count += 1
                    pose_data = obj_data[pose_key]
                    waypoints = pose_data.get("waypoints", [])

                    if not waypoints:
                        wx.CallAfter(
                            self.log_output.AppendText,
                            f"  {pose_key}: No waypoints, skipping\n"
                        )
                        continue

                    wx.CallAfter(
                        self.log_output.AppendText,
                        f"  [{pose_count}/{total_poses}] {pose_key}: {len(waypoints)} waypoints... "
                    )

                    # Create trajectory
                    trajectory = Trajectory.from_waypoints(
                        waypoints,
                        duration=len(waypoints) * 0.1  # ~100ms per waypoint
                    )

                    # Execute trajectory using moveJ (accurate mode)
                    success = driver.execute_trajectory(
                        trajectory,
                        blocking=True,
                        use_servo=False,  # Use moveJ for accuracy
                    )

                    if success:
                        wx.CallAfter(self.log_output.AppendText, "✅\n")
                    else:
                        wx.CallAfter(self.log_output.AppendText, "❌ FAILED\n")

                    # Pause between poses
                    wx.CallAfter(
                        self.log_output.AppendText,
                        f"    Pausing 1 second at pose...\n"
                    )
                    time.sleep(1.0)

            # Return to home position
            wx.CallAfter(self.log_output.AppendText, f"\nReturning to home position...\n")
            driver.move_to_joints(home_joints, velocity=real_robot_cfg.default_velocity, blocking=True)
            wx.CallAfter(self.log_output.AppendText, "✅ Returned to home\n")

            driver.disconnect()
            wx.CallAfter(self._handle_deploy_complete, pose_count)

        except Exception as exc:
            self.logger.exception("Deployment failed: %s", exc)
            wx.CallAfter(self._handle_deploy_error, str(exc))
            try:
                driver.disconnect()
            except Exception:
                pass

    def _prepare_deploy_ui(self) -> None:
        """Disable UI controls during deployment."""
        self.simulate_button.Enable(False)
        self.export_button.Enable(False)
        self.deploy_button.Enable(False)
        for control in self.parameter_controls.values():
            control.enable(False)
        self.object_list.Enable(False)
        self.robot_choice.Enable(False)

    def _handle_deploy_error(self, error_msg: str) -> None:
        """Handle deployment error."""
        self.log_output.AppendText(f"\n❌ DEPLOYMENT ERROR: {error_msg}\n")
        self.status_bar.SetStatusText("Deployment failed")
        wx.MessageBox(f"Deployment failed:\n{error_msg}", "Deploy Error", wx.OK | wx.ICON_ERROR)
        self._finalise_run_ui(bool(self._last_result))

    def _handle_deploy_complete(self, pose_count: int) -> None:
        """Handle successful deployment completion."""
        self.log_output.AppendText(f"\n{'=' * 60}\n")
        self.log_output.AppendText(f"✅ DEPLOYMENT COMPLETE: {pose_count} poses executed\n")
        self.log_output.AppendText(f"{'=' * 60}\n")
        self.status_bar.SetStatusText(f"Deployment complete: {pose_count} poses")
        wx.MessageBox(
            f"Deployment complete!\n\n{pose_count} poses executed successfully.",
            "Deploy Complete",
            wx.OK | wx.ICON_INFORMATION,
        )
        self._finalise_run_ui(bool(self._last_result))


def run_proto_sim_gui(
    config_path: str,
    *,
    backend: Backend = Backend.GPU,
    log_level: str = "INFO",
    debug: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Launch the proto_sim GUI."""
    if not HAS_WX:
        raise RuntimeError("wxPython is required for the GUI. Install with 'pip install wxpython'.")

    level_value = getattr(logging, log_level.upper(), logging.INFO)
    if logger is None:
        logger = setup_logging(level_value, debug=debug).getChild("proto_gui")
    else:
        logger.setLevel(level_value)

    controller = SessionController(config_path, backend, logger)
    try:
        app = wx.App(False)
        ProtoSimFrame(controller, logger=logger, debug=debug)
        app.MainLoop()
    finally:
        controller.shutdown()


__all__ = ["run_proto_sim_gui", "ProtoSimFrame", "ParameterInputControl"]
