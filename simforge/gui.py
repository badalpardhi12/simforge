from __future__ import annotations

import math
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Dict, List

from .config import SimforgeConfig
from .controller import Controller, ControlMode
from .urdf_utils import parse_joint_limits


class SimforgeApp:
    def __init__(self, root: tk.Tk, config: SimforgeConfig, debug: bool = False) -> None:
        self.root = root
        self.config = config
        self.controller = Controller(config, debug=debug)
        self._is_shutting_down = False
        # Hide the window until scene is built
        self.root.withdraw()

        # Add logger to GUI class
        from .logging_utils import setup_logging
        self.logger = setup_logging(debug)

        self.root.title("Simforge")
        self.root.geometry("720x540")

        # Set up window close handler for graceful shutdown
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # Handle SIGINT for GUI mode too
        import signal
        def signal_handler(signum, frame):
            if not self._is_shutting_down:
                self._on_window_close()
        signal.signal(signal.SIGINT, signal_handler)

        # 1) Build the UI widgets first
        self._build_widgets()

        # 2) Set up callbacks and UI-thread executor
        self.controller.sim.set_ui_executor(self._run_on_ui_thread_sync)

        def shutdown_callback():
            try:
                self.logger.info("Shutdown callback triggered from controller")
            except AttributeError:
                pass
            self.root.quit()
        self.controller._shutdown_callback = shutdown_callback

        def scene_callback():
            try:
                self.logger.info("Scene built, showing GUI window")
                self.root.deiconify()
                self.root.lift()
                self.root.focus_force()
            except Exception as e:
                try:
                    self.logger.warning(f"Failed to show GUI window: {e}")
                except AttributeError:
                    pass
        self.controller._scene_callback = scene_callback

        # 3) Now, build the scene on the main thread. The callback will show the window.
        self.controller.sim.build_scene()

        # 4) Start the simulation thread and UI update loop
        self.controller.start()
        self._ui_updater_running = True
        self.root.after(100, self._ui_update_loop)

    def _run_on_ui_thread_sync(self, fn):
        """Run a callable on the Tk main thread and wait for result."""
        result = {}
        done = threading.Event()

        def wrapper():
            try:
                result['value'] = fn()
            except Exception as e:
                result['error'] = e
            finally:
                done.set()

        # Schedule on Tk main loop
        self.root.after(0, wrapper)
        # Wait for completion
        done.wait()
        if 'error' in result:
            raise result['error']
        return result.get('value')

    def _build_widgets(self):
        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(top, text="Mode:").pack(side=tk.LEFT)
        # Initialize mode based on the selected robot's effective mode
        initial_robot = (self.config.robots[0].name if self.config.robots else "")
        self.mode_var = tk.StringVar(value=self.controller.get_mode(initial_robot))
        mode_combo = ttk.Combobox(top, textvariable=self.mode_var, values=[ControlMode.JOINT, ControlMode.CARTESIAN], state="readonly")
        mode_combo.pack(side=tk.LEFT, padx=6)
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        ttk.Label(top, text="Robot:").pack(side=tk.LEFT, padx=(12, 0))
        self.robot_var = tk.StringVar(value=(self.config.robots[0].name if self.config.robots else ""))
        robot_names = [r.name for r in self.config.robots]
        robot_combo = ttk.Combobox(top, textvariable=self.robot_var, values=robot_names, state="readonly")
        robot_combo.pack(side=tk.LEFT, padx=6)
        robot_combo.bind("<<ComboboxSelected>>", self._on_robot_change)

        # Effective control info (per-robot)
        self.ctrl_info = ttk.Label(self.root, text="", anchor=tk.W, justify=tk.LEFT)
        self.ctrl_info.pack(fill=tk.X, padx=8, pady=(4, 0))

        self.content = ttk.Frame(self.root)
        self.content.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        # Initialize control info display
        self._refresh_ctrl_info()

        self.status = ttk.Label(self.root, text="Ready", anchor=tk.W)
        self.status.pack(fill=tk.X, padx=8, pady=(0, 8))

        self._build_controls_for_mode()

    def _build_controls_for_mode(self):
        for w in self.content.winfo_children():
            w.destroy()
        if self.mode_var.get() == ControlMode.JOINT:
            self._build_joint_controls()
        else:
            self._build_cartesian_controls()

    def _build_joint_controls(self):
        self.joint_sliders: List[tk.Scale] = []

        rob = next((r for r in self.config.robots if r.name == self.robot_var.get()), None)
        if not rob:
            ttk.Label(self.content, text="No robot configured").pack()
            return

        joints = parse_joint_limits(rob.urdf)
        if not joints:
            ttk.Label(self.content, text="No joint limits found in URDF; using -180..180°").pack()

        # Use controller's current targets for this robot to preserve state
        current_targets = self.controller.get_joint_targets_deg(rob.name) or (rob.initial_joint_positions or [])
        dof_count = max(len(current_targets), len(joints) if joints else 6)
        for idx in range(dof_count):
            lower = math.degrees(joints[idx].lower) if idx < len(joints) and joints[idx].lower is not None else -180
            upper = math.degrees(joints[idx].upper) if idx < len(joints) and joints[idx].upper is not None else 180
            val = float(current_targets[idx]) if idx < len(current_targets) else 0.0

            row = ttk.Frame(self.content)
            row.pack(fill=tk.X, pady=4)
            ttk.Label(row, text=f"J{idx+1}").pack(side=tk.LEFT, padx=(0,6))

            s = tk.Scale(row, from_=lower, to=upper, orient=tk.HORIZONTAL, resolution=0.5, length=420)
            s.set(val)
            s.pack(side=tk.LEFT, fill=tk.X, expand=True)
            s.configure(command=lambda v, i=idx: self._on_joint_slider(i, float(v)))
            self.joint_sliders.append(s)

            val_lbl = ttk.Label(row, text=f"{val:.1f}°")
            val_lbl.pack(side=tk.LEFT, padx=6)
            s.bind("<B1-Motion>", lambda e, sl=s, lbl=val_lbl: lbl.configure(text=f"{sl.get():.1f}°"))
            s.bind("<ButtonRelease-1>", lambda e, sl=s, lbl=val_lbl: lbl.configure(text=f"{sl.get():.1f}°"))

    def _build_cartesian_controls(self):
        # XYZ in meters, RPY in degrees
        robot = self.robot_var.get()
        # Prefer last commanded Cartesian target for this robot
        last = self.controller.get_last_cartesian_target(robot)
        if last and len(last) == 6:
            init_pose = dict(x=last[0], y=last[1], z=last[2], roll=last[3], pitch=last[4], yaw=last[5])
        else:
            # Fallback to current EE pose, else defaults
            ee = self.controller.get_ee_pose_xyzrpy(robot)
            if ee and len(ee) == 6:
                init_pose = dict(x=ee[0], y=ee[1], z=ee[2], roll=ee[3], pitch=ee[4], yaw=ee[5])
            else:
                init_pose = dict(x=0.4, y=0.0, z=0.3, roll=0.0, pitch=180.0, yaw=0.0)

        self.cart_vars = {
            'x': tk.DoubleVar(value=float(init_pose['x'])),
            'y': tk.DoubleVar(value=float(init_pose['y'])),
            'z': tk.DoubleVar(value=float(init_pose['z'])),
            'roll': tk.DoubleVar(value=float(init_pose['roll'])),
            'pitch': tk.DoubleVar(value=float(init_pose['pitch'])),
            'yaw': tk.DoubleVar(value=float(init_pose['yaw'])),
        }
        ranges = {
            'x': (-1.0, 1.0),
            'y': (-1.0, 1.0),
            'z': (0.0, 1.5),
            'roll': (-180.0, 180.0),
            'pitch': (-180.0, 180.0),
            'yaw': (-180.0, 180.0),
        }
        order = ['x','y','z','roll','pitch','yaw']
        for key in order:
            row = ttk.Frame(self.content)
            row.pack(fill=tk.X, pady=4)
            ttk.Label(row, text=key.upper()).pack(side=tk.LEFT, padx=(0,6))
            lo, hi = ranges[key]
            s = tk.Scale(row, from_=lo, to=hi, orient=tk.HORIZONTAL, resolution=0.005 if key in ('x','y','z') else 0.5, length=420, variable=self.cart_vars[key])
            s.pack(side=tk.LEFT, fill=tk.X, expand=True)
            s.configure(command=lambda v, k=key: self._on_cart_slider(k, float(v)))
            val_lbl = ttk.Label(row, text=f"{self.cart_vars[key].get():.3f}" if key in ('x','y','z') else f"{self.cart_vars[key].get():.1f}°")
            val_lbl.pack(side=tk.LEFT, padx=6)
            s.bind("<B1-Motion>", lambda e, sv=self.cart_vars[key], k=key, lbl=val_lbl: lbl.configure(text=(f"{sv.get():.3f}" if k in ('x','y','z') else f"{sv.get():.1f}°")))
            s.bind("<ButtonRelease-1>", lambda e, sv=self.cart_vars[key], k=key, lbl=val_lbl: lbl.configure(text=(f"{sv.get():.3f}" if k in ('x','y','z') else f"{sv.get():.1f}°")))

        # Action buttons
        btns = ttk.Frame(self.content)
        btns.pack(fill=tk.X, pady=(12,0))
        plan_btn = ttk.Button(btns, text="Plan & Move", command=self._on_cart_plan_move)
        plan_btn.pack(side=tk.LEFT)
        stop_btn = ttk.Button(btns, text="Stop", command=self._on_cart_stop)
        stop_btn.pack(side=tk.LEFT, padx=8)

    def _rebuild_joint_sliders(self):
        self._build_controls_for_mode()

    def _on_robot_change(self, event=None):
        # Update mode selector to reflect this robot's preserved mode
        robot = self.robot_var.get()
        self.mode_var.set(self.controller.get_mode(robot))
        # Rebuild UI for the new robot and refresh effective control display
        self._build_controls_for_mode()
        self._refresh_ctrl_info()

    def _refresh_ctrl_info(self):
        try:
            robot = self.robot_var.get()
            ctrl = self.config.control_for(robot)
            info = (
                f"Effective control for {robot} — "
                f"planner={getattr(ctrl, 'planner', '')} | "
                f"timeout={getattr(ctrl, 'planner_timeout', '')} | "
                f"resolution={getattr(ctrl, 'planner_resolution', '')} | "
                f"waypoints={getattr(ctrl, 'cartesian_waypoints', '')} | "
                f"strict_cartesian={getattr(ctrl, 'strict_cartesian', '')}"
            )
        except Exception as e:
            info = f"Effective control: unavailable ({e})"
        if hasattr(self, 'ctrl_info') and self.ctrl_info:
            self.ctrl_info.configure(text=info)


    def _on_mode_change(self, event=None):
        robot = self.robot_var.get()
        mode = self.mode_var.get()
        self.controller.switch_mode(robot, mode)
        self.status.configure(text=f"Switched {robot} to {mode} mode; preserving other robots' state")
        # Rebuild control panel to reflect mode (joint vs cartesian) for this robot
        self._build_controls_for_mode()
        # Refresh effective control display
        self._refresh_ctrl_info()

    def _on_joint_slider(self, idx: int, value_deg: float):
        robot = self.robot_var.get()
        self.controller.set_joint_deg(robot, idx, value_deg)

    def _on_cart_slider(self, key: str, _value: float):
        # Do not command on every slider change; use Plan & Move button
        pass

    def _on_cart_plan_move(self):
        robot = self.robot_var.get()
        pose = (
            self.cart_vars['x'].get(),
            self.cart_vars['y'].get(),
            self.cart_vars['z'].get(),
            self.cart_vars['roll'].get(),
            self.cart_vars['pitch'].get(),
            self.cart_vars['yaw'].get(),
        )
        self.controller.set_cartesian_target(robot, pose, frame='base')
        self.status.configure(text=f"Planning move to {pose}")

    def _on_cart_stop(self):
        robot = self.robot_var.get()
        self.controller.cancel_cartesian(robot)
        self.status.configure(text="Stopped active motion")

    def _on_window_close(self):
        """Handle window close event for graceful shutdown."""
        if self._is_shutting_down:
            return
        self._is_shutting_down = True

        # Stop UI update loop FIRST
        self._ui_updater_running = False

        # Stop the controller (simulation) gracefully
        try:
            print("Stopping simulation controller...")
            self.controller.stop()
        except Exception as e:
            print(f"Error stopping controller: {e}")

        # Destroy the window safely
        try:
            print("Closing GUI windows...")
            self.root.destroy()
        except Exception as e:
            print(f"Error closing window: {e}")

        # Use a cleaner exit
        import sys
        sys.exit(0)

    def _ui_update_loop(self):
        if not self._ui_updater_running:
            return
        # If shutting down, don't continue the loop
        if self._is_shutting_down:
            return

        # Check if simulator requested shutdown (e.g., viewer window closed)
        if self.controller.sim._shutdown_evt.is_set():
            try:
                self.logger.info("Received shutdown request from simulator, closing GUI")
            except AttributeError:
                pass  # Logger may not be available during shutdown
            self.root.quit()
            return

        # Update status with simple heartbeat
        try:
            current_robot = self.robot_var.get()
            self.status.configure(text=f"Mode: {self.controller.get_mode(current_robot)} | Robot: {current_robot}")
        except Exception:
            # Skip status updates during shutdown
            pass
        # If last Cartesian failed, reset sliders to actual EE pose
        if self.mode_var.get() == ControlMode.CARTESIAN:
            robot = self.robot_var.get()
            st = self.controller.get_last_cartesian_status(robot)
            if st and st.startswith("cartesian_failed"):
                pose = self.controller.get_ee_pose_xyzrpy(robot)
                if pose and hasattr(self, 'cart_vars'):
                    x,y,z, r,p,yw = pose
                    self.cart_vars['x'].set(x)
                    self.cart_vars['y'].set(y)
                    self.cart_vars['z'].set(z)
                    self.cart_vars['roll'].set(r)
                    self.cart_vars['pitch'].set(p)
                    self.cart_vars['yaw'].set(yw)
                # clear last event
                self.controller.clear_last_cartesian_status(robot)
        self.root.after(200, self._ui_update_loop)


def run_app(config: SimforgeConfig, debug: bool = False):
    root = None
    app = None
    try:
        root = tk.Tk()
        app = SimforgeApp(root, config=config, debug=debug)
        print("Starting GUI main loop...")
        root.mainloop()
        print("GUI main loop exited cleanly")
    except KeyboardInterrupt:
        print("Received keyboard interrupt in GUI mode")
    except Exception as e:
        print(f"Error in GUI mode: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("GUI application shutdown complete")
