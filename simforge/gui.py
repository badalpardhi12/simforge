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

        self.root.title("Simforge")
        self.root.geometry("720x540")

        self._build_widgets()
        # Provide a UI-thread executor to simulator for main-thread-only Genesis calls
        self.controller.sim.set_ui_executor(self._run_on_ui_thread_sync)
        self._start_background()

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
        self.mode_var = tk.StringVar(value=ControlMode.JOINT)
        mode_combo = ttk.Combobox(top, textvariable=self.mode_var, values=[ControlMode.JOINT, ControlMode.CARTESIAN], state="readonly")
        mode_combo.pack(side=tk.LEFT, padx=6)
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        ttk.Label(top, text="Robot:").pack(side=tk.LEFT, padx=(12, 0))
        self.robot_var = tk.StringVar(value=(self.config.robots[0].name if self.config.robots else ""))
        robot_names = [r.name for r in self.config.robots]
        robot_combo = ttk.Combobox(top, textvariable=self.robot_var, values=robot_names, state="readonly")
        robot_combo.pack(side=tk.LEFT, padx=6)
        robot_combo.bind("<<ComboboxSelected>>", lambda e: self._rebuild_joint_sliders())

        self.content = ttk.Frame(self.root)
        self.content.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

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

        for idx in range(len(rob.initial_joint_positions or [0]*6)):
            lower = math.degrees(joints[idx].lower) if idx < len(joints) and joints[idx].lower is not None else -180
            upper = math.degrees(joints[idx].upper) if idx < len(joints) and joints[idx].upper is not None else 180
            val = (rob.initial_joint_positions[idx] if rob.initial_joint_positions else 0.0)

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
        self.cart_vars = {
            'x': tk.DoubleVar(value=0.4),
            'y': tk.DoubleVar(value=0.0),
            'z': tk.DoubleVar(value=0.3),
            'roll': tk.DoubleVar(value=0.0),
            'pitch': tk.DoubleVar(value=180.0),
            'yaw': tk.DoubleVar(value=0.0),
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

    def _on_mode_change(self, event=None):
        mode = self.mode_var.get()
        self.controller.switch_mode(mode)
        self.status.configure(text=f"Switched to {mode} mode; resetting to home")
        # Rebuild control panel to reflect mode (joint vs cartesian)
        self._build_controls_for_mode()

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

    def _start_background(self):
        self.controller.start()
        self._ui_updater_running = True
        self.root.after(100, self._ui_update_loop)

    def _ui_update_loop(self):
        if not self._ui_updater_running:
            return
        # Update status with simple heartbeat
        self.status.configure(text=f"Mode: {self.controller.mode} | Robot: {self.robot_var.get()}")
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
    root = tk.Tk()
    app = SimforgeApp(root, config=config, debug=debug)
    root.mainloop()
