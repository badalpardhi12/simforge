"""wxPython-based GUI for robot control.

Cross-platform interface with joint sliders and Cartesian controls.
Compatible with macOS Apple Silicon.
"""
from __future__ import annotations

import wx
from typing import Dict, List

from .config_reader import SimforgeConfig
from .movement_controller import MovementController, ControlMode
from .logging_utils import setup_logging


JOINT_COUNT = 6


class RobotControlFrame(wx.Frame):
    """Main GUI frame for robot control."""
    
    def __init__(self, config: SimforgeConfig, debug: bool = False):
        super().__init__(
            parent=None, 
            title="Simforge Robot Control", 
            size=(600, 650),
            style=wx.DEFAULT_FRAME_STYLE & ~(wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        )
        self.config = config
        self.logger = setup_logging(debug)
        self.joint_sliders: Dict[str, List[wx.Slider]] = {}
        self.cart_fields: Dict[str, List[wx.TextCtrl]] = {}
        self._building = True
        self._running = True
        
        # Show loading dialog
        loading_dlg = wx.ProgressDialog(
            "Loading", 
            "Initializing robots...", 
            maximum=100, 
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE
        )
        loading_dlg.Pulse("Loading robots...")
        wx.Yield()
        
        try:
            # Initialize controller
            self.controller = MovementController(config, debug=debug)
            loading_dlg.Pulse("Building scene...")
            wx.Yield()
            
            self.controller.build_scene()
            loading_dlg.Pulse("Starting simulation...")
            wx.Yield()
            
            self.controller.start()
            loading_dlg.Update(100, "Complete!")
            
        except Exception as e:
            loading_dlg.Destroy()
            wx.MessageBox(f"Failed to initialize: {e}", "Error", wx.OK | wx.ICON_ERROR)
            self.Destroy()
            return
        finally:
            loading_dlg.Destroy()
        
        # Build GUI
        self._setup_gui()
        
        # Show frame
        self.Show()
        self.Raise()
        
        # Start update timer
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_timer, self.timer)
        self.timer.Start(300)  # 300ms updates
        
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self._building = False
        self.logger.info("wxPython GUI initialized successfully")
    
    def _setup_gui(self):
        """Create the GUI interface."""
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        
        # Status bar
        self.status_bar = self.CreateStatusBar()
        self.status_bar.SetStatusText("Ready")
        
        # Robot tabs
        self.notebook = wx.Notebook(panel)
        vbox.Add(self.notebook, 1, wx.EXPAND | wx.ALL, 5)
        
        for robot_config in self.config.robots:
            self._create_robot_tab(robot_config)
        
        panel.SetSizer(vbox)
    
    def _create_robot_tab(self, robot_config):
        """Create a tab for one robot."""
        robot_name = robot_config.name
        tab = wx.Panel(self.notebook)
        tab_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Joint control section
        joint_box = wx.StaticBoxSizer(wx.VERTICAL, tab, "Joint Control (degrees)")
        sliders = []
        
        for joint_idx in range(JOINT_COUNT):
            row_sizer = wx.BoxSizer(wx.HORIZONTAL)
            
            # Joint label
            label = wx.StaticText(tab, label=f"J{joint_idx + 1}:")
            label.SetMinSize((30, -1))
            row_sizer.Add(label, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
            
            # Joint slider
            slider = wx.Slider(
                tab, value=0, minValue=-180, maxValue=180,
                style=wx.SL_HORIZONTAL | wx.SL_VALUE_LABEL,
                size=(300, -1)
            )
            slider.Bind(
                wx.EVT_SLIDER, 
                lambda evt, rn=robot_name, idx=joint_idx: 
                    self._on_joint_slider(rn, idx, evt)
            )
            row_sizer.Add(slider, 1, wx.EXPAND | wx.RIGHT, 10)
            
            # Value display
            value_text = wx.StaticText(tab, label="0°")
            value_text.SetMinSize((40, -1))
            row_sizer.Add(value_text, 0, wx.ALIGN_CENTER_VERTICAL)
            
            joint_box.Add(row_sizer, 0, wx.EXPAND | wx.ALL, 3)
            sliders.append(slider)
        
        self.joint_sliders[robot_name] = sliders
        tab_sizer.Add(joint_box, 0, wx.EXPAND | wx.ALL, 5)
        
        # Cartesian control section
        cart_box = wx.StaticBoxSizer(wx.VERTICAL, tab, "Cartesian Control")
        
        # Create a grid with 3 rows and 4 columns: [Label1] [Field1] [Label2] [Field2]
        cart_grid = wx.FlexGridSizer(rows=3, cols=4, hgap=15, vgap=12)
        cart_grid.AddGrowableCol(1, 1)  # Make position fields expandable
        cart_grid.AddGrowableCol(3, 1)  # Make orientation fields expandable
        
        # Labels and field data
        labels_and_fields = [
            ("X (m):", "Roll (°):"),
            ("Y (m):", "Pitch (°):"),
            ("Z (m):", "Yaw (°):")
        ]
        
        pos_fields = []
        ori_fields = []
        
        for pos_label, ori_label in labels_and_fields:
            # Position label and field
            pos_lbl = wx.StaticText(tab, label=pos_label)
            pos_lbl.SetMinSize((50, -1))
            pos_field = wx.TextCtrl(tab, value="0.0", size=(100, -1))
            pos_field.SetFont(wx.Font(9, wx.FONTFAMILY_MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            
            # Orientation label and field
            ori_lbl = wx.StaticText(tab, label=ori_label)
            ori_lbl.SetMinSize((60, -1))
            ori_field = wx.TextCtrl(tab, value="0.0", size=(100, -1))
            ori_field.SetFont(wx.Font(9, wx.FONTFAMILY_MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            
            # Add to grid: pos_label, pos_field, ori_label, ori_field
            cart_grid.Add(pos_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
            cart_grid.Add(pos_field, 1, wx.EXPAND)
            cart_grid.Add(ori_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
            cart_grid.Add(ori_field, 1, wx.EXPAND)
            
            pos_fields.append(pos_field)
            ori_fields.append(ori_field)
        
        cart_box.Add(cart_grid, 0, wx.EXPAND | wx.ALL, 15)
        
        # Action button with better styling
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        move_btn = wx.Button(tab, label="🎯 Move to Pose", size=(150, 35))
        move_btn.SetFont(wx.Font(
            10, 
            wx.FONTFAMILY_DEFAULT, 
            wx.FONTSTYLE_NORMAL, 
            wx.FONTWEIGHT_BOLD
        ))
        move_btn.SetBackgroundColour(wx.Colour(70, 130, 180))  # Steel blue
        move_btn.SetForegroundColour(wx.Colour(255, 255, 255))  # White text
        move_btn.Bind(
            wx.EVT_BUTTON, 
            lambda evt, rn=robot_name: self._on_move_cartesian(rn)
        )
        
        button_sizer.AddStretchSpacer()
        button_sizer.Add(move_btn, 0, wx.ALIGN_CENTER)
        button_sizer.AddStretchSpacer()
        
        cart_box.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Store all fields for this robot
        self.cart_fields[robot_name] = pos_fields + ori_fields
        tab_sizer.Add(cart_box, 0, wx.EXPAND | wx.ALL, 5)
        
        tab.SetSizer(tab_sizer)
        self.notebook.AddPage(tab, robot_name)
    
    def _on_joint_slider(self, robot_name: str, joint_idx: int, event):
        """Handle joint slider changes."""
        if self._building:
            return
        
        value_deg = event.GetInt()
        try:
            self.controller.set_joint_position(robot_name, joint_idx, float(value_deg))
            self.status_bar.SetStatusText(f"{robot_name} J{joint_idx + 1}: {value_deg}°")
        except Exception as e:
            self.logger.debug(f"Joint update failed: {e}")
    
    def _on_move_cartesian(self, robot_name: str):
        """Handle Cartesian move command."""
        fields = self.cart_fields[robot_name]
        try:
            values = [float(field.GetValue()) for field in fields]
            position = tuple(values[:3])  # x, y, z
            orientation = tuple(values[3:])  # roll, pitch, yaw
            
            self.controller.move_cartesian(robot_name, position, orientation)
            self.status_bar.SetStatusText(
                f"{robot_name} moving to {position} {orientation}"
            )
            self.logger.info(
                f"Cartesian move: {robot_name} -> pos={position}, rpy={orientation}"
            )
            
        except ValueError:
            wx.MessageBox(
                "Invalid cartesian values", 
                "Input Error", 
                wx.OK | wx.ICON_WARNING
            )
        except Exception as e:
            wx.MessageBox(
                f"Move failed: {e}", 
                "Error", 
                wx.OK | wx.ICON_ERROR
            )
            self.logger.error(f"Cartesian move failed: {e}")
    
    def _on_timer(self, event):
        """Update GUI from robot state."""
        if not self._running:
            return
        
        try:
            # Update joint sliders from current robot state
            for robot_name, sliders in self.joint_sliders.items():
                positions = self.controller.get_joint_positions(robot_name)
                if not positions:
                    continue
                
                for i, slider in enumerate(sliders):
                    if i < len(positions):
                        current_val = slider.GetValue()
                        new_val = int(round(positions[i]))
                        # Only update if significantly different to avoid feedback loops
                        if abs(current_val - new_val) > 2:
                            slider.SetValue(new_val)
        except Exception as e:
            self.logger.debug(f"Timer update failed: {e}")
    
    def _on_close(self, event):
        """Handle window close."""
        self._running = False
        
        try:
            if hasattr(self, 'timer') and self.timer.IsRunning():
                self.timer.Stop()
            
            if hasattr(self, 'controller'):
                self.controller.stop()
                self.logger.info("Movement controller stopped")
        except Exception as e:
            self.logger.warning(f"Cleanup warning: {e}")
        finally:
            self.Destroy()


def run_gui(config: SimforgeConfig, debug: bool = False):
    """Run the wxPython GUI application."""
    print("🚀 Starting wxPython GUI...")
    
    app = wx.App(False)  # Don't redirect stdout/stderr
    frame = RobotControlFrame(config, debug=debug)
    
    if frame:  # Frame might be None if initialization failed
        app.MainLoop()
    
    print("✅ GUI application exited")


def run_headless(config: SimforgeConfig, debug: bool = False):
    """Run the control system without GUI for testing purposes."""
    return ControlGUI(config, debug=debug)


# Backward compatibility alias for tests
class ControlGUI:
    """Backward compatibility wrapper for tests."""
    
    def __init__(self, config: SimforgeConfig, debug: bool = False):
        self.controller = MovementController(config, debug=debug)
        self.controller.build_scene()
        self.controller.start()
        self.config = config
        
        # Add robot_states for test compatibility
        self.robot_states = {
            robot.name: {
                "mode": ControlMode.JOINT,
                "joint_targets": robot.initial_joint_positions or [0.0] * 6,
                "cartesian_target": (0.4, 0.0, 0.3, 0.0, 180.0, 0.0),
            }
            for robot in config.robots
        }
    
    def start(self):
        pass  # Already started in __init__
    
    def stop(self):
        self.controller.stop()
    
    def set_robot_mode(self, robot_name: str, mode):
        self.controller.switch_mode(robot_name, mode)
    
    def move_joint(self, robot_name: str, joint_idx: int, value_deg: float):
        self.controller.set_joint_position(robot_name, joint_idx, value_deg)
    
    def move_cartesian(self, robot_name: str, x: float, y: float, z: float,
                      roll_deg: float, pitch_deg: float, yaw_deg: float):
        position = (x, y, z)
        orientation = (roll_deg, pitch_deg, yaw_deg)
        self.controller.move_cartesian(robot_name, position, orientation)
    
    def get_available_robots(self) -> list:
        return [robot.name for robot in self.config.robots]


__all__ = ["run_gui", "run_headless", "RobotControlFrame", "ControlGUI"]
