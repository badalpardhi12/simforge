#!/usr/bin/env python3
"""
UR Robot Communication Module

Provides robust communication with Universal Robots using multiple interfaces:
1. Dashboard Server (port 29999) - Power on/off, status queries, program control
2. Secondary Interface (port 30002) - Send URScript commands at 10Hz
3. Realtime Interface (port 30003) - Read robot state at 500Hz

This avoids the RTDE "input registers in use" conflict that occurs when
EtherNet/IP adapter is enabled on the robot.

Author: Simforge Team
"""

import socket
import struct
import time
import threading
import math
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


class RobotMode(IntEnum):
    """UR Robot Mode"""
    DISCONNECTED = -1
    NO_CONTROLLER = 0
    RUNNING = 1
    IDLE = 2
    POWER_OFF = 3
    EMERGENCY_STOP = 4
    PROTECTIVE_STOP = 5
    BACKDRIVE = 6
    BOOTING = 7


class SafetyMode(IntEnum):
    """UR Safety Mode"""
    NORMAL = 1
    REDUCED = 2
    PROTECTIVE_STOP = 3
    RECOVERY = 4
    SAFEGUARD_STOP = 5
    SYSTEM_EMERGENCY_STOP = 6
    ROBOT_EMERGENCY_STOP = 7
    VIOLATION = 8
    FAULT = 9


@dataclass
class RobotState:
    """Current state of the UR robot."""
    # Connection status
    connected: bool = False
    
    # Joint state (6 joints)
    joint_positions: List[float] = field(default_factory=lambda: [0.0] * 6)
    joint_velocities: List[float] = field(default_factory=lambda: [0.0] * 6)
    joint_currents: List[float] = field(default_factory=lambda: [0.0] * 6)
    joint_temperatures: List[float] = field(default_factory=lambda: [0.0] * 6)
    
    # TCP (Tool Center Point) state
    tcp_pose: List[float] = field(default_factory=lambda: [0.0] * 6)  # x,y,z,rx,ry,rz
    tcp_speed: List[float] = field(default_factory=lambda: [0.0] * 6)
    tcp_force: List[float] = field(default_factory=lambda: [0.0] * 6)
    
    # Robot status
    robot_mode: RobotMode = RobotMode.DISCONNECTED
    safety_mode: SafetyMode = SafetyMode.NORMAL
    program_running: bool = False
    is_power_on: bool = False
    is_brakes_released: bool = False
    
    # Timestamp
    timestamp: float = 0.0


class DashboardClient:
    """
    Client for UR Dashboard Server (port 29999).
    
    Used for:
    - Power on/off robot
    - Release brakes
    - Load/play/stop programs
    - Query robot status
    - Unlock protective stops
    """
    
    PORT = 29999
    TIMEOUT = 5.0
    
    def __init__(self, robot_ip: str):
        self.robot_ip = robot_ip
        self._socket: Optional[socket.socket] = None
        self._lock = threading.Lock()
    
    def connect(self) -> bool:
        """Connect to dashboard server."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.TIMEOUT)
            self._socket.connect((self.robot_ip, self.PORT))
            # Read welcome message
            welcome = self._socket.recv(1024).decode('utf-8')
            logger.info(f"Dashboard connected: {welcome.strip()}")
            return True
        except Exception as e:
            logger.error(f"Dashboard connection failed: {e}")
            self._socket = None
            return False
    
    def disconnect(self):
        """Disconnect from dashboard server."""
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
            self._socket = None
    
    def _send_command(self, command: str) -> str:
        """Send command and receive response."""
        if not self._socket:
            if not self.connect():
                return "Error: Not connected"
        
        with self._lock:
            try:
                # Send command
                cmd = f"{command}\n"
                self._socket.sendall(cmd.encode('utf-8'))
                
                # Receive response
                response = self._socket.recv(4096).decode('utf-8').strip()
                return response
            except socket.timeout:
                logger.error(f"Dashboard command timeout: {command}")
                return "Error: Timeout"
            except Exception as e:
                logger.error(f"Dashboard command error: {e}")
                self.disconnect()
                return f"Error: {e}"
    
    # Power commands
    def power_on(self) -> bool:
        """Power on the robot."""
        response = self._send_command("power on")
        logger.info(f"Power on: {response}")
        return "Powering on" in response
    
    def power_off(self) -> bool:
        """Power off the robot."""
        response = self._send_command("power off")
        logger.info(f"Power off: {response}")
        return "Powering off" in response
    
    def release_brakes(self) -> bool:
        """Release the brakes."""
        response = self._send_command("brake release")
        logger.info(f"Brake release: {response}")
        return "Brake" in response
    
    # Program commands
    def load_program(self, program_name: str) -> bool:
        """Load a program file."""
        response = self._send_command(f"load {program_name}")
        return "Loading program" in response
    
    def play(self) -> bool:
        """Start program execution."""
        response = self._send_command("play")
        return "Starting program" in response
    
    def stop(self) -> bool:
        """Stop program execution."""
        response = self._send_command("stop")
        return "Stopped" in response
    
    def pause(self) -> bool:
        """Pause program execution."""
        response = self._send_command("pause")
        return "Pausing program" in response
    
    # Status queries
    def get_robot_mode(self) -> str:
        """Get current robot mode."""
        return self._send_command("robotmode")
    
    def get_safety_mode(self) -> str:
        """Get current safety mode."""
        return self._send_command("safetymode")
    
    def is_program_running(self) -> bool:
        """Check if a program is running."""
        response = self._send_command("running")
        return "true" in response.lower()
    
    def is_in_remote_control(self) -> bool:
        """Check if robot is in remote control mode."""
        response = self._send_command("is in remote control")
        return "true" in response.lower()
    
    # Safety commands
    def unlock_protective_stop(self) -> bool:
        """Unlock after a protective stop."""
        response = self._send_command("unlock protective stop")
        logger.info(f"Unlock protective stop: {response}")
        return "Protective stop releasing" in response
    
    def close_safety_popup(self) -> bool:
        """Close safety popup on teach pendant."""
        response = self._send_command("close safety popup")
        return "closing" in response.lower()
    
    def restart_safety(self) -> bool:
        """Restart safety system."""
        response = self._send_command("restart safety")
        return "Restarting safety" in response
    
    # UI commands
    def popup(self, message: str) -> bool:
        """Show popup on teach pendant."""
        response = self._send_command(f'popup "{message}"')
        return True
    
    def close_popup(self) -> bool:
        """Close popup on teach pendant."""
        response = self._send_command("close popup")
        return True


class URScriptSender:
    """
    Send URScript commands via Secondary Interface (port 30002).
    
    This interface accepts URScript commands at 10Hz and doesn't conflict
    with EtherNet/IP adapter like RTDE does.
    """
    
    SECONDARY_PORT = 30002
    PRIMARY_PORT = 30001
    TIMEOUT = 2.0
    
    def __init__(self, robot_ip: str, port: int = 30002):
        self.robot_ip = robot_ip
        self.port = port
        self._lock = threading.Lock()
    
    def send_script(self, script: str) -> bool:
        """
        Send a URScript command to the robot.
        
        The command is executed immediately without needing a running program.
        Commands must end with newline.
        """
        with self._lock:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.TIMEOUT)
                sock.connect((self.robot_ip, self.port))
                
                # Ensure script ends with newline
                if not script.endswith('\n'):
                    script += '\n'
                
                sock.sendall(script.encode('utf-8'))
                sock.close()
                return True
            except Exception as e:
                logger.error(f"URScript send error: {e}")
                return False
    
    def movej(self, 
              joints: List[float], 
              a: float = 1.4, 
              v: float = 1.05, 
              t: float = 0, 
              r: float = 0) -> bool:
        """
        Move to joint positions.
        
        Args:
            joints: Target joint positions in radians [j0, j1, j2, j3, j4, j5]
            a: Joint acceleration (rad/s^2)
            v: Joint velocity (rad/s)
            t: Time for move (0 = not specified)
            r: Blend radius (m)
        """
        joints_str = f"[{', '.join(f'{j:.6f}' for j in joints)}]"
        script = f"movej({joints_str}, a={a}, v={v}, t={t}, r={r})"
        logger.debug(f"Sending: {script}")
        return self.send_script(script)
    
    def movel(self, 
              pose: List[float], 
              a: float = 1.2, 
              v: float = 0.25, 
              t: float = 0, 
              r: float = 0) -> bool:
        """
        Move linearly to a Cartesian pose.
        
        Args:
            pose: Target pose [x, y, z, rx, ry, rz] in meters/radians
            a: Tool acceleration (m/s^2)
            v: Tool velocity (m/s)
            t: Time for move (0 = not specified)
            r: Blend radius (m)
        """
        pose_str = f"p[{', '.join(f'{p:.6f}' for p in pose)}]"
        script = f"movel({pose_str}, a={a}, v={v}, t={t}, r={r})"
        logger.debug(f"Sending: {script}")
        return self.send_script(script)
    
    def movep(self, 
              pose: List[float], 
              a: float = 1.2, 
              v: float = 0.25, 
              r: float = 0) -> bool:
        """
        Move in process mode (constant velocity).
        
        Args:
            pose: Target pose [x, y, z, rx, ry, rz]
            a: Tool acceleration (m/s^2)
            v: Tool velocity (m/s)
            r: Blend radius (m)
        """
        pose_str = f"p[{', '.join(f'{p:.6f}' for p in pose)}]"
        script = f"movep({pose_str}, a={a}, v={v}, r={r})"
        return self.send_script(script)
    
    def speedl(self, 
               speeds: List[float], 
               a: float = 0.5, 
               t: float = 0) -> bool:
        """
        Move with constant Cartesian velocity.
        
        Args:
            speeds: Target speeds [vx, vy, vz, wrx, wry, wrz]
            a: Tool acceleration (m/s^2)
            t: Duration (0 = run until stopped)
        """
        speed_str = f"[{', '.join(f'{s:.6f}' for s in speeds)}]"
        script = f"speedl({speed_str}, a={a}, t={t})"
        return self.send_script(script)
    
    def speedj(self, 
               speeds: List[float], 
               a: float = 0.5, 
               t: float = 0) -> bool:
        """
        Move with constant joint velocity.
        
        Args:
            speeds: Target joint speeds [qd0, qd1, qd2, qd3, qd4, qd5]
            a: Joint acceleration (rad/s^2)
            t: Duration (0 = run until stopped)
        """
        speed_str = f"[{', '.join(f'{s:.6f}' for s in speeds)}]"
        script = f"speedj({speed_str}, a={a}, t={t})"
        return self.send_script(script)
    
    def servoj(self,
               joints: List[float],
               t: float = 0.008,
               lookahead_time: float = 0.1,
               gain: float = 300) -> bool:
        """
        Servo to joint position (for real-time control).
        
        Args:
            joints: Target joint positions [j0, j1, j2, j3, j4, j5]
            t: Time for move
            lookahead_time: Smoothing time
            gain: Servo gain
        """
        joints_str = f"[{', '.join(f'{j:.6f}' for j in joints)}]"
        script = f"servoj({joints_str}, t={t}, lookahead_time={lookahead_time}, gain={gain})"
        return self.send_script(script)
    
    def stopj(self, a: float = 2.0) -> bool:
        """Stop joint motion."""
        return self.send_script(f"stopj({a})")
    
    def stopl(self, a: float = 2.0) -> bool:
        """Stop linear motion."""
        return self.send_script(f"stopl({a})")
    
    def freedrive_mode(self) -> bool:
        """Enable freedrive mode."""
        return self.send_script("freedrive_mode()")
    
    def end_freedrive_mode(self) -> bool:
        """Disable freedrive mode."""
        return self.send_script("end_freedrive_mode()")
    
    def set_digital_out(self, pin: int, value: bool) -> bool:
        """Set digital output."""
        return self.send_script(f"set_digital_out({pin}, {str(value)})")
    
    def set_analog_out(self, pin: int, value: float) -> bool:
        """Set analog output."""
        return self.send_script(f"set_analog_out({pin}, {value})")


class RealtimeStateReader:
    """
    Read robot state from Realtime Interface (port 30003).
    
    This interface provides robot state at 500Hz without conflicting
    with EtherNet/IP adapter.
    
    The data is sent as binary packets with the following structure:
    - Packet length (int32)
    - Time (double)
    - Joint positions q_target (6 doubles)
    - Joint velocities qd_target (6 doubles)
    - Joint accelerations qdd_target (6 doubles)
    - ... many more fields
    """
    
    REALTIME_PORT = 30003
    PACKET_SIZE = 1116  # UR e-Series packet size
    
    def __init__(self, robot_ip: str):
        self.robot_ip = robot_ip
        self._socket: Optional[socket.socket] = None
        self._state = RobotState()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
    
    def connect(self) -> bool:
        """Connect to realtime interface."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(2.0)
            self._socket.connect((self.robot_ip, self.REALTIME_PORT))
            self._socket.setblocking(True)
            logger.info("Realtime interface connected")
            return True
        except Exception as e:
            logger.error(f"Realtime connection failed: {e}")
            self._socket = None
            return False
    
    def disconnect(self):
        """Disconnect from realtime interface."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
            self._socket = None
    
    def start(self):
        """Start reading state in background thread."""
        if self._running:
            return
        
        if not self._socket:
            if not self.connect():
                return
        
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop reading state."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _read_loop(self):
        """Background thread to read robot state."""
        buffer = b''
        
        while self._running and self._socket:
            try:
                # Read data
                data = self._socket.recv(4096)
                if not data:
                    logger.warning("Realtime interface disconnected")
                    break
                
                buffer += data
                
                # Process complete packets
                while len(buffer) >= 4:
                    # Read packet length (first 4 bytes, big-endian int)
                    packet_len = struct.unpack('>i', buffer[:4])[0]
                    
                    if len(buffer) < packet_len:
                        break  # Wait for complete packet
                    
                    # Extract packet
                    packet = buffer[:packet_len]
                    buffer = buffer[packet_len:]
                    
                    # Parse packet
                    self._parse_packet(packet)
                    
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Realtime read error: {e}")
                break
        
        self._running = False
    
    def _parse_packet(self, packet: bytes):
        """Parse a realtime state packet."""
        if len(packet) < 140:  # Minimum size for basic state
            return
        
        with self._lock:
            try:
                # Packet structure (e-Series, version 5.x):
                # All values are big-endian doubles unless noted
                offset = 4  # Skip packet length
                
                # Time
                self._state.timestamp = struct.unpack('>d', packet[offset:offset+8])[0]
                offset += 8
                
                # q_target (6 doubles)
                offset += 6 * 8  # Skip target positions
                
                # qd_target (6 doubles)
                offset += 6 * 8  # Skip target velocities
                
                # qdd_target (6 doubles)
                offset += 6 * 8  # Skip target accelerations
                
                # i_target (6 doubles) - target currents
                offset += 6 * 8  # Skip
                
                # m_target (6 doubles) - target torques
                offset += 6 * 8  # Skip
                
                # q_actual (6 doubles) - ACTUAL JOINT POSITIONS
                self._state.joint_positions = list(struct.unpack('>6d', packet[offset:offset+48]))
                offset += 48
                
                # qd_actual (6 doubles) - ACTUAL JOINT VELOCITIES
                self._state.joint_velocities = list(struct.unpack('>6d', packet[offset:offset+48]))
                offset += 48
                
                # i_actual (6 doubles) - ACTUAL CURRENTS
                self._state.joint_currents = list(struct.unpack('>6d', packet[offset:offset+48]))
                offset += 48
                
                # Skip to TCP data (offset around 444)
                # i_control, tool_vector_actual, TCP_speed_actual
                offset = 444
                
                if len(packet) > offset + 48:
                    # TCP_force (6 doubles)
                    self._state.tcp_force = list(struct.unpack('>6d', packet[offset:offset+48]))
                    offset += 48
                
                # TCP_pose_target
                offset += 48
                
                if len(packet) > offset + 48:
                    # TCP_speed_target
                    offset += 48
                
                # Digital inputs/outputs are at different offsets
                # Skip to robot mode (around offset 756)
                if len(packet) > 764:
                    # Robot mode
                    robot_mode_raw = struct.unpack('>d', packet[756:764])[0]
                    try:
                        self._state.robot_mode = RobotMode(int(robot_mode_raw))
                    except:
                        pass
                
                if len(packet) > 772:
                    # Safety mode
                    safety_mode_raw = struct.unpack('>d', packet[764:772])[0]
                    try:
                        self._state.safety_mode = SafetyMode(int(safety_mode_raw))
                    except:
                        pass
                
                self._state.connected = True
                
            except Exception as e:
                logger.debug(f"Packet parse error: {e}")
    
    @property
    def state(self) -> RobotState:
        """Get current robot state (thread-safe copy)."""
        with self._lock:
            return RobotState(
                connected=self._state.connected,
                joint_positions=self._state.joint_positions.copy(),
                joint_velocities=self._state.joint_velocities.copy(),
                joint_currents=self._state.joint_currents.copy(),
                joint_temperatures=self._state.joint_temperatures.copy(),
                tcp_pose=self._state.tcp_pose.copy(),
                tcp_speed=self._state.tcp_speed.copy(),
                tcp_force=self._state.tcp_force.copy(),
                robot_mode=self._state.robot_mode,
                safety_mode=self._state.safety_mode,
                program_running=self._state.program_running,
                is_power_on=self._state.is_power_on,
                is_brakes_released=self._state.is_brakes_released,
                timestamp=self._state.timestamp,
            )


class URRobotController:
    """
    High-level controller for UR robots.
    
    Combines Dashboard, URScript, and Realtime interfaces for complete
    robot control without RTDE conflicts.
    
    Usage:
        robot = URRobotController("192.168.1.9")
        if robot.connect():
            robot.power_on()
            robot.movej([0, -1.57, 1.57, -1.57, -1.57, 0])
            state = robot.get_state()
    """
    
    def __init__(self, robot_ip: str):
        self.robot_ip = robot_ip
        self.dashboard = DashboardClient(robot_ip)
        self.script = URScriptSender(robot_ip)
        self.realtime = RealtimeStateReader(robot_ip)
        self._connected = False
    
    def connect(self) -> bool:
        """Connect to all robot interfaces."""
        logger.info(f"Connecting to UR robot at {self.robot_ip}...")
        
        # Connect dashboard
        if not self.dashboard.connect():
            logger.error("Dashboard connection failed")
            return False
        
        # Check if in remote mode
        if not self.dashboard.is_in_remote_control():
            logger.error("Robot is not in Remote Control mode!")
            logger.error("Please toggle the switch at top-right of teach pendant")
            return False
        
        # Connect realtime state reader
        if not self.realtime.connect():
            logger.warning("Realtime interface connection failed - state reading disabled")
        else:
            self.realtime.start()
        
        self._connected = True
        logger.info("Connected to UR robot successfully")
        return True
    
    def disconnect(self):
        """Disconnect from all interfaces."""
        self.realtime.stop()
        self.realtime.disconnect()
        self.dashboard.disconnect()
        self._connected = False
        logger.info("Disconnected from UR robot")
    
    def is_connected(self) -> bool:
        """Check if connected to robot."""
        return self._connected
    
    # Power and safety
    def power_on(self) -> bool:
        """Power on the robot and release brakes."""
        if not self.dashboard.power_on():
            return False
        time.sleep(2)  # Wait for power on
        return self.dashboard.release_brakes()
    
    def power_off(self) -> bool:
        """Power off the robot."""
        return self.dashboard.power_off()
    
    def unlock_protective_stop(self) -> bool:
        """Unlock after protective stop."""
        return self.dashboard.unlock_protective_stop()
    
    def close_safety_popup(self) -> bool:
        """Close safety popup."""
        return self.dashboard.close_safety_popup()
    
    # Motion commands
    def movej(self, joints: List[float], velocity: float = 1.05, acceleration: float = 1.4) -> bool:
        """Move to joint positions."""
        return self.script.movej(joints, a=acceleration, v=velocity)
    
    def movel(self, pose: List[float], velocity: float = 0.25, acceleration: float = 1.2) -> bool:
        """Move linearly to Cartesian pose."""
        return self.script.movel(pose, a=acceleration, v=velocity)
    
    def stop(self) -> bool:
        """Stop all motion."""
        return self.script.stopj(2.0) and self.script.stopl(2.0)
    
    def freedrive(self, enable: bool = True) -> bool:
        """Enable/disable freedrive mode."""
        if enable:
            return self.script.freedrive_mode()
        else:
            return self.script.end_freedrive_mode()
    
    # State
    def get_state(self) -> RobotState:
        """Get current robot state."""
        return self.realtime.state
    
    def get_joint_positions(self) -> List[float]:
        """Get current joint positions."""
        return self.realtime.state.joint_positions
    
    def get_tcp_pose(self) -> List[float]:
        """Get current TCP pose."""
        return self.realtime.state.tcp_pose
    
    def get_robot_mode(self) -> str:
        """Get robot mode string from dashboard."""
        return self.dashboard.get_robot_mode()
    
    def get_safety_mode(self) -> str:
        """Get safety mode string from dashboard."""
        return self.dashboard.get_safety_mode()
    
    # Program control
    def load_program(self, program_name: str) -> bool:
        """Load a program from the robot."""
        return self.dashboard.load_program(program_name)
    
    def play_program(self) -> bool:
        """Start the loaded program."""
        return self.dashboard.play()
    
    def stop_program(self) -> bool:
        """Stop the running program."""
        return self.dashboard.stop()
    
    def pause_program(self) -> bool:
        """Pause the running program."""
        return self.dashboard.pause()


# Simple test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    robot = URRobotController("192.168.1.9")
    
    if robot.connect():
        print(f"Robot mode: {robot.get_robot_mode()}")
        print(f"Safety mode: {robot.get_safety_mode()}")
        
        time.sleep(1)
        state = robot.get_state()
        print(f"Joint positions: {state.joint_positions}")
        
        robot.disconnect()
    else:
        print("Failed to connect to robot")
