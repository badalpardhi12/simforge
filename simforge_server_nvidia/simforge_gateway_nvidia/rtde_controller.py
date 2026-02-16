"""
Direct RTDE Robot Controller.

Communicates with the UR controller via the RTDE protocol using
the ur_rtde library, bypassing the ROS2 control stack.

Key methods:
  - move_j()                    — blocking joint move (for homing)
  - execute_trajectory_servoj() — streaming via servoJ at native trajectory rate
"""

import threading
import time as _time
from typing import Callable, List, Optional

from .config import RTDE_AVAILABLE

if RTDE_AVAILABLE:
    import rtde_control
    import rtde_receive


class URRTDEController:
    """Low-level wrapper around ur_rtde for a single UR arm."""

    def __init__(self, robot_name: str, ip: str, logger=None):
        self.robot_name = robot_name
        self.ip = ip
        self.logger = logger
        self._ctrl = None
        self._recv = None
        self._connected = False
        self._recv_healthy = False  # tracks receive interface health
        self.last_error: str = ""  # last error for callers to inspect

    # ── Connection ───────────────────────────────────────────────

    def connect(self) -> bool:
        """Connect both the control and receive RTDE interfaces."""
        if not RTDE_AVAILABLE:
            if self.logger:
                self.logger.error("ur_rtde not installed")
            return False
        try:
            if self.logger:
                self.logger.info(
                    f"Connecting RTDE to {self.robot_name} at {self.ip}..."
                )
            # Only create the *receive* interface here.
            # The *control* interface (RTDEControlInterface) is created
            # on demand in _ensure_ctrl() and torn down immediately
            # after each move/servoJ — its internal C++ auto-reconnect
            # thread segfaults when the UR drops the connection while
            # the object is idle.
            self._recv = rtde_receive.RTDEReceiveInterface(self.ip)
            self._connected = True
            self._recv_healthy = True
            if self.logger:
                self.logger.info(
                    f"RTDE connected to {self.robot_name} at {self.ip} ✓"
                )
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"RTDE connection failed for {self.robot_name}: {e}"
                )
            self._connected = False
            self._recv_healthy = False
            return False

    def disconnect(self):
        """Disconnect both RTDE interfaces."""
        try:
            if self._ctrl:
                self._ctrl.disconnect()
            if self._recv:
                self._recv.disconnect()
        except Exception:
            pass
        self._ctrl = None
        self._recv = None
        self._connected = False
        self._recv_healthy = False

    def reconnect_receive(self) -> bool:
        """Reconnect the RTDE receive interface (non-blocking, with timeout).

        The receive interface often dies during servoJ streaming.  This
        method creates a **fresh** RTDEReceiveInterface to restore the
        50 Hz joint-state reads.

        IMPORTANT: We never call `self._recv.reconnect()` because the
        C++ RTDE library can segfault when calling `.reconnect()` on an
        object whose internal connection state was corrupted during
        servoJ streaming.  A segfault in a thread kills the entire
        process.  Instead we dispose of the old object and create a
        brand-new one.
        """
        if not RTDE_AVAILABLE:
            return False

        if self.logger:
            self.logger.info(
                f"Reconnecting RTDE receive interface for "
                f"{self.robot_name}..."
            )

        # Dispose of the old receive interface first.
        # Do NOT call .reconnect() on it — the C++ internals can
        # segfault after servoJ corrupts the connection state.
        # We intentionally leak the old object rather than calling
        # .disconnect() or letting the destructor run, because the
        # C++ destructor can also segfault on a corrupted object.
        # The leaked memory (~few KB) is reclaimed when the process
        # exits and is far preferable to a process-killing segfault.
        old_recv = self._recv
        self._recv = None
        self._recv_healthy = False
        if old_recv is not None:
            try:
                old_recv.disconnect()
            except Exception:
                pass
            # Do not 'del old_recv' — let it leak if disconnect failed.
            # A C++ destructor on corrupted state can segfault.

        result_holder = [None]  # will hold the new interface or None

        def _do_reconnect():
            try:
                new_recv = rtde_receive.RTDEReceiveInterface(self.ip)
                result_holder[0] = new_recv
            except Exception as e:
                if self.logger:
                    self.logger.warn(
                        f"RTDE recv reconnect thread error: {e}"
                    )

        t = threading.Thread(target=_do_reconnect, daemon=True)
        t.start()
        t.join(timeout=5.0)

        if t.is_alive():
            if self.logger:
                self.logger.warn(
                    f"RTDE recv reconnect timed out for {self.robot_name}"
                )
            self._recv_healthy = False
            return False

        if not result_holder[0]:
            if self.logger:
                self.logger.warn(
                    f"RTDE recv reconnect FAILED for {self.robot_name}"
                )
            self._recv_healthy = False
            return False

        # Install the new interface
        self._recv = result_holder[0]

        # Verify it actually works
        try:
            q = self._recv.getActualQ() if self._recv else None
            if q is not None:
                self._recv_healthy = True
                if self.logger:
                    self.logger.info(
                        f"RTDE receive interface reconnected for "
                        f"{self.robot_name} ✓"
                    )
                return True
        except Exception:
            pass

        self._recv_healthy = False
        if self.logger:
            self.logger.warn(
                f"RTDE recv reconnect verify FAILED for {self.robot_name}"
            )
        return False

    @property
    def is_connected(self) -> bool:
        return (
            self._connected
            and self._recv is not None
        )

    def _teardown_ctrl(self) -> None:
        """Forcibly tear down the RTDE control interface.

        Calls stopScript() then disconnect() to kill the UR control
        script *and* ur_rtde's internal C++ auto-reconnect thread.
        Without stopScript(), disconnect() alone leaves the C++
        reconnect thread alive — it will loop, reconnect, and hold
        the RTDE input registers, causing the next
        RTDEControlInterface constructor to fail with:
          'One of the RTDE input registers are already in use!'
        """
        if self._ctrl is None:
            return
        try:
            self._ctrl.stopScript()
        except Exception:
            pass
        try:
            self._ctrl.disconnect()
        except Exception:
            pass
        self._ctrl = None

    def _ensure_ctrl(self) -> bool:
        """Create a fresh RTDEControlInterface on demand.

        The control interface is intentionally short-lived: created
        just before moveJ / servoJ and torn down immediately after,
        because its internal C++ auto-reconnect thread can segfault
        when the UR drops the RTDE link while the object is idle.

        Safety checks:
        - If the robot is in protective stop or e-stop, we refuse to
          create the interface (the UR control script upload will fail
          or segfault).
        - If the robot mode is not RUNNING (7), we refuse — the UR
          control script cannot execute without an active program.
        - If the recv interface is dead and we cannot verify the robot
          state, we refuse — creating the control interface blindly
          can segfault the process.
        """
        if self._ctrl is not None:
            try:
                if self._ctrl.isConnected():
                    return True
            except Exception:
                pass

        # Tear down any dead leftover (stopScript + disconnect)
        self._teardown_ctrl()

        # Pre-check: refuse if robot is not in a safe state.
        # The recv interface may have died (common after mode switch),
        # so try to reconnect it first for the safety check.
        if self._recv is not None and not self._recv_healthy:
            self.reconnect_receive()

        # If recv is still dead after reconnect attempt, we CANNOT
        # verify the robot state.  Refuse to create the control
        # interface — the RTDEControlInterface constructor can segfault
        # when the robot program is not running, and without recv we
        # have no way to check.
        if self._recv is None or not self._recv_healthy:
            msg = (
                f"Cannot create RTDE control interface — "
                f"{self.robot_name} receive interface is dead "
                f"(cannot verify robot state). Reconnect first."
            )
            self.last_error = msg
            if self.logger:
                self.logger.error(msg)
            return False

        # recv is healthy — run safety checks
        try:
            if self._recv.isProtectiveStopped():
                msg = (
                    f"Cannot create RTDE control interface — "
                    f"{self.robot_name} is in PROTECTIVE STOP. "
                    f"Clear on teach pendant first."
                )
                self.last_error = msg
                if self.logger:
                    self.logger.error(msg)
                return False
            if self._recv.isEmergencyStopped():
                msg = (
                    f"Cannot create RTDE control interface — "
                    f"{self.robot_name} is in EMERGENCY STOP. "
                    f"Clear on teach pendant first."
                )
                self.last_error = msg
                if self.logger:
                    self.logger.error(msg)
                return False
        except Exception as e:
            msg = (
                f"Cannot create RTDE control interface — "
                f"{self.robot_name} safety check failed: {e}"
            )
            self.last_error = msg
            if self.logger:
                self.logger.error(msg)
            return False

        # Check robot mode: 7 = RUNNING (program active).
        # Without a running program, the RTDEControlInterface
        # constructor will upload a control script that immediately
        # dies, triggering a reconnect storm that holds the RTDE
        # input registers and blocks all future connections.
        try:
            robot_mode = self._recv.getRobotMode()
            if robot_mode != 7:
                msg = (
                    f"Cannot create RTDE control interface — "
                    f"{self.robot_name} robot mode is "
                    f"{robot_mode} (need 7/RUNNING). "
                    f"Start the robot program on the teach "
                    f"pendant first."
                )
                self.last_error = msg
                if self.logger:
                    self.logger.error(msg)
                return False
        except Exception as e:
            msg = (
                f"Cannot create RTDE control interface — "
                f"{self.robot_name} mode check failed: {e}"
            )
            self.last_error = msg
            if self.logger:
                self.logger.error(msg)
            return False

        # Create the control interface directly in-process.
        # FLAG_NO_WAIT: prevents servoJ() / moveJ() from internally
        # blocking via the C++ steady_clock wait after each command.
        # On aarch64 (Jetson), the C++ steady_clock / nanosleep
        # malfunctions — causing each servoJ() call to block for
        # ~20-50 ms instead of <1 ms, dropping the 500 Hz streaming
        # rate to 21-56 Hz.  With FLAG_NO_WAIT, servoJ() returns
        # immediately and our Python timing loop handles pacing.
        # moveJ() still blocks correctly via its own isSteady() loop.
        try:
            _flags = (
                rtde_control.RTDEControlInterface.FLAG_UPLOAD_SCRIPT
                | rtde_control.RTDEControlInterface.FLAG_NO_WAIT
            )
            self._ctrl = rtde_control.RTDEControlInterface(
                self.ip, frequency=-1.0, flags=_flags,
            )
            if self.logger:
                self.logger.info(
                    f"RTDE control interface created for "
                    f"{self.robot_name} (FLAG_NO_WAIT)"
                )
            return True
        except Exception as e:
            self._ctrl = None
            msg = (
                f"Failed to create RTDE control interface "
                f"for {self.robot_name}: {e}"
            )
            self.last_error = msg
            if self.logger:
                self.logger.error(msg)
            return False

    # ── Safety state ─────────────────────────────────────────────

    def is_protective_stopped(self) -> bool:
        """Check if the robot is in a protective stop state."""
        try:
            if self._recv is not None and self._recv_healthy:
                return self._recv.isProtectiveStopped()
        except Exception:
            pass
        return False

    def is_emergency_stopped(self) -> bool:
        """Check if the robot is in an emergency stop state."""
        try:
            if self._recv is not None and self._recv_healthy:
                return self._recv.isEmergencyStopped()
        except Exception:
            pass
        return False

    def get_robot_mode(self) -> int:
        """Return the UR robot mode integer (-1 if unavailable).

        7 = RUNNING (normal), 3 = POWER_OFF, 5 = IDLE, etc.
        """
        try:
            if self._recv is not None and self._recv_healthy:
                return self._recv.getRobotMode()
        except Exception:
            pass
        return -1

    # ── Joint state reads ────────────────────────────────────────

    def get_actual_q(self) -> Optional[List[float]]:
        """Return current joint positions (radians) or None."""
        try:
            if self._recv is not None and self._recv_healthy:
                q = list(self._recv.getActualQ())
                return q
        except Exception:
            # Receive interface has died (common during servoJ)
            self._recv_healthy = False
        return None

    def get_actual_qd(self) -> Optional[List[float]]:
        """Return current joint velocities (rad/s) or None."""
        try:
            if self._recv is not None and self._recv_healthy:
                return list(self._recv.getActualQd())
        except Exception:
            self._recv_healthy = False
        return None

    # ── Blocking joint move ──────────────────────────────────────

    def move_j(
        self,
        target_q: List[float],
        speed: float = 1.05,
        acceleration: float = 1.4,
        timeout: float = 30.0,
    ) -> bool:
        """Blocking joint move via RTDE moveJ with timeout safety.

        Creates a fresh control interface if the current one is dead
        (e.g. after servoJ), then runs moveJ in a daemon thread so
        it can be killed on timeout.
        """
        if not RTDE_AVAILABLE:
            return False
        try:
            if not self._ensure_ctrl():
                if self.logger:
                    self.logger.error(
                        f"Cannot create RTDE control interface "
                        f"for {self.robot_name} — moveJ aborted"
                    )
                return False

            if self.logger:
                self.logger.info(
                    f"RTDE moveJ {self.robot_name}: "
                    f"speed={speed:.2f}, accel={acceleration:.2f}"
                )

            result_holder = [False]
            exc_holder = [None]

            def _do_move():
                try:
                    self._ctrl.moveJ(target_q, speed, acceleration)
                    result_holder[0] = True
                except Exception as e:
                    exc_holder[0] = e

            t = threading.Thread(target=_do_move, daemon=True)
            t.start()
            t.join(timeout=timeout)

            if t.is_alive():
                if self.logger:
                    self.logger.warn(
                        f"RTDE moveJ timed out after {timeout}s "
                        f"on {self.robot_name}"
                    )
                try:
                    self._ctrl.stopJ(2.0)
                except Exception:
                    pass
                self._teardown_ctrl()
                return False

            if exc_holder[0]:
                # moveJ threw — tear down before re-raising so the
                # auto-reconnect thread doesn't linger and hold the
                # RTDE input registers.
                self._teardown_ctrl()
                raise exc_holder[0]

            # moveJ finished — tear down the control interface to
            # prevent ur_rtde's internal auto-reconnect from looping
            # (and eventually segfaulting) while the UR control script
            # is stopped.  A fresh interface will be created on demand.
            self._teardown_ctrl()

            return result_holder[0]

        except Exception as e:
            if self.logger:
                self.logger.error(f"RTDE moveJ failed: {e}")
            self._teardown_ctrl()
            return False

    # ── Trajectory streaming ─────────────────────────────────────

    def execute_trajectory_servoj(
        self,
        positions: list,
        velocities: Optional[list] = None,
        dt: float = 0.002,
        timestamps: Optional[list] = None,
        lookahead_time: float = 0.1,
        gain: int = 300,
        stop_check_fn=None,
        logger=None,
        position_callback: Optional[Callable[[str, List[float]], None]] = None,
    ) -> bool:
        """Stream a trajectory using servoJ at the trajectory's native rate.

        Sends each trajectory waypoint directly via servoJ without
        upsampling to 500 Hz.  On aarch64 (Jetson AGX Thor) the
        Python→C++ servoJ() call overhead is ~25-150 ms, making 500 Hz
        impossible.  Instead we send waypoints at whatever rate the
        system can sustain (typically 15-50 Hz) and rely on servoJ's
        ``lookahead_time`` for smooth interpolation on the robot side.

        Parameters
        ----------
        positions : list[list[float]]
            Joint position waypoints (n_pts × 6).
        velocities : optional — not used by servoJ.
        dt : float
            Uniform time-step (used only when *timestamps* is None).
        timestamps : list[float], optional
            Per-waypoint time-from-start in seconds.
        lookahead_time, gain : servoJ tuning parameters.
        stop_check_fn : callable → bool, optional
            Return True to abort.
        logger : optional ROS2 logger.
        position_callback : callable(robot_name, q_target), optional
            Called at ~50 Hz with the commanded joint position.
        """
        if not self.is_connected:
            if logger:
                logger.error(f"RTDE not connected to {self.robot_name}")
            return False

        if not self._ensure_ctrl():
            if logger:
                logger.error(
                    f"Cannot create RTDE control interface "
                    f"for {self.robot_name} — servoJ aborted"
                )
            return False

        n_pts = len(positions)
        if n_pts < 2:
            if logger:
                logger.warn(f"Trajectory too short ({n_pts} points)")
            return True

        # Build per-waypoint timestamps
        wp_times = list(timestamps) if timestamps is not None else [
            i * dt for i in range(n_pts)
        ]

        total_time = wp_times[-1]

        # Mark recv unhealthy immediately so the 50 Hz RTDE timer
        # does not try to read (and publish stale data) for this
        # robot while the position_callback owns publishing.
        self._recv_healthy = False

        if logger:
            logger.info(
                f"RTDE servoJ streaming {self.robot_name}: "
                f"{n_pts} waypoints, duration={total_time:.2f}s "
                f"(native rate, no 500 Hz upsample)"
            )

        # Force higher lookahead for smoother motion given our
        # variable command rate (15-50 Hz actual on Jetson).
        effective_lookahead = max(lookahead_time, 0.2)

        try:
            # Publish every Nth waypoint to hit ~50 Hz.
            # With trajectory dt=0.02 (50 Hz), publish every point.
            traj_dt = wp_times[1] - wp_times[0] if n_pts > 1 else 0.02
            cb_interval = max(1, int(0.02 / max(traj_dt, 0.001)))
            cb_logged = False

            # Safety-check every M waypoints (~5 Hz to minimize overhead)
            safety_interval = max(1, int(0.2 / max(traj_dt, 0.001)))

            # ── Timing diagnostics ──────────────────────────────────
            wall_start = _time.monotonic()
            timing_log_interval = max(1, n_pts // 5)

            # ── Native-rate servoJ loop ─────────────────────────────
            # Instead of upsampling to 500 Hz (impossible on aarch64),
            # send original trajectory waypoints at whatever rate
            # Python + ur_rtde can sustain.  The robot's servoJ handles
            # smooth interpolation via lookahead_time.
            prev_cmd_time = _time.monotonic()

            for wi in range(n_pts):
                if stop_check_fn and stop_check_fn():
                    if logger:
                        logger.info("RTDE servoJ aborted by stop request")
                    try:
                        self._ctrl.servoStop()
                    except Exception:
                        pass
                    self._teardown_ctrl()
                    self.reconnect_receive()
                    return False

                q_target = positions[wi]

                # ── Safety checks (at ~5 Hz) ────────────────────────
                if wi % safety_interval == 0:
                    try:
                        if not self._ctrl.isProgramRunning():
                            reason = "PROGRAM STOPPED"
                            try:
                                if (self._recv is not None
                                        and self._recv.isProtectiveStopped()):
                                    reason = "PROTECTIVE STOP"
                                elif (self._recv is not None
                                        and self._recv.isEmergencyStopped()):
                                    reason = "EMERGENCY STOP"
                            except Exception:
                                pass
                            msg = (
                                f"{reason} on {self.robot_name} "
                                f"at wp {wi}/{n_pts} — aborting servoJ"
                            )
                            self.last_error = msg
                            if logger:
                                logger.error(msg)
                            self._teardown_ctrl()
                            self.reconnect_receive()
                            return False
                    except Exception:
                        msg = (
                            f"RTDE control interface dead during "
                            f"servoJ safety check on {self.robot_name} "
                            f"at wp {wi}/{n_pts}"
                        )
                        self.last_error = msg
                        if logger:
                            logger.error(msg)
                        self._teardown_ctrl()
                        self.reconnect_receive()
                        return False

                    # TCP disconnect check
                    try:
                        ctrl_connected = self._ctrl.isConnected()
                    except Exception:
                        ctrl_connected = False
                    if not ctrl_connected:
                        msg = (
                            f"RTDE control interface lost during servoJ "
                            f"on {self.robot_name} at wp {wi}/{n_pts}"
                        )
                        self.last_error = msg
                        if logger:
                            logger.warn(msg)
                        self._teardown_ctrl()
                        self.reconnect_receive()
                        return False

                # ── Compute adaptive servo_dt ───────────────────────
                # Tell the robot how long until the next command.
                # Use the ACTUAL measured interval between commands
                # (clamped) so the robot's internal servo matches
                # our real update rate.
                now = _time.monotonic()
                measured_dt = now - prev_cmd_time
                # For the first command, use a nominal value
                if wi == 0:
                    servo_dt_cmd = 0.02
                else:
                    # Clamp to [8 ms, 200 ms] — reasonable for servoJ
                    servo_dt_cmd = max(0.008, min(0.2, measured_dt))

                try:
                    servo_ok = self._ctrl.servoJ(
                        q_target, 0.0, 0.0, servo_dt_cmd,
                        effective_lookahead, gain,
                    )
                except Exception as servo_exc:
                    msg = (
                        f"RTDE servoJ exception on "
                        f"{self.robot_name} at wp {wi}/{n_pts}: "
                        f"{servo_exc}"
                    )
                    self.last_error = msg
                    if logger:
                        logger.error(msg)
                    self._teardown_ctrl()
                    self.reconnect_receive()
                    return False

                prev_cmd_time = _time.monotonic()

                if not servo_ok:
                    msg = (
                        f"servoJ returned False on {self.robot_name} "
                        f"at wp {wi}/{n_pts} — UR control script "
                        f"not running (likely protective stop)"
                    )
                    self.last_error = msg
                    if logger:
                        logger.error(msg)
                    self._teardown_ctrl()
                    self.reconnect_receive()
                    return False

                # Publish commanded position
                if position_callback and wi % cb_interval == 0:
                    try:
                        position_callback(self.robot_name, q_target)
                        if not cb_logged and logger:
                            logger.info(
                                f"servoJ position callback active for "
                                f"{self.robot_name}"
                            )
                            cb_logged = True
                    except Exception:
                        pass

                # ── Rate pacing ─────────────────────────────────────
                # Sleep until the target time for the NEXT waypoint.
                # This keeps motion at real-time speed when the system
                # can sustain the trajectory rate.  When it can't, the
                # sleep is skipped and commands are sent as fast as
                # possible — the motion stretches but stays smooth.
                if wi + 1 < n_pts:
                    target_wall = wall_start + wp_times[wi + 1]
                    remaining = target_wall - _time.monotonic()
                    if remaining > 0.001:
                        _time.sleep(remaining)

                # ── Timing sample ───────────────────────────────────
                if wi % timing_log_interval == 0 or wi == n_pts - 1:
                    wall_elapsed = _time.monotonic() - wall_start
                    expected = wp_times[wi]
                    drift = wall_elapsed - expected
                    if logger:
                        logger.info(
                            f"servoJ timing [{self.robot_name}] "
                            f"wp {wi}/{n_pts}: "
                            + (
                                f"wall={wall_elapsed:.3f}s, "
                                f"expected={expected:.3f}s, "
                                f"drift={drift:+.3f}s "
                                f"({drift/expected*100:+.1f}%)"
                                if expected > 0 else "start"
                            )
                        )

            # Clean stop
            try:
                self._ctrl.servoStop()
            except Exception:
                pass
            _time.sleep(0.5)

            self._teardown_ctrl()
            self.reconnect_receive()
            self.last_error = ""

            # ── Timing summary ──────────────────────────────────────
            wall_total = _time.monotonic() - wall_start
            drift_total = wall_total - total_time
            if logger:
                actual_rate = n_pts / wall_total if wall_total > 0 else 0
                logger.info(
                    f"RTDE servoJ complete for {self.robot_name} ✓  "
                    f"wall={wall_total:.2f}s vs planned={total_time:.2f}s  "
                    f"drift={drift_total:+.2f}s  "
                    f"actual_rate={actual_rate:.0f}Hz "
                    f"({n_pts} waypoints, no upsample)"
                )
            return True

        except Exception as e:
            if logger:
                logger.error(
                    f"RTDE servoJ error on {self.robot_name}: {e}"
                )
            try:
                self._ctrl.servoStop()
            except Exception:
                pass
            self._teardown_ctrl()
            self.reconnect_receive()
            return False

    # ── Position check ───────────────────────────────────────────

    def is_at_position(
        self, target_q: List[float], tolerance: float = 0.05
    ) -> bool:
        actual = self.get_actual_q()
        if actual is None:
            return False
        return all(
            abs(actual[i] - target_q[i]) < tolerance for i in range(6)
        )


# ── Helper ───────────────────────────────────────────────────────


def _reconnect_with_timeout(
    ctrl, *, timeout: float = 5.0, logger=None, robot_name: str = ""
):
    """Try ctrl.reconnect() in a daemon thread with a hard timeout."""
    def _try():
        try:
            if not ctrl.isConnected():
                ctrl.reconnect()
        except Exception:
            pass

    rc = threading.Thread(target=_try, daemon=True)
    rc.start()
    rc.join(timeout=timeout)
    if rc.is_alive() and logger:
        logger.warn(
            f"RTDE reconnect timed out for {robot_name} — "
            f"will reconnect later"
        )
