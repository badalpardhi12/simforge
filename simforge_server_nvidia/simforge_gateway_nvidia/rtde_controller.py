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

        # Position verification — populated after each trajectory
        self.last_target_q: Optional[List[float]] = None
        self.last_actual_q_post_exec: Optional[List[float]] = None
        self.last_position_error_rad: Optional[List[float]] = None

    # ── Connection ───────────────────────────────────────────────

    def connect(self, timeout: float = 10.0) -> bool:
        """Connect the RTDE receive interface.

        Args:
            timeout: Maximum seconds to wait for the TCP connection
                     before giving up.  Default 10 s (vs the Linux
                     TCP default of ~130 s).
        """
        if not RTDE_AVAILABLE:
            if self.logger:
                self.logger.error("ur_rtde not installed")
            return False

        if self.logger:
            self.logger.info(
                f"Connecting RTDE to {self.robot_name} at {self.ip} "
                f"(timeout {timeout}s)..."
            )

        # RTDEReceiveInterface() blocks on TCP connect with no
        # timeout parameter.  Run it in a daemon thread so we can
        # enforce our own deadline and avoid hanging the gateway
        # for 2+ minutes when the robot is unreachable.
        result_holder: list = [None]
        error_holder: list = [None]

        def _do_connect():
            try:
                recv = rtde_receive.RTDEReceiveInterface(self.ip)
                result_holder[0] = recv
            except Exception as e:
                error_holder[0] = e

        t = threading.Thread(target=_do_connect, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            # Thread still blocked on connect — treat as failure
            if self.logger:
                self.logger.error(
                    f"RTDE connection timed out for {self.robot_name} "
                    f"at {self.ip} after {timeout}s — is the robot "
                    f"powered on and reachable?"
                )
            self._connected = False
            self._recv_healthy = False
            return False

        if error_holder[0] is not None:
            if self.logger:
                self.logger.error(
                    f"RTDE connection failed for {self.robot_name}: "
                    f"{error_holder[0]}"
                )
            self._connected = False
            self._recv_healthy = False
            return False

        self._recv = result_holder[0]
        self._connected = True
        self._recv_healthy = True
        if self.logger:
            self.logger.info(
                f"RTDE connected to {self.robot_name} at {self.ip} ✓"
            )
        return True

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
            sm = self.get_safety_mode()
            if sm in (
                self.SAFETY_MODE_SAFEGUARD_STOP,
                self.SAFETY_MODE_AUTO_SAFEGUARD_STOP,
            ):
                msg = (
                    f"Cannot create RTDE control interface — "
                    f"{self.robot_name} is in SAFEGUARD STOP "
                    f"(safety_mode={sm}). Wait for area to clear."
                )
                self.last_error = msg
                if self.logger:
                    self.logger.warn(msg)
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

    # UR safety mode constants (from RTDE specification)
    SAFETY_MODE_NORMAL = 1
    SAFETY_MODE_REDUCED = 2
    SAFETY_MODE_PROTECTIVE_STOP = 3
    SAFETY_MODE_RECOVERY = 4
    SAFETY_MODE_SAFEGUARD_STOP = 5
    SAFETY_MODE_SYSTEM_EMERGENCY_STOP = 6
    SAFETY_MODE_ROBOT_EMERGENCY_STOP = 7
    SAFETY_MODE_VIOLATION = 8
    SAFETY_MODE_FAULT = 9
    SAFETY_MODE_AUTO_SAFEGUARD_STOP = 12

    def get_safety_mode(self) -> int:
        """Return the UR safety mode integer (-1 if unavailable).

        Uses ``getSafetyMode()`` from ur_rtde.  Constants:
          1=NORMAL, 2=REDUCED, 3=PROTECTIVE_STOP, 4=RECOVERY,
          5=SAFEGUARD_STOP, 6=SYS_ESTOP, 7=ROBOT_ESTOP,
          8=VIOLATION, 9=FAULT, 12=AUTO_SAFEGUARD_STOP

        NOTE: This intentionally does **not** gate on ``_recv_healthy``.
        During servoJ streaming ``_recv_healthy`` is set to False to
        suppress joint-state publishing, but the recv interface is
        still alive and can read safety registers.  Gating on
        ``_recv_healthy`` would make safety checks blind during
        trajectory execution — exactly when they matter most.
        """
        try:
            if self._recv is not None:
                return self._recv.getSafetyMode()
        except Exception:
            pass
        return -1

    def is_protective_stopped(self) -> bool:
        """Check if the robot is in a protective stop state."""
        try:
            if self._recv is not None and self._recv_healthy:
                return self._recv.isProtectiveStopped()
        except Exception:
            pass
        return False

    def is_safeguard_stopped(self) -> bool:
        """Check if the robot is in a safeguard stop (laser scanner, etc.).

        Safeguard stops auto-clear once the safety zone is vacated,
        unlike protective stops which require manual acknowledgment.
        """
        sm = self.get_safety_mode()
        return sm in (
            self.SAFETY_MODE_SAFEGUARD_STOP,
            self.SAFETY_MODE_AUTO_SAFEGUARD_STOP,
        )

    def is_emergency_stopped(self) -> bool:
        """Check if the robot is in an emergency stop state."""
        try:
            if self._recv is not None and self._recv_healthy:
                return self._recv.isEmergencyStopped()
        except Exception:
            pass
        return False

    def wait_for_safeguard_clear(
        self,
        timeout: float = float('inf'),
        poll_interval: float = 0.25,
        logger=None,
    ) -> bool:
        """Block until the safeguard stop clears or *timeout* expires.

        The default timeout is infinite — the robot waits as long as
        needed for the safety zone to clear.  Pass a finite value
        to cap the wait.

        Returns True if the robot returned to NORMAL/REDUCED mode
        within the timeout, False otherwise.
        """
        log = logger or self.logger
        deadline = _time.monotonic() + timeout
        logged_once = False

        while _time.monotonic() < deadline:
            sm = self.get_safety_mode()
            if sm in (
                self.SAFETY_MODE_NORMAL,
                self.SAFETY_MODE_REDUCED,
            ):
                if log:
                    log.info(
                        f"Safeguard cleared on {self.robot_name} "
                        f"(safety_mode={sm})"
                    )
                return True

            # If it transitioned to a harder fault, bail out
            if sm in (
                self.SAFETY_MODE_PROTECTIVE_STOP,
                self.SAFETY_MODE_SYSTEM_EMERGENCY_STOP,
                self.SAFETY_MODE_ROBOT_EMERGENCY_STOP,
                self.SAFETY_MODE_VIOLATION,
                self.SAFETY_MODE_FAULT,
            ):
                if log:
                    log.error(
                        f"Safety escalated from safeguard to "
                        f"mode {sm} on {self.robot_name} — "
                        f"cannot auto-resume"
                    )
                return False

            if not logged_once and log:
                timeout_str = (
                    f"{timeout:.0f}s" if timeout < 1e9
                    else "indefinitely"
                )
                log.warn(
                    f"Safeguard stop active on {self.robot_name} "
                    f"(safety_mode={sm}) — waiting "
                    f"{timeout_str} for clearance…"
                )
                logged_once = True

            _time.sleep(poll_interval)

        if log:
            log.error(
                f"Safeguard stop did NOT clear on "
                f"{self.robot_name} within {timeout:.0f}s"
            )
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

    def get_actual_tcp_pose(self) -> Optional[List[float]]:
        """Return current TCP pose [x, y, z, rx, ry, rz] or None.

        Position in metres, orientation as axis-angle (rotation vector
        whose magnitude is the angle in radians).
        """
        try:
            if self._recv is not None and self._recv_healthy:
                return list(self._recv.getActualTCPPose())
        except Exception:
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
        gain: int = 600,
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

        # Use moderate lookahead for balance between smooth
        # motion and tight tracking.  0.2 was too high — caused
        # the robot to lag 0.2 s behind the commanded position,
        # leaving 2-5° residual error after servoStop.  0.1 gives
        # tighter tracking at our 15-50 Hz command rate on Jetson.
        effective_lookahead = max(lookahead_time, 0.1)

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
                    # ── Proactive safeguard check ───────────────────
                    # The UR reports robot_mode=7 (RUNNING) and
                    # isProgramRunning()=True during safeguard stop,
                    # so we must check getSafetyMode() directly.
                    if self.is_safeguard_stopped():
                        if logger:
                            logger.warn(
                                f"SAFEGUARD STOP detected on "
                                f"{self.robot_name} at wp "
                                f"{wi}/{n_pts} (proactive check) "
                                f"— pausing servoJ…"
                            )
                        try:
                            self._ctrl.servoStop()
                        except Exception:
                            pass
                        self._teardown_ctrl()
                        self.reconnect_receive()

                        cleared = self.wait_for_safeguard_clear(
                            poll_interval=0.25,
                            logger=logger,
                        )
                        if not cleared:
                            msg = (
                                f"Safeguard did not clear on "
                                f"{self.robot_name} — aborting "
                                f"servoJ"
                            )
                            self.last_error = msg
                            if logger:
                                logger.error(msg)
                            return False

                        _resume_deadline = (
                            _time.monotonic() + 15.0
                        )
                        while _time.monotonic() < _resume_deadline:
                            rm = self.get_robot_mode()
                            if rm == 7:
                                break
                            _time.sleep(0.25)
                        else:
                            msg = (
                                f"Robot mode did not return to "
                                f"RUNNING on {self.robot_name} "
                                f"after safeguard clear (mode="
                                f"{self.get_robot_mode()})"
                            )
                            self.last_error = msg
                            if logger:
                                logger.error(msg)
                            return False

                        if not self._ensure_ctrl():
                            msg = (
                                f"Cannot recreate RTDE control "
                                f"interface on {self.robot_name} "
                                f"after safeguard clear"
                            )
                            self.last_error = msg
                            if logger:
                                logger.error(msg)
                            return False

                        if logger:
                            logger.info(
                                f"Resuming servoJ on "
                                f"{self.robot_name} from wp "
                                f"{wi}/{n_pts} after safeguard"
                            )
                        wall_start = (
                            _time.monotonic() - wp_times[wi]
                        )
                        prev_cmd_time = _time.monotonic()
                        self._recv_healthy = False
                        continue

                    # ── isProgramRunning check ──────────────────────
                    try:
                        if not self._ctrl.isProgramRunning():
                            # ── Identify the stop reason ────────────
                            reason = "PROGRAM STOPPED"
                            is_safeguard = False
                            try:
                                sm = self.get_safety_mode()
                                if sm in (
                                    self.SAFETY_MODE_SAFEGUARD_STOP,
                                    self.SAFETY_MODE_AUTO_SAFEGUARD_STOP,
                                ):
                                    reason = "SAFEGUARD STOP"
                                    is_safeguard = True
                                elif (self._recv is not None
                                        and self._recv.isProtectiveStopped()):
                                    reason = "PROTECTIVE STOP"
                                elif (self._recv is not None
                                        and self._recv.isEmergencyStopped()):
                                    reason = "EMERGENCY STOP"
                            except Exception:
                                pass

                            # ── Safeguard stop → pause & resume ─────
                            if is_safeguard:
                                if logger:
                                    logger.warn(
                                        f"SAFEGUARD STOP on "
                                        f"{self.robot_name} at wp "
                                        f"{wi}/{n_pts} — pausing "
                                        f"servoJ, waiting for "
                                        f"clearance…"
                                    )
                                # Tear down dead control interface
                                try:
                                    self._ctrl.servoStop()
                                except Exception:
                                    pass
                                self._teardown_ctrl()
                                self.reconnect_receive()

                                # Block until safeguard clears
                                cleared = self.wait_for_safeguard_clear(
                                    poll_interval=0.25,
                                    logger=logger,
                                )
                                if not cleared:
                                    msg = (
                                        f"Safeguard did not clear "
                                        f"on {self.robot_name} — "
                                        f"aborting servoJ"
                                    )
                                    self.last_error = msg
                                    if logger:
                                        logger.error(msg)
                                    return False

                                # Wait for robot mode to return to
                                # RUNNING (7) — the UR controller
                                # needs a moment after safeguard
                                # clearance before accepting commands.
                                _resume_deadline = (
                                    _time.monotonic() + 15.0
                                )
                                while _time.monotonic() < _resume_deadline:
                                    rm = self.get_robot_mode()
                                    if rm == 7:
                                        break
                                    _time.sleep(0.25)
                                else:
                                    msg = (
                                        f"Robot mode did not return "
                                        f"to RUNNING on "
                                        f"{self.robot_name} after "
                                        f"safeguard clear (mode="
                                        f"{self.get_robot_mode()})"
                                    )
                                    self.last_error = msg
                                    if logger:
                                        logger.error(msg)
                                    return False

                                # Re-create control interface
                                if not self._ensure_ctrl():
                                    msg = (
                                        f"Cannot recreate RTDE "
                                        f"control interface on "
                                        f"{self.robot_name} after "
                                        f"safeguard clear"
                                    )
                                    self.last_error = msg
                                    if logger:
                                        logger.error(msg)
                                    return False

                                if logger:
                                    logger.info(
                                        f"Resuming servoJ on "
                                        f"{self.robot_name} from "
                                        f"wp {wi}/{n_pts}"
                                    )

                                # Reset timing so rate-pacing doesn't
                                # try to "catch up" the paused time.
                                wall_start = (
                                    _time.monotonic() - wp_times[wi]
                                )
                                prev_cmd_time = _time.monotonic()
                                # Mark recv unhealthy again — servoJ
                                # owns position publishing.
                                self._recv_healthy = False
                                continue  # retry this waypoint

                            # ── Non-safeguard stop → abort ──────────
                            msg = (
                                f"{reason} on {self.robot_name} "
                                f"at wp {wi}/{n_pts} — aborting "
                                f"servoJ"
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
                    # servoJ threw — check if a safeguard stop
                    if self.is_safeguard_stopped():
                        if logger:
                            logger.warn(
                                f"servoJ exception on "
                                f"{self.robot_name} at wp "
                                f"{wi}/{n_pts} due to SAFEGUARD "
                                f"STOP — pausing…"
                            )
                        try:
                            self._ctrl.servoStop()
                        except Exception:
                            pass
                        self._teardown_ctrl()
                        self.reconnect_receive()

                        cleared = self.wait_for_safeguard_clear(
                            logger=logger,
                        )
                        if not cleared:
                            self.last_error = (
                                f"Safeguard did not clear on "
                                f"{self.robot_name}"
                            )
                            return False

                        _rd = _time.monotonic() + 15.0
                        while _time.monotonic() < _rd:
                            if self.get_robot_mode() == 7:
                                break
                            _time.sleep(0.25)
                        else:
                            self.last_error = (
                                f"Robot mode not RUNNING after "
                                f"safeguard clear"
                            )
                            return False

                        if not self._ensure_ctrl():
                            self.last_error = (
                                f"Cannot recreate RTDE ctrl "
                                f"after safeguard clear"
                            )
                            return False

                        if logger:
                            logger.info(
                                f"Resuming servoJ on "
                                f"{self.robot_name} from "
                                f"wp {wi}/{n_pts}"
                            )
                        wall_start = (
                            _time.monotonic() - wp_times[wi]
                        )
                        prev_cmd_time = _time.monotonic()
                        self._recv_healthy = False
                        continue  # retry this waypoint

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
                    # Check if this is a safeguard stop (auto-clears)
                    if self.is_safeguard_stopped():
                        if logger:
                            logger.warn(
                                f"servoJ returned False on "
                                f"{self.robot_name} at wp "
                                f"{wi}/{n_pts} — SAFEGUARD STOP "
                                f"detected, pausing…"
                            )
                        try:
                            self._ctrl.servoStop()
                        except Exception:
                            pass
                        self._teardown_ctrl()
                        self.reconnect_receive()

                        cleared = self.wait_for_safeguard_clear(
                            logger=logger,
                        )
                        if not cleared:
                            msg = (
                                f"Safeguard did not clear on "
                                f"{self.robot_name} — aborting"
                            )
                            self.last_error = msg
                            if logger:
                                logger.error(msg)
                            return False

                        # Wait for RUNNING mode
                        _rd = _time.monotonic() + 15.0
                        while _time.monotonic() < _rd:
                            if self.get_robot_mode() == 7:
                                break
                            _time.sleep(0.25)
                        else:
                            msg = (
                                f"Robot mode not RUNNING after "
                                f"safeguard clear on "
                                f"{self.robot_name}"
                            )
                            self.last_error = msg
                            if logger:
                                logger.error(msg)
                            return False

                        if not self._ensure_ctrl():
                            msg = (
                                f"Cannot recreate RTDE ctrl on "
                                f"{self.robot_name} after "
                                f"safeguard clear"
                            )
                            self.last_error = msg
                            if logger:
                                logger.error(msg)
                            return False

                        if logger:
                            logger.info(
                                f"Resuming servoJ on "
                                f"{self.robot_name} from "
                                f"wp {wi}/{n_pts}"
                            )
                        wall_start = (
                            _time.monotonic() - wp_times[wi]
                        )
                        prev_cmd_time = _time.monotonic()
                        self._recv_healthy = False
                        continue  # retry this waypoint

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

            # ── Precision finish: moveJ to final target ────────
            # servoStop() decelerates the robot from its current
            # position — it does NOT drive to the last commanded
            # waypoint.  With lookahead smoothing the robot is
            # always slightly behind, leaving 0.5-5° residual
            # error.  A short moveJ to the exact final position
            # guarantees convergence to the UR's internal
            # tolerance (~0.01°).  moveJ is synchronous (blocks
            # until complete), typically <1 s for the small
            # residual distance.
            try:
                self._ctrl.servoStop()
            except Exception:
                pass
            _time.sleep(0.05)  # brief settle after servo stop

            final_q = list(positions[-1])
            try:
                # moveJ: speed 0.5 rad/s, accel 1.0 rad/s²
                # synchronous=False so we can monitor + timeout
                movej_ok = self._ctrl.moveJ(
                    final_q, 0.5, 1.0, False,
                )
                if logger:
                    logger.info(
                        f"moveJ precision finish for "
                        f"{self.robot_name} — "
                        f"{'OK' if movej_ok else 'FAILED'}"
                    )
            except Exception as mj_exc:
                if logger:
                    logger.warn(
                        f"moveJ precision finish failed on "
                        f"{self.robot_name}: {mj_exc} — "
                        f"continuing with servoJ final position"
                    )

            self._teardown_ctrl()
            self.reconnect_receive()
            self.last_error = ""

            # ── Position verification ───────────────────────────────
            # Read the robot's actual joint positions after trajectory
            # completion and compare to the planned final waypoint.
            # This detects positional error from servoJ dynamics
            # (lookahead smoothing, deceleration, timing).
            target_final = list(positions[-1])
            self.last_target_q = target_final
            self.last_actual_q_post_exec = None
            self.last_position_error_rad = None

            actual_q = self.get_actual_q()
            if actual_q is not None:
                self.last_actual_q_post_exec = actual_q
                errors = [actual_q[j] - target_final[j]
                          for j in range(len(target_final))]
                self.last_position_error_rad = errors
                rss_deg = (
                    sum(e ** 2 for e in errors) ** 0.5
                ) * 180.0 / 3.141592653589793
                max_err_deg = max(
                    abs(e) for e in errors
                ) * 180.0 / 3.141592653589793
                if logger:
                    err_mdeg = [
                        round(e * 180000.0 / 3.141592653589793, 1)
                        for e in errors
                    ]
                    logger.info(
                        f"Position verification [{self.robot_name}]: "
                        f"error per joint (mDeg): {err_mdeg}  "
                        f"RSS={rss_deg*1000:.1f}mDeg  "
                        f"max={max_err_deg*1000:.1f}mDeg"
                    )
                    if rss_deg > 0.5:
                        logger.warn(
                            f"⚠ LARGE position error on "
                            f"{self.robot_name}: {rss_deg:.2f}° — "
                            f"jumps expected at next trajectory start"
                        )
            elif logger:
                logger.warn(
                    f"Could not read actual position on "
                    f"{self.robot_name} after trajectory — "
                    f"recv interface not healthy"
                )

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
