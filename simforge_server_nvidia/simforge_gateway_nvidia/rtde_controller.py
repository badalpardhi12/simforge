"""
Direct RTDE Robot Controller.

Bypasses the ROS2 control stack (UR driver, ros2_control,
scaled_joint_trajectory_controller) and communicates directly with the
UR controller via the RTDE protocol.

Key methods:
  - move_j()                    — blocking joint move (for homing)
  - execute_trajectory_servoj() — 500 Hz streaming for smooth trajectories
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

    # RTDE servo rate and period
    _frequency = 500.0
    _dt = 1.0 / _frequency

    def __init__(self, robot_name: str, ip: str, logger=None):
        self.robot_name = robot_name
        self.ip = ip
        self.logger = logger
        self._ctrl = None
        self._recv = None
        self._connected = False
        self._recv_healthy = False  # tracks receive interface health

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
            self._ctrl = rtde_control.RTDEControlInterface(self.ip)
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
        method creates a fresh RTDEReceiveInterface to restore the
        50 Hz joint-state reads.
        """
        if not RTDE_AVAILABLE:
            return False

        if self.logger:
            self.logger.info(
                f"Reconnecting RTDE receive interface for "
                f"{self.robot_name}..."
            )

        result_holder = [False]

        def _do_reconnect():
            try:
                # Try reconnecting existing object first
                if self._recv is not None:
                    try:
                        self._recv.reconnect()
                        result_holder[0] = True
                        return
                    except Exception:
                        pass

                # Create a brand-new receive interface
                self._recv = rtde_receive.RTDEReceiveInterface(self.ip)
                result_holder[0] = True
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
            and self._ctrl is not None
            and self._recv is not None
        )

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
            # Ensure a working control interface.  After servoJ the
            # old one is often dead and reconnect() alone can't
            # re-upload the UR control script, so build a fresh one.
            if not self._ctrl or not self._ctrl.isConnected():
                if self.logger:
                    self.logger.info(
                        f"RTDE control interface dead for "
                        f"{self.robot_name} — creating fresh connection"
                    )
                try:
                    if self._ctrl:
                        self._ctrl.disconnect()
                except Exception:
                    pass
                self._ctrl = rtde_control.RTDEControlInterface(self.ip)

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
                return False

            if exc_holder[0]:
                raise exc_holder[0]

            # moveJ finished — disconnect the control interface to
            # prevent ur_rtde's internal auto-reconnect from looping
            # (and eventually segfaulting) while the UR control script
            # is stopped.  A fresh interface will be created on demand.
            try:
                self._ctrl.disconnect()
            except Exception:
                pass
            self._ctrl = None

            return result_holder[0]

        except Exception as e:
            if self.logger:
                self.logger.error(f"RTDE moveJ failed: {e}")
            # Also tear down the control interface on error
            try:
                if self._ctrl:
                    self._ctrl.disconnect()
            except Exception:
                pass
            self._ctrl = None
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
        """Stream a trajectory using servoJ at the robot's native rate.

        The trajectory is upsampled via linear interpolation to the RTDE
        servo rate (500 Hz).  When *timestamps* is provided the
        interpolation respects per-waypoint timing (including dwell
        pauses); otherwise a uniform *dt* is assumed.

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
            Called every ~20 ms (50 Hz) with the interpolated joint
            position currently being commanded.  Used by the joint-state
            manager to publish ``/joint_states`` in real time while the
            RTDE receive interface is unavailable.
        """
        if not self.is_connected:
            if logger:
                logger.error(f"RTDE not connected to {self.robot_name}")
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

        servo_dt = self._dt
        total_time = wp_times[-1]
        n_servo = int(total_time / servo_dt) + 1

        # Mark recv unhealthy immediately so the 50 Hz RTDE timer
        # does not try to read (and publish stale data) for this
        # robot while the position_callback owns publishing.
        self._recv_healthy = False

        if logger:
            logger.info(
                f"RTDE servoJ streaming {self.robot_name}: "
                f"{n_pts} waypoints → "
                f"{n_servo} servo commands @ {self._frequency:.0f}Hz, "
                f"duration={total_time:.2f}s"
            )

        try:
            seg_idx = 0

            # Publish every 10th iteration → 50 Hz callback rate
            cb_interval = max(1, int(self._frequency / 50.0))  # 10
            cb_logged = False  # log first callback only

            for si in range(n_servo):
                if stop_check_fn and stop_check_fn():
                    if logger:
                        logger.info("RTDE servoJ aborted by stop request")
                    self._ctrl.servoStop()
                    return False

                t = si * servo_dt

                # Advance segment cursor
                while seg_idx < n_pts - 2 and wp_times[seg_idx + 1] <= t:
                    seg_idx += 1

                idx0 = seg_idx
                idx1 = min(idx0 + 1, n_pts - 1)

                seg_dur = wp_times[idx1] - wp_times[idx0]
                if seg_dur > 0:
                    alpha = max(0.0, min(1.0,
                        (t - wp_times[idx0]) / seg_dur))
                else:
                    alpha = 1.0

                q_target = [
                    positions[idx0][j] * (1.0 - alpha)
                    + positions[idx1][j] * alpha
                    for j in range(6)
                ]

                # Guard against RTDE control interface dropping mid-stream.
                # ur_rtde's internal auto-reconnect can race and segfault
                # if we keep calling servoJ after the link is lost.
                if not self._ctrl.isConnected():
                    if logger:
                        logger.warn(
                            f"RTDE control interface lost during servoJ "
                            f"on {self.robot_name} at cmd {si}/{n_servo}"
                        )
                    # Publish the last target so the sim holds position
                    if position_callback:
                        try:
                            position_callback(self.robot_name, q_target)
                        except Exception:
                            pass
                    return False

                try:
                    t_start = self._ctrl.initPeriod()
                    self._ctrl.servoJ(
                        q_target, 0.0, 0.0, servo_dt,
                        lookahead_time, gain,
                    )
                    self._ctrl.waitPeriod(t_start)
                except Exception as servo_exc:
                    if logger:
                        logger.error(
                            f"RTDE servoJ exception on "
                            f"{self.robot_name} at cmd {si}/{n_servo}: "
                            f"{servo_exc}"
                        )
                    if position_callback:
                        try:
                            position_callback(self.robot_name, q_target)
                        except Exception:
                            pass
                    return False

                # Publish commanded position at ~50 Hz
                if position_callback and si % cb_interval == 0:
                    try:
                        position_callback(self.robot_name, q_target)
                        if not cb_logged and logger:
                            logger.info(
                                f"servoJ position callback active for "
                                f"{self.robot_name} (publishing at "
                                f"~{self._frequency / cb_interval:.0f} Hz)"
                            )
                            cb_logged = True
                    except Exception:
                        pass  # never let callback errors kill the loop

            # Clean stop — control interface may already be dead
            try:
                self._ctrl.servoStop()
            except Exception:
                pass
            _time.sleep(0.5)

            # Stop the UR control script explicitly before disconnecting.
            # This prevents ur_rtde's internal auto-reconnect from
            # trying to re-upload the script and racing into a segfault.
            try:
                self._ctrl.stopScript()
            except Exception:
                pass

            # Tear down the control interface — after servoStop the UR
            # control script is no longer running, and ur_rtde's
            # internal auto-reconnect will loop and can segfault.
            # A fresh interface will be created on demand by move_j().
            try:
                self._ctrl.disconnect()
            except Exception:
                pass
            self._ctrl = None

            # Reconnect receive interface (it dies during servoJ)
            self.reconnect_receive()

            if logger:
                logger.info(
                    f"RTDE servoJ complete for {self.robot_name} ✓"
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
            try:
                self._ctrl.stopScript()
            except Exception:
                pass
            # Tear down control interface to prevent auto-reconnect loop
            try:
                if self._ctrl:
                    self._ctrl.disconnect()
            except Exception:
                pass
            self._ctrl = None
            # Still try to restore receive interface
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
