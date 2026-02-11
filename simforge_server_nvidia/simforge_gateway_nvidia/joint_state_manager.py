"""
Joint State Manager — mode-aware joint state publishing.

Handles the dual-source problem:
  • **Simulation mode** → sim's ``joint_state_broadcaster`` publishes to
    ``/joint_states`` at ~625 Hz (fake hardware).  This module does nothing
    extra.
  • **Real mode** → The sim's ``joint_state_broadcaster`` is deactivated.
    A 50 Hz timer reads real joint positions from RTDE and publishes them
    to ``/joint_states``, which drives ``robot_state_publisher`` → TF →
    Foxglove.

The ``sim FollowJointTrajectory`` action is **not** used for visualisation
in real mode; the real joint positions are the single source of truth so
the Foxglove model always matches the physical robot.
"""

import subprocess
import time
from typing import Dict, Optional

from sensor_msgs.msg import JointState

from .config import ROBOT_CONFIG, RobotStateInfo, RTDE_AVAILABLE


class JointStateManager:
    """Manages joint state publishing and the joint_state_broadcaster."""

    def __init__(self, node):
        """
        Parameters
        ----------
        node : rclpy.node.Node
            The parent ROS2 node (for create_publisher / create_timer /
            get_clock / get_logger).
        """
        self._node = node
        self._log = node.get_logger()

        # Caches — shared with the gateway node
        self._robot_states: Dict[str, RobotStateInfo] = {
            n: RobotStateInfo() for n in ROBOT_CONFIG
        }

        # RTDE controllers (set by the gateway after construction)
        self._rtde_controllers: Dict = {}

        # Internal state
        self._js_pub = None
        self._js_timer = None
        self._jsb_deactivated = False
        self._real_mode = False  # when True, ignore /joint_states from sim

    # ── Properties ───────────────────────────────────────────────

    @property
    def robot_states(self) -> Dict[str, RobotStateInfo]:
        return self._robot_states

    # ── /joint_states subscriber callback ────────────────────────

    def on_joint_states(self, msg: JointState) -> None:
        """Called by the node's /joint_states subscriber.

        In real mode the RTDE publisher and position_callback are the
        sole source of truth — they update the cache directly in
        ``_publish_rtde_joint_states`` and ``publish_positions``.
        Any messages arriving via the subscriber (e.g. from the sim's
        latched ``joint_state_broadcaster`` or stale DDS data) are
        ignored to prevent the cache from flipping between real and
        sim positions.
        """
        if self._real_mode:
            return  # RTDE/position_callback owns the cache
        for robot_name, cfg in ROBOT_CONFIG.items():
            positions, velocities = [], []
            for jn in cfg["joints"]:
                if jn in msg.name:
                    idx = msg.name.index(jn)
                    positions.append(msg.position[idx])
                    velocities.append(
                        msg.velocity[idx]
                        if idx < len(msg.velocity) else 0.0
                    )
            if len(positions) == 6:
                st = self._robot_states[robot_name]
                st.joint_positions = positions
                st.joint_velocities = velocities
                st.last_update = time.time()

    # ── RTDE → /joint_states publisher (real mode) ───────────────

    def start_rtde_publisher(self, rtde_controllers=None) -> None:
        """Start a 50 Hz timer that publishes real joint states.

        Parameters
        ----------
        rtde_controllers : dict, optional
            Map of robot_name → URRTDEController.  If given, updates
            the internal reference used by the timer callback.
        """
        if rtde_controllers is not None:
            self._rtde_controllers = rtde_controllers
        if self._js_timer is not None:
            return  # already running

        self._real_mode = True  # ignore sim /joint_states messages

        self._js_pub = self._node.create_publisher(
            JointState, "/joint_states", 10
        )
        self._js_timer = self._node.create_timer(
            0.02, self._publish_rtde_joint_states
        )
        self._log.info("RTDE → /joint_states publisher started (50 Hz)")

    def stop_rtde_publisher(self) -> None:
        """Stop the RTDE joint state publisher."""
        self._real_mode = False  # allow sim /joint_states again
        if self._js_timer is not None:
            self._js_timer.cancel()
            self._js_timer = None
        if self._js_pub is not None:
            self._node.destroy_publisher(self._js_pub)
            self._js_pub = None
        self._log.info("RTDE → /joint_states publisher stopped")

    def _publish_rtde_joint_states(self) -> None:
        """Timer callback — read from RTDE, publish to /joint_states."""
        for robot_name, rtde in self._rtde_controllers.items():
            if not rtde._recv:
                continue
            actual_q = rtde.get_actual_q()
            actual_qd = rtde.get_actual_qd()
            if actual_q is None:
                # Receive interface is dead.  Skip for now — the
                # explicit reconnect_receive() calls after moveJ
                # and servoJ will restore it.
                continue

            cfg = ROBOT_CONFIG[robot_name]
            msg = JointState()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.name = list(cfg["joints"])
            msg.position = [float(v) for v in actual_q]
            if actual_qd is not None:
                msg.velocity = [float(v) for v in actual_qd]
            self._js_pub.publish(msg)

            # Also update the cache
            st = self._robot_states.get(robot_name)
            if st:
                st.joint_positions = list(actual_q)
                if actual_qd:
                    st.joint_velocities = list(actual_qd)
                st.last_update = time.time()

    # ── Direct publish (called from servoJ loop) ─────────────────

    def publish_positions(
        self, robot_name: str, positions: list
    ) -> None:
        """Publish joint positions directly to /joint_states.

        Called from the servoJ loop at ~50 Hz when the RTDE receive
        interface is unavailable.  This keeps the sim model tracking
        the real robot in Foxglove during streaming.
        """
        if self._js_pub is None:
            return

        cfg = ROBOT_CONFIG.get(robot_name)
        if cfg is None:
            return

        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = list(cfg["joints"])
        msg.position = [float(v) for v in positions]
        self._js_pub.publish(msg)

        # Update the cache too
        st = self._robot_states.get(robot_name)
        if st:
            st.joint_positions = list(positions)
            st.last_update = time.time()

    # ── Sim joint_state_broadcaster control ──────────────────────

    def set_sim_broadcaster(self, active: bool) -> None:
        """Activate / deactivate the sim stack's joint_state_broadcaster
        and trajectory controllers.

        In real mode the sim's broadcaster floods ``/joint_states``
        at ~625 Hz with stale values, drowning out the RTDE publisher's
        real readings.  Deactivating it lets the RTDE publisher own
        ``/joint_states`` so Foxglove reflects the real robot.

        The trajectory controllers are also deactivated in real mode to
        prevent the fake hardware from accepting commands that could
        leak back through latched publishers.
        """
        if active and not self._jsb_deactivated:
            return
        if not active and self._jsb_deactivated:
            return

        action = "activate" if active else "deactivate"
        flag = "--activate" if active else "--deactivate"

        # Build controller list: broadcaster + all trajectory controllers
        controllers = ["joint_state_broadcaster"]
        for cfg in ROBOT_CONFIG.values():
            ctrl = cfg.get("controller")
            if ctrl:
                controllers.append(ctrl)

        try:
            # IMPORTANT: ros2 control switch_controllers uses argparse
            # nargs='*' for --deactivate/--activate, so repeated flags
            # like --deactivate A --deactivate B only processes B.
            # Must use ONE flag with space-separated controller names:
            #   --deactivate A B C
            cmd = [
                "ros2", "control", "switch_controllers",
                flag,
            ] + controllers
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10.0,
            )
            if result.returncode == 0:
                self._jsb_deactivated = not active
                self._log.info(
                    f"Sim controllers {action}d "
                    f"({', '.join(controllers)}) — "
                    f"RTDE now {'secondary' if active else 'primary'} "
                    f"on /joint_states"
                )
            else:
                self._log.warn(
                    f"Failed to {action} sim controllers: "
                    f"{result.stderr.strip()}"
                )
        except subprocess.TimeoutExpired:
            self._log.warn(
                f"Timeout trying to {action} sim controllers"
            )
        except Exception as e:
            self._log.warn(
                f"Error switching joint_state_broadcaster: {e}"
            )
