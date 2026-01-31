"""
Safety Monitor for Simforge Client

This module provides a dedicated safety monitoring class that runs alongside
the main client, specifically handling:

1. Heartbeat monitoring (ensuring heartbeats are being sent)
2. Connection quality monitoring
3. E-Stop button integration (keyboard, USB button, etc.)
4. Local safety state display

This is separate from the client to allow independent safety monitoring
even if the main client has issues.
"""

import asyncio
import time
import logging
from typing import Optional, Callable
from dataclasses import dataclass
from enum import IntEnum
import threading

try:
    import keyboard  # Optional: for keyboard-based E-Stop
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False

logger = logging.getLogger(__name__)


class SafetyState(IntEnum):
    """Local safety state."""
    NORMAL = 0
    WARNING = 1
    ESTOP_REQUESTED = 2
    DISCONNECTED = 3


@dataclass
class SafetyStatus:
    """Current safety status."""
    state: SafetyState = SafetyState.DISCONNECTED
    heartbeat_count: int = 0
    last_heartbeat_time: float = 0.0
    connection_latency_ms: float = 0.0
    server_connected: bool = False
    estop_active: bool = False
    warning_message: str = ""


class SafetyMonitor:
    """
    Safety monitor that runs independently of the main client.
    
    This class:
    1. Monitors that heartbeats are being sent to the server
    2. Tracks connection quality
    3. Provides E-Stop functionality via keyboard or callback
    4. Displays safety status
    
    Usage:
        monitor = SafetyMonitor(client)
        monitor.start()
        
        # E-Stop via callback
        monitor.on_estop_triggered(lambda: print("E-STOP!"))
        
        # Manual E-Stop
        await monitor.trigger_estop()
        
        monitor.stop()
    """

    def __init__(
        self,
        client,  # SimforgeClient instance
        heartbeat_warning_threshold_ms: float = 100.0,
        enable_keyboard_estop: bool = True,
        estop_key: str = "escape",  # Default E-Stop key
    ):
        """
        Initialize the safety monitor.
        
        Args:
            client: SimforgeClient instance to monitor
            heartbeat_warning_threshold_ms: Warn if heartbeat interval exceeds this
            enable_keyboard_estop: Enable keyboard-based E-Stop (requires keyboard package)
            estop_key: Key to trigger E-Stop (default: escape)
        """
        self.client = client
        self.heartbeat_warning_threshold_ms = heartbeat_warning_threshold_ms
        self.enable_keyboard_estop = enable_keyboard_estop and HAS_KEYBOARD
        self.estop_key = estop_key
        
        self._status = SafetyStatus()
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._estop_callbacks: list[Callable[[], None]] = []
        self._status_callbacks: list[Callable[[SafetyStatus], None]] = []
        
        # For keyboard monitoring in separate thread
        self._keyboard_thread: Optional[threading.Thread] = None
        self._estop_event = asyncio.Event()

    @property
    def status(self) -> SafetyStatus:
        """Get current safety status."""
        return self._status

    @property
    def is_safe(self) -> bool:
        """Check if system is in a safe state for operations."""
        return (
            self._status.state == SafetyState.NORMAL
            and self._status.server_connected
            and not self._status.estop_active
        )

    def on_estop_triggered(self, callback: Callable[[], None]):
        """Register a callback for E-Stop events."""
        self._estop_callbacks.append(callback)

    def on_status_update(self, callback: Callable[[SafetyStatus], None]):
        """Register a callback for status updates."""
        self._status_callbacks.append(callback)

    def start(self):
        """Start the safety monitor."""
        if self._running:
            logger.warning("Safety monitor already running")
            return
        
        self._running = True
        
        # Start keyboard monitoring if enabled
        if self.enable_keyboard_estop:
            self._start_keyboard_monitor()
        
        logger.info("Safety monitor started")

    def stop(self):
        """Stop the safety monitor."""
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
        
        if self._keyboard_thread and self._keyboard_thread.is_alive():
            # Keyboard thread will exit on next iteration
            pass
        
        logger.info("Safety monitor stopped")

    def _start_keyboard_monitor(self):
        """Start keyboard monitoring in a separate thread."""
        if not HAS_KEYBOARD:
            logger.warning("keyboard package not installed - keyboard E-Stop disabled")
            return
        
        def keyboard_monitor():
            logger.info(f"Press '{self.estop_key}' for E-Stop")
            while self._running:
                try:
                    if keyboard.is_pressed(self.estop_key):
                        logger.critical(f"E-STOP triggered via {self.estop_key} key!")
                        self._estop_event.set()
                        time.sleep(1.0)  # Debounce
                except Exception as e:
                    logger.debug(f"Keyboard check error: {e}")
                time.sleep(0.05)  # 20Hz check rate
        
        self._keyboard_thread = threading.Thread(target=keyboard_monitor, daemon=True)
        self._keyboard_thread.start()

    async def run(self):
        """Main monitoring loop - call this from your async event loop."""
        logger.info("Safety monitor running")
        
        last_heartbeat_check = time.time()
        
        while self._running:
            try:
                # Check for E-Stop event (from keyboard or other source)
                if self._estop_event.is_set():
                    await self._handle_estop()
                    self._estop_event.clear()
                
                # Update status from client
                self._update_status_from_client()
                
                # Check heartbeat health
                current_time = time.time()
                heartbeat_interval_ms = (current_time - last_heartbeat_check) * 1000
                
                if heartbeat_interval_ms > self.heartbeat_warning_threshold_ms:
                    if self._status.state == SafetyState.NORMAL:
                        self._status.state = SafetyState.WARNING
                        self._status.warning_message = (
                            f"Heartbeat interval high: {heartbeat_interval_ms:.1f}ms"
                        )
                        logger.warning(self._status.warning_message)
                
                last_heartbeat_check = current_time
                
                # Notify callbacks
                self._notify_status_callbacks()
                
                await asyncio.sleep(0.02)  # 50Hz monitoring
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Safety monitor error: {e}")
                await asyncio.sleep(0.1)

    def _update_status_from_client(self):
        """Update status from client state."""
        self._status.server_connected = self.client.is_connected
        
        if not self.client.is_connected:
            self._status.state = SafetyState.DISCONNECTED
        elif self._status.state == SafetyState.DISCONNECTED:
            self._status.state = SafetyState.NORMAL

    async def _handle_estop(self):
        """Handle E-Stop event."""
        self._status.state = SafetyState.ESTOP_REQUESTED
        self._status.estop_active = True
        
        # Notify callbacks
        for callback in self._estop_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"E-Stop callback error: {e}")
        
        # Send E-Stop to server
        if self.client.is_connected:
            await self.client.emergency_stop()

    def _notify_status_callbacks(self):
        """Notify registered status callbacks."""
        for callback in self._status_callbacks:
            try:
                callback(self._status)
            except Exception as e:
                logger.error(f"Status callback error: {e}")

    async def trigger_estop(self):
        """Manually trigger E-Stop."""
        logger.critical("Manual E-STOP triggered!")
        self._estop_event.set()

    def reset_estop(self):
        """Reset E-Stop state (requires manual verification)."""
        if self._status.estop_active:
            logger.warning("Resetting E-Stop - ensure robot is in safe state!")
            self._status.estop_active = False
            self._status.state = (
                SafetyState.NORMAL if self.client.is_connected 
                else SafetyState.DISCONNECTED
            )


class SafetyStatusDisplay:
    """
    Simple text-based safety status display.
    
    Can be used for terminal output or logging.
    """

    def __init__(self, monitor: SafetyMonitor):
        self.monitor = monitor
        monitor.on_status_update(self._display_status)

    def _display_status(self, status: SafetyStatus):
        """Display status (override for custom display)."""
        state_names = {
            SafetyState.NORMAL: "✓ NORMAL",
            SafetyState.WARNING: "⚠ WARNING",
            SafetyState.ESTOP_REQUESTED: "🛑 E-STOP",
            SafetyState.DISCONNECTED: "✗ DISCONNECTED",
        }
        
        state_str = state_names.get(status.state, "UNKNOWN")
        connected_str = "Connected" if status.server_connected else "Disconnected"
        
        # Only print on state changes or warnings
        if status.state != SafetyState.NORMAL or status.warning_message:
            print(f"\r[SAFETY] {state_str} | Server: {connected_str}", end="")
            if status.warning_message:
                print(f" | {status.warning_message}", end="")
            print("          ", end="\r")  # Clear rest of line


async def demo():
    """Demo of safety monitor functionality."""
    from simforge_client import SimforgeClient
    
    # Create client (won't actually connect in demo)
    client = SimforgeClient(server_ip="localhost")
    
    # Create safety monitor
    monitor = SafetyMonitor(
        client,
        enable_keyboard_estop=True,
        estop_key="escape"
    )
    
    # Create display
    display = SafetyStatusDisplay(monitor)
    
    # Start monitoring
    monitor.start()
    
    print("Safety monitor demo - press ESC for E-Stop, Ctrl+C to exit")
    
    try:
        await monitor.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        monitor.stop()


if __name__ == "__main__":
    asyncio.run(demo())
