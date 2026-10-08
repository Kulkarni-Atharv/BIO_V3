"""
device/relay.py
---------------
Machine enable relay (LED for now) on a Raspberry Pi GPIO pin.

  * OFF at start-up and on exit.
  * ON after a successful face authentication.
  * Turns OFF again after RELAY_ON_SECONDS (0 = stay ON until the next scan starts).

On a PC without GPIO it only prints what it would do, so hmi.py still runs there.

Test on the Pi:  python3 device/relay.py
"""

import os
import sys
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.config import RELAY_GPIO_PIN, RELAY_ACTIVE_HIGH, RELAY_ON_SECONDS

try:
    from gpiozero import OutputDevice
except ImportError:
    OutputDevice = None


class MachineRelay:
    def __init__(self, pin=RELAY_GPIO_PIN, active_high=RELAY_ACTIVE_HIGH, on_seconds=RELAY_ON_SECONDS):
        self.on_seconds = on_seconds
        self._timer = None
        self._lock = threading.Lock()
        self._device = None
        if OutputDevice is None:
            print("[Relay] gpiozero not available - running in simulation mode")
            return
        try:
            self._device = OutputDevice(pin, active_high=active_high, initial_value=False)
            print(f"[Relay] Ready on GPIO{pin} (OFF)")
        except Exception as e:
            print(f"[Relay] Could not open GPIO{pin}: {e} - running in simulation mode")

    def _set(self, on):
        if self._device is not None:
            self._device.on() if on else self._device.off()
        print(f"[Relay] {'ON' if on else 'OFF'}")

    def _cancel_timer(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def grant(self):
        """Authentication succeeded: switch ON (and schedule OFF if a duration is set)."""
        with self._lock:
            self._cancel_timer()
            self._set(True)
            if self.on_seconds > 0:
                self._timer = threading.Timer(self.on_seconds, self.off)
                self._timer.daemon = True
                self._timer.start()

    def off(self):
        with self._lock:
            self._cancel_timer()
            self._set(False)

    def close(self):
        self.off()
        if self._device is not None:
            self._device.close()
            self._device = None


if __name__ == "__main__":
    # Quick hardware test: ON for 2 s, then OFF
    import time
    relay = MachineRelay(on_seconds=0)
    relay.grant()
    time.sleep(2)
    relay.close()
