"""
device/relay.py
---------------
Machine enable relay (LED for now) on a Raspberry Pi GPIO pin.

  * OFF while hmi.py runs, until a face is authenticated.
  * ON after ACCESS GRANTED.
  * Stays ON until hmi.py stops (or for RELAY_ON_SECONDS if that is > 0).

Active-low relay boards (RELAY_ACTIVE_HIGH = False) powered from 5V do not switch
OFF when the Pi drives the pin HIGH (3.3V is not enough). So for those boards:
  ON  = drive the pin LOW
  OFF = release the pin (input / high impedance) - the board's pull-up switches it off,
        the same state the pin is in when no program is using it.

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
        self.pin = pin
        self.active_high = active_high
        self.on_seconds = on_seconds
        self.is_on = False
        self._timer = None
        self._lock = threading.Lock()
        self._device = None       # only exists while the pin is driven
        self.simulated = OutputDevice is None
        if self.simulated:
            print("[Relay] gpiozero not available - running in simulation mode")
        elif active_high:
            # Active-high board: drive the pin LOW for OFF the whole time
            self._open(initial_on=False)
        print(f"[Relay] Ready on GPIO{pin} (OFF)")

    def _open(self, initial_on):
        try:
            self._device = OutputDevice(self.pin, active_high=self.active_high, initial_value=initial_on)
        except Exception as e:
            print(f"[Relay] Could not open GPIO{self.pin}: {e} - running in simulation mode")
            self.simulated = True
            self._device = None

    def _release(self):
        if self._device is not None:
            self._device.close()   # pin back to input (high impedance)
            self._device = None

    def _set(self, on):
        if not self.simulated:
            if self.active_high:
                if self._device is None:
                    self._open(initial_on=on)
                elif on:
                    self._device.on()
                else:
                    self._device.off()
            elif on:
                if self._device is None:
                    self._open(initial_on=True)   # drive LOW -> relay ON
                else:
                    self._device.on()
            else:
                self._release()                   # release pin -> relay OFF
        self.is_on = on
        print(f"[Relay] {'ON' if on else 'OFF'}")

    def _cancel_timer(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def grant(self):
        """Authentication succeeded: switch ON (and schedule OFF only if a duration is set)."""
        with self._lock:
            self._cancel_timer()
            if not self.is_on:
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
        """Called when hmi.py stops."""
        self.off()
        self._release()


if __name__ == "__main__":
    # Wiring test: OFF 2 s, ON 2 s, OFF
    import time
    relay = MachineRelay(on_seconds=0)
    print("OFF for 2 s..."); time.sleep(2)
    relay.grant()
    print("ON for 2 s...");  time.sleep(2)
    relay.close()
