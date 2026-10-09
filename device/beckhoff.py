"""
device/beckhoff.py
------------------
Machine enable output through a Beckhoff controller (CX7000) over ADS (pyads).

Same interface as device/relay.py MachineRelay, so hmi.py can use either:
  grant()  -> write BECKHOFF_VARIABLE = TRUE   (machine / LED ON)
  off()    -> write BECKHOFF_VARIABLE = FALSE  (machine / LED OFF)
  close()  -> write FALSE and close the connection (hmi.py stops)

grant() returns False if the controller could not be reached, so the HMI can
tell the operator the machine did not start. The connection is retried on the
next command if it was lost.

Interactive test on the Pi:  python3 device/beckhoff.py
"""

import os
import sys
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.config import BECKHOFF_AMS_NET_ID, BECKHOFF_PLC_IP, BECKHOFF_VARIABLE, BECKHOFF_TIMEOUT_MS

try:
    import pyads
except ImportError:
    pyads = None


class BeckhoffOutput:
    name = "Beckhoff PLC"

    def __init__(self, ams_net_id=BECKHOFF_AMS_NET_ID, plc_ip=BECKHOFF_PLC_IP, variable=BECKHOFF_VARIABLE):
        self.ams_net_id = ams_net_id
        self.plc_ip = plc_ip
        self.variable = variable
        self.is_on = False
        self._plc = None
        self._lock = threading.Lock()
        if pyads is None:
            print("[Beckhoff] pyads not installed (pip install pyads) - commands will fail")
            return
        if self._connect():
            self._write(False)   # make sure the machine starts OFF

    # --- connection ---
    def _connect(self):
        if pyads is None:
            return False
        try:
            self._plc = pyads.Connection(self.ams_net_id, pyads.PORT_TC3PLC1, self.plc_ip)
            self._plc.open()
            self._plc.set_timeout(BECKHOFF_TIMEOUT_MS)
            self._plc.read_state()   # fails fast if the PLC is not reachable
            print(f"[Beckhoff] Connected to {self.plc_ip} ({self.ams_net_id})")
            return True
        except Exception as e:
            print(f"[Beckhoff] Cannot connect to {self.plc_ip}: {e}")
            self._disconnect()
            return False

    def _disconnect(self):
        if self._plc is not None:
            try:
                self._plc.close()
            except Exception:
                pass
            self._plc = None

    def _write(self, value):
        """Write the BOOL variable; reconnect once if the connection was lost."""
        for attempt in (1, 2):
            if self._plc is None and not self._connect():
                return False
            try:
                self._plc.write_by_name(self.variable, value, pyads.PLCTYPE_BOOL)
                self.is_on = value
                print(f"[Beckhoff] {self.variable} = {value}  (LED {'ON' if value else 'OFF'})")
                return True
            except Exception as e:
                print(f"[Beckhoff] Write failed (attempt {attempt}): {e}")
                self._disconnect()
        return False

    # --- same interface as MachineRelay ---
    def grant(self):
        with self._lock:
            return self._write(True)

    def off(self):
        with self._lock:
            return self._write(False)

    def status(self):
        with self._lock:
            if self._plc is None and not self._connect():
                return None
            try:
                return self._plc.read_by_name(self.variable, pyads.PLCTYPE_BOOL)
            except Exception as e:
                print(f"[Beckhoff] Read failed: {e}")
                self._disconnect()
                return None

    def close(self):
        with self._lock:
            if self._plc is not None:
                self._write(False)
            self._disconnect()


if __name__ == "__main__":
    # Interactive test, same commands as the original controller script
    out = BeckhoffOutput()
    print("Commands: on / off / status / quit")
    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == "on":
                out.grant()
            elif cmd == "off":
                out.off()
            elif cmd == "status":
                val = out.status()
                print("LED is", "unknown (not connected)" if val is None else ("ON" if val else "OFF"))
            elif cmd in ("quit", "exit"):
                break
            else:
                print("Use: on / off / status / quit")
    except KeyboardInterrupt:
        pass
    finally:
        out.close()
        print("\nClosed. LED off.")
