"""Two optional VL53L0X sensors on I2C bus 4 (GPIO 22/23).

If the sensors are not connected the system keeps running and every reading
returns None. Autonomous mode then relies only on the camera to estimate
distances.

Wiring:
  SDA         -> GPIO 23 (Pin 16)
  SCL         -> GPIO 22 (Pin 15)
  XSHUT front -> GPIO 24 (Pin 18)  -- free, reserved for a future front lidar
  XSHUT rear  -> GPIO 27 (Pin 13)

If the front sensor does not respond (not physically connected) the rear one
keeps working and the system degrades gracefully: front_mm stays None.
"""

import threading
import time

import lgpio

from config import (
    PIN_TOF_XSHUT_FRONT, PIN_TOF_XSHUT_REAR,
    TOF_ADDR_FRONT, TOF_ADDR_REAR,
    TOF_TIMING_BUDGET_US, TOF_MAX_RANGE_MM, TOF_POLL_INTERVAL_S,
)

_CHIP = 4


class DistanceSensor:
    """
    Manages two VL53L0X sensors. If unavailable, always returns None.
    """

    def __init__(self):
        self._available = False
        self._front = None
        self._rear  = None

        self._h = lgpio.gpiochip_open(_CHIP)
        lgpio.gpio_claim_output(self._h, PIN_TOF_XSHUT_FRONT, 0, 0)
        lgpio.gpio_claim_output(self._h, PIN_TOF_XSHUT_REAR,  0, 0)

        try:
            import adafruit_vl53l0x
            from adafruit_extended_bus import ExtendedI2C

            i2c = ExtendedI2C(4)

            lgpio.gpio_write(self._h, PIN_TOF_XSHUT_FRONT, 1)
            time.sleep(0.1)
            try:
                front = adafruit_vl53l0x.VL53L0X(i2c)
                front.set_address(TOF_ADDR_FRONT)
                front.measurement_timing_budget = TOF_TIMING_BUDGET_US
                self._front = front
                print(f"[TOF] Front OK @ 0x{TOF_ADDR_FRONT:02X}")
            except Exception as e:
                self._front = None
                print(f"[TOF] Front not detected ({e}) - degrading to rear-only")

            lgpio.gpio_write(self._h, PIN_TOF_XSHUT_REAR, 1)
            time.sleep(0.1)
            try:
                rear = adafruit_vl53l0x.VL53L0X(i2c)
                rear.measurement_timing_budget = TOF_TIMING_BUDGET_US
                self._rear = rear
                print(f"[TOF] Rear OK @ 0x{TOF_ADDR_REAR:02X}")
            except Exception as e:
                self._rear = None
                print(f"[TOF] Rear not detected ({e})")

            self._available = (self._front is not None) or (self._rear is not None)
            if not self._available:
                print("[TOF] No sensor responds - continuing without ToF")

        except Exception as e:
            lgpio.gpio_write(self._h, PIN_TOF_XSHUT_FRONT, 1)
            lgpio.gpio_write(self._h, PIN_TOF_XSHUT_REAR,  1)
            print(f"[TOF] Init failed ({e}) - continuing without ToF")

        self._front_mm: float | None = None
        self._rear_mm:  float | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if not self._available:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, name="ToF", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        lgpio.gpio_free(self._h, PIN_TOF_XSHUT_FRONT)
        lgpio.gpio_free(self._h, PIN_TOF_XSHUT_REAR)
        lgpio.gpiochip_close(self._h)

    @property
    def front_mm(self) -> float | None:
        with self._lock:
            return self._front_mm

    @property
    def rear_mm(self) -> float | None:
        with self._lock:
            return self._rear_mm

    @property
    def distance_mm(self) -> float | None:
        """Front alias for compatibility with the rest of the code."""
        return self.front_mm

    def is_obstacle_near(self, threshold_mm: float) -> bool:
        d = self.front_mm
        return d is not None and d < threshold_mm

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                rf = self._front.range if self._front else None
                rr = self._rear.range  if self._rear  else None
                front = rf if (rf and rf < TOF_MAX_RANGE_MM) else None
                rear  = rr if (rr and rr < TOF_MAX_RANGE_MM) else None
            except Exception:
                front = rear = None

            with self._lock:
                self._front_mm = front
                self._rear_mm  = rear

            time.sleep(TOF_POLL_INTERVAL_S)


class NullDistanceSensor:
    """
    Stand-in for the ToF pair when it is disabled or absent.

    Presents the same interface and always reports "no reading", which is what
    every consumer already handles: the FSM treats `lidar_mm is None` as "no
    distance information" and falls back to the camera path, and the telemetry
    prints "---". Using a null object keeps the decision in one place instead of
    scattering `if sensor is not None` through main.py's eight call sites.
    """

    def start(self):
        print("[ToF] DISABLED (config.py:USE_TOF_SENSORS=False) - "
              "distance readings will always be None")

    def stop(self):
        pass

    @property
    def front_mm(self) -> float | None:
        return None

    @property
    def rear_mm(self) -> float | None:
        return None

    @property
    def distance_mm(self) -> float | None:
        return None

    def is_obstacle_near(self, threshold_mm: float) -> bool:
        return False


def make_distance_sensor():
    """Real sensor pair, or a null one when config.py disables them."""
    try:
        from config import USE_TOF_SENSORS
    except ImportError:
        USE_TOF_SENSORS = True
    return DistanceSensor() if USE_TOF_SENSORS else NullDistanceSensor()
