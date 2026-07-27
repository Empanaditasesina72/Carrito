"""Entry point for the TMR 2026 autonomous vehicle.
Raspberry Pi 5 - Sony IMX500 - IBT-2 - PCA9685 - VL53L0X

Thread architecture:
  CameraStream   -> daemon thread, updates the BGR frame at 30 FPS
  SignDetector   -> daemon thread, YOLO inference at ~12 FPS
  DistanceSensor -> daemon thread, VL53L0X polling at 50 Hz
  Main loop      -> 50 Hz: gamepad + FSM + servo + motor

Modes:
  STANDBY    -> Motor OFF, servo centred. Waits for the gamepad.
  MANUAL     -> Left stick -> servo | R2 -> forward | L2 -> reverse
  AUTONOMOUS -> 5-state FSM (CRUCERO/PRECAUCION/FRENADO/ESPERA/REANUDAR)
  PARKING    -> Battery parking (SEARCH -> MANEUVER -> PARKED)

Gamepad buttons (generic PS4/Xbox):
  Square   / X -> Toggle AUTONOMOUS (enable/disable)
  Cross    / A -> MANUAL
  Circle   / B -> VISION mode (camera ON, motors OFF, debug)
  Triangle / Y -> Toggle PARKING (battery parking)
  Start        -> Emergency: instant brake + MANUAL
"""

import os
import sys
import time
import signal
import subprocess

import cv2

_DISPLAY = "--display" in sys.argv
if _DISPLAY:
    os.environ.setdefault("DISPLAY", ":0")


def _release_gpio_from_systemd() -> None:
    if os.environ.get("INVOCATION_ID"):
        return
    try:
        active = subprocess.run(
            ["systemctl", "is-active", "--quiet", "carrito_tmr"],
            timeout=2,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return
    if active.returncode != 0:
        return

    print("[SYS] carrito_tmr.service active - stopping it to free the GPIO...")
    try:
        subprocess.run(
            ["sudo", "-n", "systemctl", "stop", "carrito_tmr"],
            timeout=10,
            check=True,
        )
    except subprocess.CalledProcessError:
        print("[SYS] Could not stop the service (sudo without NOPASSWD?).")
        print("[SYS] Run:  sudo systemctl stop carrito_tmr")
        sys.exit(1)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[SYS] Error stopping the service: {e}")
        sys.exit(1)
    time.sleep(0.5)
    print("[SYS] Service stopped - GPIO free.")

from config import (
    PIN_MOTOR_RPWM     as MOTOR_RPWM,
    PIN_MOTOR_LPWM     as MOTOR_LPWM,
    SERVO_CENTER_ANGLE as SERVO_CENTER,
    SERVO_MIN_ANGLE    as SERVO_MIN,
    SERVO_MAX_ANGLE    as SERVO_MAX,
    PIN_LED_TURN_LEFT, PIN_LED_TURN_RIGHT, PIN_LED_BRAKE, SIGNAL_BLINK_HZ,
    BTN_MANUAL, BTN_VISION, BTN_AUTONOMOUS, BTN_PARKING, BTN_EMERGENCY,
    AXIS_STEER, AXIS_THROTTLE, AXIS_BRAKE,
    JOYSTICK_DEADBAND as DEADBAND,
    USE_IMX500_NPU, IMX500_RPK_PATH, IMX500_LABELS_PATH, IMX500_CONF,
    USE_DRIVE_NET, DRIVE_NET_WEIGHTS, DRIVE_NET_CONF_MIN,
)

LOOP_HZ           = 50
CAMERA_W          = 640
CAMERA_H          = 480
CAMERA_FPS        = 30
AWB_WARMUP_S      = 2.0
YOLO_MODEL        = "weights/tmr_signs.pt"
YOLO_CONF         = 0.55
YOLO_IMGSZ        = 320

# Lane-following gains come from config.py -- ONE definition. These three
# entry points used to redefine them locally (0.08/0.002/0.025), agreeing
# with each other but not with config's 0.09, so tuning config changed
# nothing on the car.
from config import STEER_KP as PID_KP, STEER_KI as PID_KI, STEER_KD as PID_KD
PID_OUT_MIN       = -(SERVO_CENTER - SERVO_MIN)
PID_OUT_MAX       =  (SERVO_MAX - SERVO_CENTER)

import json


def _load_track_calib() -> dict:
    """Load on-track calibration (HSV white / right_bias / roi_frac / PID gains)
    saved by tools/tune_track.py, if it exists. An empty dict means the built-in
    defaults are used, so the file is optional and fully backward-compatible."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "track_calib.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            calib = json.load(f)
        print(f"[CALIB] Track calibration loaded: {path}")
        return calib
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[CALIB] Could not read track_calib.json ({e}); using defaults.")
        return {}


_TRACK_CALIB = _load_track_calib()

from hardware.motor            import MotorDriver
from hardware.steering_driver  import SteeringDriver
from hardware.distance_sensor  import DistanceSensor, make_distance_sensor
from hardware.signals          import TurnSignals, SignalMode
from hardware.brake_light      import BrakeLight
from vision.camera_stream      import CameraStream
from vision.lane_pipeline      import LanePipeline, LaneResult
from vision.sign_detector      import SignDetector
from control.pid_controller    import PIDController
from control.fsm               import AutonomousFSM
from control.parking_fsm       import ParkingFSM, ParkingState


class _KeyboardReader:
    """
    Reads one key per iteration from stdin WITHOUT blocking the control loop.
    Puts the TTY in cbreak mode (each key arrives instantly, no Enter).
    If stdin is not a TTY (e.g. running under systemd) it stays inert.
    Restores the terminal attributes in close().
    """

    def __init__(self):
        self.enabled = False
        self._old_attrs = None
        try:
            import termios, tty
            if not sys.stdin.isatty():
                return
            self._old_attrs = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self.enabled = True
        except Exception as e:
            print(f"[KB] Keyboard not available: {e}")

    def get_key(self):
        if not self.enabled:
            return None
        import select
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            return sys.stdin.read(1)
        return None

    def close(self):
        if self._old_attrs is None:
            return
        try:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_attrs)
        except Exception:
            pass
        self._old_attrs = None
        self.enabled = False


class VehicleTMR:
    """
    Full controller for the TMR 2026 vehicle.

    Integrates hardware, vision and the FSM in a single 50 Hz control loop.
    Image processing happens in separate threads -- the control loop NEVER
    waits for vision.
    """

    class Mode:
        STANDBY    = "STANDBY"
        MANUAL     = "MANUAL"
        VISION     = "VISION"
        AUTONOMOUS = "AUTONOMOUS"
        PARKING    = "PARKING"

    DISPLAY_MODES = ("VISION", "AUTONOMOUS", "PARKING")

    MODE_COOLDOWN_S = 0.4

    def __init__(self):
        print("=" * 55)
        print("  TMR 2026 - Initializing hardware...")
        print("=" * 55)

        self.motor    = MotorDriver(pin_rpwm=MOTOR_RPWM, pin_lpwm=MOTOR_LPWM)
        self.steering = SteeringDriver()
        self.sensor   = make_distance_sensor()
        self.signals  = TurnSignals(
            pin_left  = PIN_LED_TURN_LEFT,
            pin_right = PIN_LED_TURN_RIGHT,
            blink_hz  = SIGNAL_BLINK_HZ,
        )
        self.brake_light = BrakeLight(pin=PIN_LED_BRAKE)

        self.camera, self.sign_det = self._build_vision()
        _c = _TRACK_CALIB
        self.lane_pipe = LanePipeline(
            frame_w=CAMERA_W, frame_h=CAMERA_H, debug=_DISPLAY,
            right_bias=_c.get("right_bias", 0.70),
            roi_frac=_c.get("roi_frac", 0.5),
            hsv_white_lo=_c.get("hsv_white_lo"),
            hsv_white_hi=_c.get("hsv_white_hi"),
        )
        self.lane_pipe = self._maybe_drive_net(self.lane_pipe)

        _p = _TRACK_CALIB.get("pid", {})
        self.pid = PIDController(
            kp=_p.get("kp", PID_KP), ki=_p.get("ki", PID_KI), kd=_p.get("kd", PID_KD),
            setpoint=0.0,
            output_limits=(PID_OUT_MIN, PID_OUT_MAX),
            integral_limits=(-25.0, 25.0),
        )
        self.fsm = AutonomousFSM(
            self.motor, self.steering, self.pid,
            signals     = self.signals,
            brake_light = self.brake_light,
        )
        self.parking = ParkingFSM(self.motor, self.steering)

        self._joystick = None
        self._init_gamepad()

        self._kb = _KeyboardReader()
        if self._kb.enabled:
            print("[KB] Shortcuts:  A=MANUAL  B=VISION  X=AUTO  P=PARKING  "
                  "Space=EMERG  S=STANDBY  Q=quit")

        self._mode            = self.Mode.STANDBY
        self._running         = True
        self._last_mode_t     = 0.0
        self._last_lane       = LaneResult(error_px=0.0, confidence=0.0)
        self._btn_prev: dict  = {}
        self._sign_action     = ""

        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        print("[INIT] Hardware ready. Waiting for the Bluetooth gamepad...")


    def _maybe_drive_net(self, classic_pipe):
        """Opt-in swap of the classic lane follower for the learned DriveNet.

        Controlled by config.py:USE_DRIVE_NET (default False -> no change). The
        returned object honours the same .process()/.draw_debug() contract, so
        the FSM, PID, sign gating and lights are untouched. Any failure (missing
        weights, missing torch) silently keeps the classic LanePipeline.
        """
        if not USE_DRIVE_NET:
            return classic_pipe
        from pathlib import Path
        if not Path(DRIVE_NET_WEIGHTS).exists():
            print(f"[INIT] USE_DRIVE_NET set but {DRIVE_NET_WEIGHTS} missing; "
                  f"using classic LanePipeline")
            return classic_pipe
        try:
            from vision.drive_net import DriveNet
            net = DriveNet(DRIVE_NET_WEIGHTS, conf_min=DRIVE_NET_CONF_MIN,
                           debug=_DISPLAY)
            print(f"[INIT] DriveNet steering ENABLED ({DRIVE_NET_WEIGHTS})")
            return net
        except Exception as e:
            print(f"[INIT] DriveNet unavailable ({e}); using classic LanePipeline")
            return classic_pipe

    def _build_vision(self):
        """
        Pick the sign-detection backend based on the available hardware:

          1. IMX500 NPU (`vision/imx500_detector.py`) -- the model runs INSIDE
             the camera and the CPU stays free. Requires the .rpk generated
             with `tools/export_imx500.py`. A single object acts as camera and
             detector.
          2. CPU (CameraStream + SignDetector NCNN) -- the validated path.

        Returns (camera, sign_det). If the NPU fails for any reason, it falls
        back to the CPU path without interrupting startup.
        """
        if USE_IMX500_NPU and os.path.isfile(IMX500_RPK_PATH):
            try:
                from vision.imx500_detector import IMX500CameraStream
                npu = IMX500CameraStream(
                    rpk_path     = IMX500_RPK_PATH,
                    labels_path  = IMX500_LABELS_PATH,
                    width        = CAMERA_W,
                    height       = CAMERA_H,
                    fps          = CAMERA_FPS,
                    conf         = IMX500_CONF,
                    awb_warmup_s = AWB_WARMUP_S,
                )
                print("[VISION] Backend: IMX500 NPU (on-chip inference)")
                return npu, npu
            except Exception as e:
                print(f"[VISION] NPU unavailable ({e}) - using the CPU path.")
        elif USE_IMX500_NPU:
            print(f"[VISION] No .rpk ({IMX500_RPK_PATH}) - using the CPU path."
                  "  Generate it with: python tools/export_imx500.py")

        camera = CameraStream(
            width=CAMERA_W, height=CAMERA_H,
            fps=CAMERA_FPS, awb_warmup_s=AWB_WARMUP_S,
        )
        sign_det = SignDetector(
            model_path=YOLO_MODEL, conf=YOLO_CONF, imgsz=YOLO_IMGSZ
        )
        print("[VISION] Backend: CPU (CameraStream + SignDetector)")
        return camera, sign_det



    def run(self) -> None:
        self.sensor.start()
        self.camera.start()
        self.sign_det.start()

        print(f"[MAIN] Control loop at {LOOP_HZ} Hz started.")
        dt = 1.0 / LOOP_HZ
        t_last = time.monotonic()

        try:
            while self._running:
                now   = time.monotonic()
                dt    = now - t_last
                t_last = now

                self._pump_gamepad_events()
                self._process_gamepad()
                self._poll_keyboard()
                self._update_vision()
                self._run_mode(dt)

                self.signals.tick()

                elapsed = time.monotonic() - now
                sleep   = max(0.0, (1.0 / LOOP_HZ) - elapsed)
                if sleep > 0:
                    time.sleep(sleep)

        finally:
            self._shutdown()


    def _init_gamepad(self) -> None:
        """
        Initialize pygame's joystick subsystem. The actual connection to the
        first gamepad happens in `_pump_gamepad_events()` (via SDL2's
        JOYDEVICEADDED event), so the system starts even if the controller is
        off and latches onto it automatically when powered on.
        """
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            print("[PAD] Waiting for a gamepad - it will latch on automatically when powered on.")
        except Exception as e:
            print(f"[PAD] pygame not available: {e}")

    def _pump_gamepad_events(self) -> None:
        """
        Drain the SDL event queue to keep the joystick state up to date and
        react to JOYDEVICEADDED/REMOVED. Called once per iteration of the
        main loop.
        """
        try:
            import pygame
        except Exception:
            return
        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEADDED:
                try:
                    joy = pygame.joystick.Joystick(event.device_index)
                    joy.init()
                    self._joystick = joy
                    self._btn_prev.clear()
                    print(f"[PAD] Gamepad connected: {joy.get_name()}")
                except Exception as e:
                    print(f"[PAD] Error latching gamepad: {e}")
            elif event.type == pygame.JOYDEVICEREMOVED:
                if self._joystick is not None:
                    print("[PAD] Gamepad disconnected - waiting for reconnection...")
                self._joystick = None
                self._btn_prev.clear()

    def _process_gamepad(self) -> None:
        if self._joystick is None:
            return

        now = time.monotonic()
        if now - self._last_mode_t < self.MODE_COOLDOWN_S:
            return

        def btn(idx: int) -> bool:
            """True ONLY on the button's rising edge."""
            cur = bool(self._joystick.get_button(idx))
            prev = self._btn_prev.get(idx, False)
            self._btn_prev[idx] = cur
            return cur and not prev

        if btn(BTN_EMERGENCY):
            print("[PAD] EMERGENCY - brake + MANUAL")
            self.motor.brake()
            self.fsm.deactivate()
            self._set_mode(self.Mode.MANUAL)
            return

        if btn(BTN_AUTONOMOUS):
            if self._mode == self.Mode.AUTONOMOUS:
                self.fsm.deactivate()
                self._set_mode(self.Mode.MANUAL)
            else:
                self.motor.brake()
                self._set_mode(self.Mode.AUTONOMOUS)
                self.fsm.activate()
            return

        if btn(BTN_MANUAL):
            self.fsm.deactivate()
            self._set_mode(self.Mode.MANUAL)
            return

        if btn(BTN_VISION):
            if self._mode == self.Mode.VISION:
                self._set_mode(self.Mode.MANUAL)
            else:
                self.motor.brake()
                self._set_mode(self.Mode.VISION)
            return

        if btn(BTN_PARKING):
            if self._mode == self.Mode.PARKING:
                self._set_mode(self.Mode.MANUAL)
            else:
                self.fsm.deactivate()
                self.motor.brake()
                self._set_mode(self.Mode.PARKING)
                self.parking.activate()
            return


    def _poll_keyboard(self) -> None:
        if not self._kb.enabled:
            return
        key = self._kb.get_key()
        if key:
            self._process_key(key)

    def _process_key(self, key: str) -> None:
        """
        A=MANUAL  B=VISION  X=AUTO (toggle)  P=PARKING (toggle)
        Space=EMERG  S=STANDBY  Q=quit
        Space and Q ignore the cooldown (immediate stop).
        """
        k = key.lower()

        if k == "q":
            print("\n[KB] Quit requested.")
            self._running = False
            return

        if key == " ":
            print("\n[KB] EMERGENCY - brake + MANUAL")
            self.motor.brake()
            self.fsm.deactivate()
            self._set_mode(self.Mode.MANUAL)
            return

        if time.monotonic() - self._last_mode_t < self.MODE_COOLDOWN_S:
            return

        if k == "a":
            print("\n[KB] -> MANUAL")
            self.fsm.deactivate()
            self._set_mode(self.Mode.MANUAL)
        elif k == "b":
            if self._mode == self.Mode.VISION:
                print("\n[KB] VISION -> MANUAL")
                self._set_mode(self.Mode.MANUAL)
            else:
                print("\n[KB] -> VISION (camera preview)")
                self.motor.brake()
                self._set_mode(self.Mode.VISION)
        elif k == "x":
            if self._mode == self.Mode.AUTONOMOUS:
                print("\n[KB] AUTONOMOUS -> MANUAL")
                self.fsm.deactivate()
                self._set_mode(self.Mode.MANUAL)
            else:
                print("\n[KB] -> AUTONOMOUS")
                self.motor.brake()
                self._set_mode(self.Mode.AUTONOMOUS)
                self.fsm.activate()
        elif k == "p":
            if self._mode == self.Mode.PARKING:
                print("\n[KB] PARKING -> MANUAL")
                self._set_mode(self.Mode.MANUAL)
            else:
                print("\n[KB] -> PARKING (battery parking)")
                self.fsm.deactivate()
                self.motor.brake()
                self._set_mode(self.Mode.PARKING)
                self.parking.activate()
        elif k == "s":
            print("\n[KB] -> STANDBY")
            self.fsm.deactivate()
            self.motor.brake()
            self._set_mode(self.Mode.STANDBY)

    def _set_mode(self, mode: str) -> None:
        if mode != self._mode:
            print(f"[MODE] {self._mode} -> {mode}")
            if self._mode == self.Mode.PARKING:
                self.parking.deactivate()
            if (_DISPLAY and self._mode in self.DISPLAY_MODES
                    and mode not in self.DISPLAY_MODES):
                cv2.destroyAllWindows()
            if mode == self.Mode.VISION or self._mode == self.Mode.VISION:
                self.pid.reset()
        self._mode = mode
        self._last_mode_t = time.monotonic()


    def _update_vision(self) -> None:
        """
        Get the most recent frame and update LanePipeline + SignDetector.
        Never blocks -- if there is no new frame, it uses the previous result.
        """
        frame = self.camera.get_frame()
        if frame is None:
            return

        self._last_lane = self.lane_pipe.process(frame)

        self.sign_det.update_frame(frame)

        self.fsm.lane_error   = self._last_lane.error_px
        self.fsm.lane_heading = self._last_lane.heading
        self.fsm.lane_conf    = self._last_lane.confidence
        self.fsm.lidar_mm     = self.sensor.front_mm

        stop_like = (self.sign_det.has_sign("stop_sign")
                     or self.sign_det.has_sign("red"))
        self.fsm.sign_visible = stop_like

        closest = (self.sign_det.closest_sign("stop_sign")
                   or self.sign_det.closest_sign("red"))
        if closest is not None and closest.distance_m is not None:
            self.fsm.sign_distance_mm = closest.distance_m * 1000.0
        else:
            self.fsm.sign_distance_mm = None

        self._sign_action = self._decide_sign_action()

    SIGN_ACTIONS = {
        "stop_sign": "Full STOP (5 s)",
        "red":       "RED light: brake",
        "green":     "GREEN light: go",
        "yellow":    "YELLOW light: caution",
        "left":      "LEFT arrow",
        "right":     "RIGHT arrow",
        "straight":  "Go STRAIGHT",
    }

    def _decide_sign_action(self) -> str:
        """Return the action text of the closest detected sign."""
        dets = self.sign_det.get_detections()
        if not dets:
            return ""
        with_dist = [d for d in dets if d.distance_m is not None]
        if not with_dist:
            return self.SIGN_ACTIONS.get(dets[0].label, "")
        closest = min(with_dist, key=lambda d: d.distance_m)
        return self.SIGN_ACTIONS.get(closest.label, closest.label)


    def _run_mode(self, dt: float) -> None:
        match self._mode:
            case self.Mode.STANDBY:
                self._do_standby()
            case self.Mode.MANUAL:
                self._do_manual()
            case self.Mode.VISION:
                self._do_vision(dt)
            case self.Mode.AUTONOMOUS:
                self.fsm.update(dt)
                self._log_autonomous()
                if _DISPLAY:
                    self._render_debug_view(mode_label="AUT")
            case self.Mode.PARKING:
                self._do_parking(dt)

    def _do_parking(self, dt: float) -> None:
        """Battery parking -- same ParkingFSM validated in the simulator."""
        self.parking.lidar_mm = self.sensor.front_mm
        self.parking.update(dt)

        self.signals.set_mode(SignalMode.HAZARD)
        if self.parking.state == ParkingState.PARKED:
            self.brake_light.on()
        else:
            self.brake_light.off()

        print(f"\r[PARK] {self.parking.state.name:<16} "
              f"duty:{self.motor.current_duty:+.0f}%  "
              f"angle:{self.steering.current_angle:5.1f}°   ",
              end="", flush=True)
        if _DISPLAY:
            self._render_debug_view(mode_label="PARK")

    def _log_autonomous(self) -> None:
        """Autonomous-mode status line -- includes what YOLO sees."""
        dets = self.sign_det.get_detections()
        if dets:
            sign_txt = ", ".join(
                f"{d.label}@{(d.distance_m or 0)*100:.0f}cm" for d in dets[:2]
            )
        else:
            sign_txt = "-"
        lidar_txt = f"{self.sensor.front_mm:.0f}" if self.sensor.front_mm else "---"
        action = self._sign_action or "-"
        print(f"\r[AUT] {self.fsm.state.name:<10}  "
              f"err:{self._last_lane.error_px:+5.0f}px  "
              f"angle:{self.steering.current_angle:5.1f}°  "
              f"duty:{self.motor.current_duty:+.0f}%  "
              f"lidar:{lidar_txt}mm  signs:{sign_txt}  -> {action}   ",
              end="", flush=True)

    def _do_standby(self) -> None:
        self.motor.brake()
        self.steering.center()
        self.signals.set_mode(SignalMode.OFF)
        self.brake_light.off()
        if self._joystick is None:
            return
        try:
            import pygame
            pygame.event.pump()
            for i in range(self._joystick.get_numbuttons()):
                if self._joystick.get_button(i):
                    self._set_mode(self.Mode.MANUAL)
                    return
        except Exception:
            pass

    def _do_manual(self) -> None:
        """Manual control: left stick X -> servo | R2 -> forward | L2 -> reverse."""
        if self._joystick is None:
            self.motor.brake()
            return

        steer_raw = self._joystick.get_axis(AXIS_STEER)
        if abs(steer_raw) < DEADBAND:
            steer_raw = 0.0
        angle = SERVO_CENTER + steer_raw * (SERVO_CENTER - SERVO_MIN)
        self.steering.set_angle(angle)

        if   steer_raw < -0.30:
            self.signals.set_mode(SignalMode.LEFT)
        elif steer_raw > +0.30:
            self.signals.set_mode(SignalMode.RIGHT)
        else:
            self.signals.set_mode(SignalMode.OFF)

        throttle = self._joystick.get_axis(AXIS_THROTTLE)
        brake    = self._joystick.get_axis(AXIS_BRAKE)

        t = (throttle + 1.0) / 2.0
        b = (brake    + 1.0) / 2.0

        if b > DEADBAND:
            self.motor.set_speed(-(b ** 2) * 40.0)
        elif t > DEADBAND:
            self.motor.set_speed((t ** 1.3) * 55.0)
        else:
            self.motor.set_speed(0.0)

        if self.motor.current_duty < -1.0:
            self.brake_light.on()
        else:
            self.brake_light.off()

        dets = self.sign_det.get_detections()
        if dets:
            sign_txt = ", ".join(
                f"{d.label}@{(d.distance_m or 0)*100:.0f}cm" for d in dets[:2]
            )
        else:
            sign_txt = "-"

        print(f"\r[MAN] steer:{steer_raw:+.2f} ({angle:.0f}°)  "
              f"t:{t:.2f}  b:{b:.2f}  duty:{self.motor.current_duty:+.0f}%  "
              f"signs:{sign_txt}   ",
              end="", flush=True)

    def _do_vision(self, dt: float) -> None:
        """Vision debug -- motors OFF, shows the pipeline + PID on screen."""
        self.motor.brake()
        self.steering.center()
        self.signals.set_mode(SignalMode.OFF)
        self.brake_light.off()

        if self.camera.get_frame() is None:
            return

        self.pid.compute(self._last_lane.error_px, dt)

        if _DISPLAY:
            self._render_debug_view(mode_label="VIS")

        dets = self.sign_det.get_detections()
        sign_txt = ", ".join(f"{d.label}({d.confidence:.0%})" for d in dets) or "-"
        lidar_txt = f"{self.sensor.front_mm:.0f}" if self.sensor.front_mm else "---"
        angle_target = max(
            SERVO_MIN, min(SERVO_MAX, SERVO_CENTER - self.pid.last_output)
        )
        print(f"\r[VIS] err:{self._last_lane.error_px:+.0f}px "
              f"conf:{self._last_lane.confidence:.0%}  "
              f"P:{self.pid.last_p:+5.2f} I:{self.pid.last_i:+5.2f} "
              f"D:{self.pid.last_d:+5.2f}  corr:{self.pid.last_output:+5.2f}d "
              f"angle:{angle_target:5.1f}d  lidar:{lidar_txt}mm  "
              f"signs:{sign_txt}   ",
              end="", flush=True)


    def _render_debug_view(self, mode_label: str) -> None:
        """
        Draw a single window with ALL the visual debug:
          - Frame annotated with the detected lane centre line.
          - Top mosaic: BEV (top-down view) + HSV mask.
          - YOLO bounding boxes with label + confidence + distance.
          - Bottom-left panel: PID values (P/I/D/corr) and error.
          - Bottom-right panel: detected objects + sign action.

        Called from VISION, AUTONOMOUS and PARKING when --display is active.
        Non-blocking: cv2.waitKey(1).
        """
        frame = self.camera.get_frame()
        if frame is None:
            return

        lane  = self._last_lane
        dets  = self.sign_det.get_detections()
        lidar = self.sensor.front_mm

        vis = self.lane_pipe.draw_debug(frame, lane)

        if lane.bev_frame is not None and lane.mask_frame is not None:
            vis[0:180, 0:320]   = cv2.resize(lane.bev_frame,  (320, 180))
            vis[0:180, 320:640] = cv2.resize(lane.mask_frame, (320, 180))
            cv2.putText(vis, "BEV (bird's-eye)", (8, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(vis, "HSV white mask",   (328, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        angle_target = max(
            SERVO_MIN, min(SERVO_MAX, SERVO_CENTER - self.pid.last_output)
        )

        self._draw_panel(vis, x=8, y=200, w=320, h=160, lines=[
            f"MODE  : {mode_label}",
            f"err   :{lane.error_px:+7.1f}px  conf:{lane.confidence:.0%}",
            f"P     :{self.pid.last_p:+7.2f}   kp={self.pid.kp:.3f}",
            f"I     :{self.pid.last_i:+7.2f}   ki={self.pid.ki:.3f}",
            f"D     :{self.pid.last_d:+7.2f}   kd={self.pid.kd:.3f}",
            f"corr  :{self.pid.last_output:+7.2f}d -> servo {angle_target:5.1f}d",
            f"lidar :{lidar:.0f}mm" if lidar else "lidar :---",
        ])

        if dets:
            obj_lines = ["DETECTED OBJECTS:"]
            for d in dets[:4]:
                dist = f" @{(d.distance_m or 0)*100:.0f}cm" if d.distance_m else ""
                obj_lines.append(f"- {d.label}  {d.confidence:.0%}{dist}")
            if self._sign_action:
                obj_lines.append(f"-> {self._sign_action}")
        else:
            obj_lines = ["DETECTED OBJECTS:", "- (none)"]
        self._draw_panel(vis, x=336, y=200, w=296, h=160, lines=obj_lines)

        try:
            if self._mode == self.Mode.PARKING:
                fsm_txt = f"FSM:{self.parking.state.name}"
            else:
                fsm_txt = f"FSM:{self.fsm.state.name}"
        except Exception:
            fsm_txt = ""
        cv2.putText(vis, f"{mode_label}  {fsm_txt}  duty:{self.motor.current_duty:+.0f}%",
                    (8, CAMERA_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)

        for d in dets:
            cv2.rectangle(vis, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
            dist_txt = f" {(d.distance_m or 0)*100:.0f}cm" if d.distance_m else ""
            label_txt = f"{d.label} {d.confidence:.0%}{dist_txt}"
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = max(d.y1 - 6, th + 4)
            cv2.rectangle(vis, (d.x1, ly - th - 4), (d.x1 + tw + 4, ly + 2),
                          (0, 0, 0), -1)
            cv2.putText(vis, label_txt, (d.x1 + 2, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("TMR 2026 - Vision Debug", vis)
        cv2.waitKey(1)

    @staticmethod
    def _draw_panel(img, x: int, y: int, w: int, h: int, lines: list[str]) -> None:
        """Semi-transparent box with multi-line text."""
        ov = img.copy()
        cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, dst=img)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 220, 0), 1)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (x + 8, y + 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 220, 0), 1, cv2.LINE_AA)


    def _handle_signal(self, signum, _frame) -> None:
        print(f"\n[SYS] Signal {signum} received -> shutting down...")
        self._running = False

    def _shutdown(self) -> None:
        print("\n[SYS] Shutting down the system...")
        try:    self._kb.close()
        except: pass
        if _DISPLAY:
            cv2.destroyAllWindows()
        self.fsm.deactivate()
        self.parking.deactivate()
        self.motor.brake()
        time.sleep(0.1)
        self.sensor.stop()
        self.sign_det.stop()
        self.camera.stop()
        try:    self.signals.close()
        except: pass
        try:    self.brake_light.close()
        except: pass
        self.motor.cleanup()
        print("[SYS] Shutdown complete.")


if __name__ == "__main__":
    _release_gpio_from_systemd()
    VehicleTMR().run()
