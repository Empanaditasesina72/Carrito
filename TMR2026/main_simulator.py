"""PC entry point for the TMR 2026 vehicle (Sim2Real validation).

A version of main.py adapted to run on a PC connected to the Unity simulator.
It replaces:
  - MotorDriver         -> MockMotorDriver (via SimulatorClient)
  - SteeringDriver      -> MockSteeringDriver (via SimulatorClient)
  - DistanceSensor      -> MockDistanceSensor (via SimulatorClient)
  - CameraStream        -> MockCameraStream (via SimulatorClient)
  - GPIO signals/brake  -> NoOp versions (no physical hardware)

Usage:
  python main_simulator.py               # PC mode, no display
  python main_simulator.py --display     # PC mode with a debug window

The Unity simulator must be listening on 127.0.0.1:5005 and sending:
  - JPEG frames (4-byte size header + JPEG data)
  - TOF messages: "TOF:front_mm,rear_mm"

The architecture is still multi-threaded with a 50 Hz main loop.
"""

import os
import sys
import time
import signal
import threading
from typing import Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2
import numpy as np

_DISPLAY = "--display" in sys.argv
_START_STANDBY = "--standby" in sys.argv
_VALIDATE = "--validate" in sys.argv
_PARKING = "--parking" in sys.argv
_PARK_AFTER_STOP_S = 6.0
_PARK_FALLBACK_S = 22.0
if _DISPLAY:
    os.environ.setdefault("DISPLAY", ":0")

_DURATION = 0.0
for _i, _a in enumerate(sys.argv):
    if _a == "--duration" and _i + 1 < len(sys.argv):
        try: _DURATION = float(sys.argv[_i + 1])
        except ValueError: _DURATION = 0.0

from config import (
    SERVO_CENTER_ANGLE as SERVO_CENTER,
    SERVO_MIN_ANGLE    as SERVO_MIN,
    SERVO_MAX_ANGLE    as SERVO_MAX,
    USE_DRIVE_NET, DRIVE_NET_WEIGHTS, DRIVE_NET_CONF_MIN,
)

LOOP_HZ           = 50
CAMERA_W          = 640
CAMERA_H          = 480
CAMERA_FPS        = 30
PID_KP            = 0.08
PID_KI            = 0.002
PID_KD            = 0.025
PID_OUT_MIN       = -(SERVO_CENTER - SERVO_MIN)
PID_OUT_MAX       =  (SERVO_MAX - SERVO_CENTER)

SIMULATOR_HOST    = "127.0.0.1"
SIMULATOR_PORT    = 5005

from sim_hardware_mocks import SimulatorClient
from vision.lane_pipeline import LanePipeline, LaneResult
from vision.sign_detector import SignDetector
from control.pid_controller import PIDController
from control.fsm import AutonomousFSM, FSMState
from control.parking_fsm import ParkingFSM
from validation_logger import ValidationLogger

class NoOpSignals:
    """TurnSignals mock for PC."""

    def __init__(self, *args, **kwargs):
        pass

    def set_mode(self, mode):
        pass

    def tick(self):
        pass

    def close(self):
        pass


class NoOpBrakeLight:
    """BrakeLight mock for PC."""
    def __init__(self, *args, **kwargs):
        pass

    def on(self):
        pass

    def off(self):
        pass

    def close(self):
        pass


class SimulatedCameraWrapper:
    """
    Wraps MockCameraStream to provide an interface compatible with
    vision/lane_pipeline.py (which expects get_frame() returning a BGR array).
    """

    def __init__(self, mock_camera, out_w=CAMERA_W, out_h=CAMERA_H):
        self.mock_camera = mock_camera
        self._lock = threading.Lock()
        self._out_w = out_w
        self._out_h = out_h

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Return the latest BGR frame, ALWAYS resized to (out_w, out_h).
        Unity sends 320x240 (RenderTexture) but the pipeline expects 640x480;
        without this resize, warpPerspective gets an empty frame and crashes.
        """
        with self._lock:
            frame = self.mock_camera.get_latest_frame()
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        if (w, h) != (self._out_w, self._out_h):
            frame = cv2.resize(frame, (self._out_w, self._out_h))
        return frame

    def start(self):
        """No-op: MockCameraStream is already running in threads."""
        pass

    def stop(self):
        """No-op."""
        pass


class VehicleSimulator:
    """
    TMR 2026 controller in simulation mode (PC).
    Identical to VehicleTMR except it uses SimulatorClient instead of Pi
    hardware.
    """

    class Mode:
        STANDBY    = "STANDBY"
        MANUAL     = "MANUAL"
        VISION     = "VISION"
        AUTONOMOUS = "AUTONOMOUS"
        PARKING    = "PARKING"

    MODE_COOLDOWN_S = 0.4

    def __init__(self):
        print("=" * 70)
        print("  TMR 2026 - SIMULATOR MODE (PC)")
        print("=" * 70)

        try:
            self.sim = SimulatorClient(host=SIMULATOR_HOST, port=SIMULATOR_PORT)
        except Exception as e:
            print(f"[SIM] ERROR: cannot connect to the simulator at {SIMULATOR_HOST}:{SIMULATOR_PORT}")
            print(f"[SIM] Is Unity running?")
            raise

        self.motor    = self.sim.motor
        self.steering = self.sim.steering
        self.sensor   = self.sim.distance
        self.camera_raw = self.sim.camera

        self.camera = SimulatedCameraWrapper(self.camera_raw)

        self.signals    = NoOpSignals()
        self.brake_light = NoOpBrakeLight()

        self.lane_pipe = LanePipeline(
            frame_w=CAMERA_W, frame_h=CAMERA_H, debug=_DISPLAY,
            right_bias=0.75,
            roi_frac=0.30,
            hsv_white_lo=[0, 0, 200],
            hsv_white_hi=[179, 40, 255],
            # The simulator's camera is exactly on the vehicle centreline, so it
            # has none of the physical car's mechanical bias. Pinning 0 keeps
            # config's LANE_ERROR_OFFSET_PX -- a measurement of THIS chassis --
            # from leaking into the sim and breaking Sim2Real parity.
            error_offset_px=0.0,
        )
        if USE_DRIVE_NET:
            from pathlib import Path as _Path
            if _Path(DRIVE_NET_WEIGHTS).exists():
                try:
                    from vision.drive_net import DriveNet
                    self.lane_pipe = DriveNet(DRIVE_NET_WEIGHTS,
                                              conf_min=DRIVE_NET_CONF_MIN,
                                              debug=_DISPLAY)
                    print(f"[SIM] DriveNet steering ENABLED ({DRIVE_NET_WEIGHTS})")
                except Exception as e:
                    print(f"[SIM] DriveNet unavailable ({e}); classic LanePipeline")
            else:
                print(f"[SIM] USE_DRIVE_NET set but {DRIVE_NET_WEIGHTS} missing")
        self.sign_det = SignDetector(
            model_path="weights/tmr_signs.pt",
            conf=0.55,
            imgsz=320
        )

        self.pid = PIDController(
            kp=PID_KP, ki=PID_KI, kd=PID_KD,
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

        self._mode            = self.Mode.STANDBY
        self._running         = True
        self._last_lane       = LaneResult(error_px=0.0, confidence=0.0)
        self._start_time      = time.monotonic()
        self._sign_action     = ""

        self.vlog = ValidationLogger("validation_results") if _VALIDATE else None

        self._stop_seen   = False
        self._espera_done = False
        self._stop_done_t = 0.0

        self._metrics = {
            "frame_count": 0,
            "loop_count": 0,
            "errors_px": [],
            "pid_outputs": [],
            "servo_angles": [],
            "motor_duties": [],
            "lidar_readings": [],
            "fsm_states": [],
        }

        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        print("[INIT] Connection to the simulator established.")
        print("[INIT] Starting vision and FSM...")

    def run(self) -> None:
        """Main control loop at 50 Hz."""
        self.sign_det.start()

        if not _START_STANDBY:
            self._mode = self.Mode.AUTONOMOUS
            self.fsm.activate()
            print("[MAIN] AUTONOMOUS mode enabled (the car drives by itself).")
            if _DISPLAY:
                print("[MAIN] Window keys:  A=Autonomous  S=Stop  Q=Quit")

        print("[MAIN] Waiting for Unity frames...")
        t_wait = time.monotonic()
        while (self.camera.get_frame() is None
               and self.sensor.front_mm is None
               and (time.monotonic() - t_wait) < 20.0):
            time.sleep(0.1)
        if self.camera.get_frame() is None and self.sensor.front_mm is None:
            print("=" * 60)
            print("[ERROR] Unity connected but is NOT sending frames/sensors.")
            print("  Typical cause: ANOTHER Python instance is connected.")
            print("  Fix: close ALL Python terminals, in Unity press STOP and")
            print("  then PLAY, and run ONLY this script again.")
            print("=" * 60)
        else:
            print("[MAIN] Frames received. Starting control.")

        print(f"[MAIN] Control loop at {LOOP_HZ} Hz started.")
        if _DURATION > 0:
            print(f"[MAIN] Fixed duration: {_DURATION:.0f} s (then saves and exits).")
        dt = 1.0 / LOOP_HZ
        t_last = time.monotonic()
        t_run0 = time.monotonic()

        try:
            while self._running:
                now   = time.monotonic()
                dt    = now - t_last
                t_last = now

                if _DURATION > 0 and (now - t_run0) >= _DURATION:
                    self._running = False
                    break

                if _PARKING and self._mode == self.Mode.AUTONOMOUS:
                    est = self.fsm.state
                    if est in (FSMState.PRECAUCION, FSMState.FRENADO, FSMState.ESPERA):
                        self._stop_seen = True
                    if est == FSMState.ESPERA:
                        self._espera_done = True
                    if (self._espera_done and self._stop_done_t == 0.0
                            and est == FSMState.CRUCERO):
                        self._stop_done_t = now
                    listo_tras_stop = (self._stop_done_t > 0.0
                                       and (now - self._stop_done_t) >= _PARK_AFTER_STOP_S)
                    respaldo = (not self._stop_seen
                                and _DURATION > 0
                                and (now - t_run0) >= (_DURATION - 14.0))
                    if listo_tras_stop or respaldo:
                        self.fsm.deactivate()
                        self._mode = self.Mode.PARKING
                        self.parking.activate()

                cycle_t0 = time.monotonic()
                self._update_vision()
                self._run_mode(dt)
                latency_ms = (time.monotonic() - cycle_t0) * 1000.0

                self.signals.tick()

                if self.vlog is not None:
                    self.vlog.log_cycle(latency_ms,
                                        self._last_lane.error_px,
                                        self._last_lane.confidence)
                    dist_stop = (self.fsm.sign_distance_mm
                                 if self.fsm.sign_distance_mm is not None
                                 else self.sensor.front_mm)
                    estado = (self.parking.state.name
                              if self._mode == self.Mode.PARKING
                              else self.fsm.state.name)
                    self.vlog.log_stop(dist_stop,
                                       self.motor.current_duty,
                                       self._last_lane.error_px,
                                       estado)
                    self.vlog.log_fsm(estado)

                elapsed = time.monotonic() - now
                sleep   = max(0.0, (1.0 / LOOP_HZ) - elapsed)
                if sleep > 0:
                    time.sleep(sleep)

                self._metrics["loop_count"] += 1

        finally:
            self._shutdown()


    def _update_vision(self) -> None:
        """Process frame + update the FSM with detections."""
        frame = self.camera.get_frame()
        if frame is None:
            return

        self._last_lane = self.lane_pipe.process(frame)
        self._metrics["frame_count"] += 1
        self._metrics["errors_px"].append(self._last_lane.error_px)

        self.sign_det.update_frame(frame)

        self.fsm.lane_error   = self._last_lane.error_px
        self.fsm.lane_conf    = self._last_lane.confidence
        self.fsm.lidar_mm     = self.sensor.front_mm

        stop_like = self.sign_det.has_sign("stop_sign") or self.sign_det.has_sign("red")
        self.fsm.sign_visible = stop_like

        if self.sensor.front_mm is not None:
            self._metrics["lidar_readings"].append(self.sensor.front_mm)

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
        dets = [d for d in dets if d.distance_m is not None]
        if not dets:
            return self.SIGN_ACTIONS.get(self.sign_det.get_detections()[0].label, "")
        closest = min(dets, key=lambda d: d.distance_m)
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
                self._metrics["fsm_states"].append(self.fsm.state.name)
                self._metrics["pid_outputs"].append(self.pid.last_output)
                self._metrics["servo_angles"].append(self.steering.current_angle)
                self._metrics["motor_duties"].append(self.motor.current_duty)
                self._log_autonomous()
                if _DISPLAY:
                    self._render_debug_view(mode_label="AUT")
            case self.Mode.PARKING:
                self.parking.lidar_mm = self.sensor.front_mm
                self.parking.update(dt)
                print(f"\r[PARK] {self.parking.state.name:<16} "
                      f"duty:{self.motor.current_duty:+.0f}%  "
                      f"angle:{self.steering.current_angle:5.1f}°   ",
                      end="", flush=True)
                if _DISPLAY:
                    self._render_debug_view(mode_label="PARK")

    def _log_autonomous(self) -> None:
        """Status line for autonomous mode."""
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
        """Wait for an automatic command (for tests)."""
        self.motor.brake()
        self.steering.set_angle(SERVO_CENTER)
        print(f"\r[STANDBY] Waiting for command... ("
              f"run: sim.set_mode(VehicleSimulator.Mode.AUTONOMOUS))  ",
              end="", flush=True)

    def _do_manual(self) -> None:
        """Manual mode -- on PC it needs an API, not a gamepad."""
        self.motor.brake()
        self.steering.set_angle(SERVO_CENTER)
        print(f"\r[MANUAL] Not available in the simulator (needs a gamepad). "
              f"Switching to AUTONOMOUS...", end="", flush=True)
        self._mode = self.Mode.AUTONOMOUS
        self.fsm.activate()

    def _do_vision(self, dt: float) -> None:
        """Vision debug mode -- PID in simulation, motors OFF."""
        self.motor.brake()
        self.steering.set_angle(SERVO_CENTER)

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
        """Multi-info debug panel."""
        frame = self.camera.get_frame()
        if frame is None:
            return

        lane  = self._last_lane
        dets  = self.sign_det.get_detections()
        lidar = self.sensor.front_mm

        vis = self.lane_pipe.draw_debug(frame, lane)
        for d in dets:
            cv2.rectangle(vis, (d.x1, d.y1), (d.x2, d.y2), (0, 0, 255), 2)
            dist_txt = f" {(d.distance_m or 0)*100:.0f}cm" if d.distance_m else ""
            cv2.putText(vis, f"{d.label} {d.confidence:.0%}{dist_txt}",
                        (d.x1, max(d.y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        BW, BH = 1280, 960
        big = cv2.resize(vis, (BW, BH), interpolation=cv2.INTER_LINEAR)

        mw, mh = 380, 250
        if lane.bev_frame is not None and lane.mask_frame is not None:
            big[0:mh, 0:mw]        = cv2.resize(lane.bev_frame,  (mw, mh))
            big[0:mh, BW - mw:BW]  = cv2.resize(lane.mask_frame, (mw, mh))
            cv2.putText(big, "BEV (bird's-eye)", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(big, "HSV white mask", (BW - mw + 10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        angle_target = max(
            SERVO_MIN, min(SERVO_MAX, SERVO_CENTER - self.pid.last_output)
        )

        self._draw_panel(big, x=10, y=mh + 12, w=440, h=210, lines=[
            f"MODE  : {mode_label}",
            f"err   :{lane.error_px:+7.1f}px  conf:{lane.confidence:.0%}",
            f"P     :{self.pid.last_p:+7.2f}   kp={self.pid.kp:.3f}",
            f"I     :{self.pid.last_i:+7.2f}   ki={self.pid.ki:.3f}",
            f"D     :{self.pid.last_d:+7.2f}   kd={self.pid.kd:.3f}",
            f"corr  :{self.pid.last_output:+7.2f}d -> {angle_target:5.1f}d",
            f"lidar :{lidar:.0f}mm" if lidar else "lidar :---",
        ])

        if dets:
            obj_lines = ["DETECTED OBJECTS:"]
            for d in dets[:5]:
                dist = f" @{(d.distance_m or 0)*100:.0f}cm" if d.distance_m else ""
                obj_lines.append(f"- {d.label} {d.confidence:.0%}{dist}")
        else:
            obj_lines = ["DETECTED OBJECTS:", "- (none)"]
        self._draw_panel(big, x=BW - mw - 10, y=mh + 12, w=mw, h=210, lines=obj_lines)

        try:    fsm_txt = f"FSM:{self.fsm.state.name}"
        except: fsm_txt = ""
        cv2.putText(big, f"{mode_label}  {fsm_txt}  duty:{self.motor.current_duty:+.0f}%",
                    (12, BH - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)

        cv2.namedWindow("TMR 2026 - Simulator Debug", cv2.WINDOW_NORMAL)
        cv2.imshow("TMR 2026 - Simulator Debug", big)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self._running = False
        elif key == ord('a'):
            if self._mode != self.Mode.AUTONOMOUS:
                self._mode = self.Mode.AUTONOMOUS
                self.fsm.activate()
        elif key == ord('s'):
            self._mode = self.Mode.STANDBY
            self.fsm.deactivate()

    @staticmethod
    def _draw_panel(img, x: int, y: int, w: int, h: int, lines: list[str]) -> None:
        """Debug text box (readable in a large window)."""
        ov = img.copy()
        cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, dst=img)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 220, 0), 2)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (x + 12, y + 30 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 220, 0), 2, cv2.LINE_AA)


    def set_mode(self, mode: str) -> None:
        """Change the operating mode."""
        if mode != self._mode:
            print(f"\n[MODE] {self._mode} -> {mode}")
            if _DISPLAY and self._mode in (self.Mode.VISION, self.Mode.AUTONOMOUS):
                if mode not in (self.Mode.VISION, self.Mode.AUTONOMOUS):
                    cv2.destroyAllWindows()
            if mode == self.Mode.VISION or self._mode == self.Mode.VISION:
                self.pid.reset()
            if mode == self.Mode.AUTONOMOUS:
                self.fsm.activate()
            else:
                self.fsm.deactivate()
        self._mode = mode

    def run_autonomous_test(self, duration_s: float) -> dict:
        """
        Run AUTONOMOUS mode for N seconds and return metrics.
        Useful for Phase 1 validation.
        """
        print(f"\n[TEST] Starting autonomous test ({duration_s}s)...")
        self.set_mode(self.Mode.AUTONOMOUS)
        start = time.monotonic()

        while time.monotonic() - start < duration_s and self._running:
            time.sleep(0.001)

        self.set_mode(self.Mode.STANDBY)
        return self.get_metrics()

    def get_metrics(self) -> dict:
        """Return collected metrics for analysis."""
        elapsed = time.monotonic() - self._start_time
        return {
            "elapsed_s": elapsed,
            "loop_count": self._metrics["loop_count"],
            "frame_count": self._metrics["frame_count"],
            "avg_loop_hz": self._metrics["loop_count"] / elapsed if elapsed > 0 else 0,
            "errors_px": self._metrics["errors_px"],
            "pid_outputs": self._metrics["pid_outputs"],
            "servo_angles": self._metrics["servo_angles"],
            "motor_duties": self._metrics["motor_duties"],
            "lidar_readings": self._metrics["lidar_readings"],
            "fsm_states": self._metrics["fsm_states"],
        }


    def _handle_signal(self, signum, _frame) -> None:
        print(f"\n[SYS] Signal {signum} received -> shutting down...")
        self._running = False

    def _shutdown(self) -> None:
        print("\n[SYS] Shutting down the simulator...")
        if _DISPLAY:
            cv2.destroyAllWindows()
        self.fsm.deactivate()
        self.motor.brake()
        time.sleep(0.1)
        self.sign_det.stop()
        self.sim.close()
        print("[SYS] Shutdown complete.")

        metrics = self.get_metrics()
        print(f"\n[METRICS] Loop ran {metrics['loop_count']} times in {metrics['elapsed_s']:.1f}s")
        print(f"[METRICS] Frames processed: {metrics['frame_count']}")
        print(f"[METRICS] Lidar readings: {len(metrics['lidar_readings'])}")

        if self.vlog is not None:
            paths = self.vlog.save_all()
            print("\n[VALIDATION] CSVs generated:")
            for k, v in paths.items():
                print(f"   - {v}")
            self.vlog.print_scoreboard()


if __name__ == "__main__":
    sim = VehicleSimulator()
    try:
        sim.run()
    except KeyboardInterrupt:
        print("\n[USER] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
