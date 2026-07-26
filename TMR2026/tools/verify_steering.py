"""Verify the steering direction through the PRODUCTION code path.

The previous version of this script duplicated the error-to-angle formula, and then
the real one was fixed while the copy was not -- so it measured the old, wrong
behaviour and reported "ruedas IZQ" for a case that the fixed FSM now steers right.
Duplicating the expression under test is exactly how that happens, so this drives
the real AutonomousFSM._apply_steering() instead.

The motor is a stub that refuses to do anything, so the car cannot drive. The
servo is real.

What the last run established, physically: moving the car RIGHT took the error from
+64.6 to +12.3, i.e. the error DECREASES as the car moves right. So error > 0 means
the car sits LEFT of the lane centre and must steer RIGHT to recover.

Correct behaviour, then:
    car LEFT  of centre (error positive, large) -> wheels RIGHT (angle > 90)
    car RIGHT of centre (error smaller/negative) -> wheels LEFT  (angle < 90)
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.camera_stream import CameraStream
from vision.lane_pipeline import LanePipeline
from control.pid_controller import PIDController
from control.fsm import AutonomousFSM, FSMState
from hardware.steering_driver import SteeringDriver
from config import SERVO_CENTER_ANGLE, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE

PID_KP, PID_KI, PID_KD = 0.08, 0.002, 0.025


class NoMotor:
    """Swallows every motor call so the wheels cannot turn.

    __getattr__ rather than a hand-listed API: naming the methods by guess already
    cost a run when the FSM called set_speed() and the stub only had set_duty().
    """
    current_duty = 0.0
    MAX_DUTY = 0.0

    def __getattr__(self, _name):
        return lambda *a, **k: None


cam = CameraStream(width=640, height=480, fps=30)
cam.start()
lp  = LanePipeline()
st  = SteeringDriver()
pid = PIDController(kp=PID_KP, ki=PID_KI, kd=PID_KD, setpoint=0.0,
                    output_limits=(SERVO_MIN_ANGLE - SERVO_CENTER_ANGLE,
                                   SERVO_MAX_ANGLE - SERVO_CENTER_ANGLE))
fsm = AutonomousFSM(motor=NoMotor(), steering=st, pid=pid)
fsm.activate()

PHASES = [
    (12.0, "CENTRA el carro, no lo toques"),
    (20.0, ">>> MUEVE el carro a la IZQUIERDA (5-8 cm) -- ruedas deben ir DERECHA"),
    (20.0, "<<< MUEVE el carro a la DERECHA   (5-8 cm) -- ruedas deben ir IZQUIERDA"),
]

try:
    for dur, label in PHASES:
        print(f"\n=== {label} ===", flush=True)
        t0 = last = time.monotonic()
        while time.monotonic() - t0 < dur:
            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.02); continue
            lane = lp.process(frame)
            now = time.monotonic()
            dt = max(1e-3, now - last); last = now

            fsm.lane_error = lane.error_px
            fsm.lane_conf  = lane.confidence
            fsm.lidar_mm   = None
            fsm.sign_visible = False          # keep it in CRUCERO, no braking
            fsm.sign_distance_mm = None
            fsm.update(dt)                    # the real path sets the servo

            a = st.current_angle
            side = "IZQ" if a < SERVO_CENTER_ANGLE - 0.5 else \
                   ("DER" if a > SERVO_CENTER_ANGLE + 0.5 else "recto")
            print(f"  err {lane.error_px:+7.1f}px conf {lane.confidence:4.0%}  "
                  f"corr {pid.last_output:+6.2f}  angulo {a:6.2f} -> ruedas {side}"
                  f"   [{fsm.state.name}]      ", end="\r", flush=True)
            time.sleep(0.10)
        print()
finally:
    fsm.deactivate()
    st.set_angle(SERVO_CENTER_ANGLE)
    time.sleep(0.4)
    cam.stop()
    print("\nservo centrado, camara liberada.")
