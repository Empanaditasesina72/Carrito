"""Global parameters for the TMR 2026 vehicle (single source of truth).

All physical values, GPIO pins and PID gains live here. Do not import real
hardware from this module.
"""

PIN_MOTOR_RPWM = 18
PIN_MOTOR_LPWM = 13
MOTOR_PWM_FREQ = 1000

PIN_LED_STOP   = 25
PIN_LED_STATUS = 26

PIN_LED_TURN_LEFT  = 17
PIN_LED_TURN_RIGHT = 5
PIN_LED_BRAKE      = 6
SIGNAL_BLINK_HZ    = 2.0


USE_TOF_SENSORS     = False
PIN_TOF_XSHUT_FRONT = 24
PIN_TOF_XSHUT_REAR  = 27
TOF_ADDR_FRONT      = 0x30
TOF_ADDR_REAR       = 0x29

PCA9685_I2C_ADDR   = 0x40
PCA9685_PWM_FREQ   = 50
SERVO_CHANNEL      = 15

SERVO_MIN_PULSE_US  = 500
SERVO_MAX_PULSE_US  = 2500
SERVO_CENTER_ANGLE  = 90.0
SERVO_MIN_ANGLE     = 58.0
SERVO_MAX_ANGLE     = 122.0
SERVO_TRIM_DEG      = 0.0

STEERING_INVERTED   = True

IMX500_MODEL_PATH = "/usr/share/imx500-models/imx500_network_efficientdet_lite0_pp.rpk"

USE_IMX500_NPU     = False
IMX500_RPK_PATH    = "weights/tmr_signs_imx500.rpk"
IMX500_LABELS_PATH = "weights/tmr_signs_imx500_labels.txt"
IMX500_CONF        = 0.55

USE_DRIVE_NET      = False
DRIVE_NET_WEIGHTS  = "weights/drive_net.pt"
DRIVE_NET_CONF_MIN = 0.30

CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30

CAMERA_AWB_MODE   = 4
CAMERA_CONTRAST   = 1.5
CAMERA_SATURATION = 1.0

# Manual exposure overrides. BOTH None = the adaptive loop below runs, which is
# the intended configuration; set both to numbers only to pin the sensor and
# disable adaptation. Either one alone overrides just that control at startup.
#
# Do not pin these casually. Every fixed value tried on 2026-07-26 failed at some
# point in the same day, because the sign and the lane want opposite things and the
# light moved 4 stops between 14:00 and 15:40:
#
#    4 ms g1.0  -> sign 0.846, sat 148, V   4.8  (sign great, track pitch black)
#   16 ms g2.0  -> sign 0.804, sat 109, V  31.5
#   33 ms g4.0  -> sign 0.784, sat  71, V  87.6  (fine at 14:00; by 15:40 the same
#                                                 setting gave a 0.0 % lane mask)
#   33 ms g22.0 -> sign 0.000, NOT SEEN,  V 190.2 (red clipped to white-pink)
#
# Note what made the gain-22 case so hard to spot: lane detection reported 100 %
# confidence the whole time it was happening, because the track is dark plastic and
# the lane lines are R=G=B, so the lane pipeline barely cares about exposure. A
# healthy lane lock is NOT evidence that the camera is configured correctly.
#
# tools/tune_exposure.py sweeps candidates and reports sign confidence, sign
# saturation, lane confidence and mask fill together. Rank by saturation.
CAMERA_EXPOSURE_US = None
CAMERA_GAIN        = None

# Continuous exposure adaptation (vision/camera_stream.py:_adapt_exposure).
#
# Why not just leave the sensor's own AE running: AE targets mid-grey over the
# whole frame, and this frame is mostly dark plastic track, so it opens up until
# the one bright object that matters clips. Measured 2026-07-26 in daylight, the
# red STOP octagon reached S 23.7 with 100 % of its pixels at 255 and `stop`
# confidence read exactly 0.000. Meanwhile a value pinned for one light level
# cannot survive the day: the same 33 ms / gain 4.0 that gave 100 % lane
# confidence at 14:00 left a 0.0 % mask -- nothing at all -- by 15:40, a 3.4x drop
# in scene brightness.
#
# So this closes the loop on OUR objective instead of the sensor's: keep the frame
# bright enough for the lane's white lines while keeping clipping low enough that
# the sign's red survives. Both bounds are measured, not guessed:
#
#   V mean 87.6 -> lane 100 %, fill 9.1 %      (14:00, sunlit)
#   V mean 42.4 -> lane 100 %, fill 6.8 %      (~15:00, the lowest that worked)
#   V mean 12.6 -> lane   0 %, fill 0.0 %      (15:40, dead)
#   V mean 190  -> 23.6 % clipped, sign 0.000  (gain 22, blinded)
#   V mean 118  -> 14.6 % clipped, sign lost at imgsz 320
#
# Set CAMERA_EXPOSURE_US / CAMERA_GAIN to numbers to pin them and disable this.
CAMERA_ADAPT_ENABLED    = True
CAMERA_ADAPT_INTERVAL_S = 1.5     # how often to re-evaluate
CAMERA_ADAPT_V_LO       = 45.0    # below this the lane loses its lines
CAMERA_ADAPT_V_HI       = 95.0    # above this clipping starts threatening the sign
CAMERA_ADAPT_CLIP_MAX   = 3.0     # % of pixels at 250+; overrides the V band
CAMERA_ADAPT_EXP_MAX_US = 33000   # 30 fps frame duration is the ceiling
CAMERA_ADAPT_EXP_MIN_US = 200
CAMERA_ADAPT_GAIN_MAX   = 16.0    # what this sensor's own AE reached in the dark
CAMERA_ADAPT_GAIN_MIN   = 1.0
CAMERA_SHARPNESS  = 4.0
CAMERA_DENOISE    = 2
CAMERA_BUFFERS    = 6

DETECTION_CONFIDENCE = 0.28

DETECTION_MIN_FRAMES = 2

CLASSES_OF_INTEREST = {
    "stop sign"    : "STOP",
    "traffic light": "SEMAFORO",
    "person"       : "PERSONA",
    "car"          : "AUTO",
}

WHEELBASE    = 0.310
TRACK_WIDTH  = 0.172
CAR_LENGTH   = 0.280
CAR_WIDTH    = 0.190
CAMERA_HEIGHT_M = 0.22
CAMERA_TILT_DEG = 10.0

MAX_STEERING_ANGLE_DEG = 35.0

ROAD_TOTAL_LENGTH_M    = 3.46
ROAD_LEFT_TO_DASHED_M  = 0.275
ROAD_DASHED_TO_RIGHT_M = 0.290
ROAD_STOP_SIGN_AT_M    = 1.50
ROAD_PARKING_GAP_M     = 0.290

LANE_WIDTH_M = ROAD_LEFT_TO_DASHED_M + ROAD_DASHED_TO_RIGHT_M

TOF_TIMING_BUDGET_US = 20_000
TOF_MAX_RANGE_MM     = 1_200
TOF_POLL_INTERVAL_S  = 0.020

STEER_KP = 0.09
STEER_KI = 0.002
STEER_KD = 0.025

VEL_STOP_KP = 0.035
VEL_STOP_KI = 0.001
VEL_STOP_KD = 0.008

SPEED_STRAIGHT   = 22
SPEED_CURVE      = 15
SPEED_APPROACH   = 10

CURVE_THRESHOLD_RAD = 0.30

LANE_LOST_THRESHOLD_PX = 280

LANE_MIN_CONFIDENCE = 0.20

# Standing lateral bias of the lane pipeline, in BEV pixels, subtracted from the
# raw error so that a car centred in its lane reads ~0. This is a MECHANICAL
# calibration: the camera is not exactly on the chassis centreline and the BEV
# trapezoid is not perfectly symmetric about the lens axis. At
# BEV_SCALE_PX_PER_CM = 6.8 px/cm, every 68 px is 10 cm the follower would
# otherwise hold off-centre on purpose.
# Re-measure after remounting the camera: centre the car in the lane by hand, run
#   python tools/diag_track.py --frames 12
# and put the reported mean error_px here. Positive means the pipeline reads high.
LANE_ERROR_OFFSET_PX = 58.7

STOP_BRAKE_START_MM  = 700
STOP_TARGET_MM       = 270
STOP_TOLERANCE_MM    = 30
STOP_WAIT_SEC        = 5.0
STOP_LED_BLINK_HZ    = 2.0

STOP_SIGN_REAL_HEIGHT_M  = 0.085
STOP_SIGN_TOTAL_HEIGHT_M = 0.175
CAMERA_FOCAL_LENGTH_PX   = 490.0

EMERGENCY_STOP_MM = 120

PARK_SIDE           = "left"
PARK_SEARCH_SPEED   = 15
PARK_MANEUVER_SPEED = 10
PARK_MIN_GAP_MM     = 240
PARK_TARGET_GAP_MM  = 290

PARK_OVERSHOOT_SEC        = 1.2
PARK_REVERSE_LOCK_SEC     = 2.5
PARK_REVERSE_STRAIGHT_SEC = 1.0

PARK_GAP_CAMERA_MIN_SEC   = 0.4
PARK_GAP_CAMERA_ZONE      = 0.55

OVERTAKE_MIN_BBOX_AREA    = 2500
OVERTAKE_LANE_RATIO       = 0.35
OVERTAKE_TRIGGER_Y_MIN    = 300

OVERTAKE_LEFT_SEC    = 1.8
OVERTAKE_PASS_SEC    = 2.2
OVERTAKE_RETURN_SEC  = 1.8
OVERTAKE_STEER_DEG   = 20.0

BTN_MANUAL     = 0
BTN_VISION     = 1
BTN_AUTONOMOUS = 2
BTN_PARKING    = 3
BTN_EMERGENCY  = 9

BTN_BACK_TO_MANUAL = BTN_MANUAL
BTN_VISION_TEST    = BTN_VISION

AXIS_STEER    = 0
AXIS_THROTTLE = 5
AXIS_BRAKE    = 2

JOYSTICK_DEADBAND = 0.08
TRIGGER_DEADBAND  = 0.05

CROSSWALK_STOP_SEC    = 3.0
CROSSWALK_WHITE_RATIO = 0.55

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.join(BASE_DIR, "logs")
