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

# Measured with tools/tune_exposure.py, 2026-07-26, daylight, sign at ~1.5 m.
# gain 22.0 was set the night before under a phone flashlight and it BLINDED the
# sign detector in daylight: the red octagon clipped to white-pink (saturation
# 23.7, 100% of its pixels at 255) and `stop` confidence read 0.000. Lane
# detection stayed at 100% throughout, because the track is dark plastic and the
# lane lines are R=G=B -- so a healthy lane lock proves nothing about exposure.
#
# The sweep, ranked by `stop` confidence:
#    4 ms g1.0  -> 0.846, sat 148, frame V   4.8  (sign great, track black)
#   16 ms g2.0  -> 0.804, sat 109, frame V  31.5
#   33 ms g4.0  -> 0.784, sat  71, frame V  87.6  <- chosen
#   auto (AE)   -> 0.716, sat  39, frame V 124.4  (AE meters the dark track and
#                                                  overexposes the bright sign)
#   33 ms g22.0 -> 0.000, NOT SEEN,        frame V 190.2
#
# The two goals pull against each other: the sign wants less light, the dark
# track wants more. 33 ms / g4.0 is the compromise that keeps `stop` well clear
# of its 0.55 gate while leaving the track bright enough for the HSV white
# filter (V_min 130). Re-run the sweep if the lighting changes materially --
# these are the correct values for THIS light, not for all light.
CAMERA_EXPOSURE_US = 33000
CAMERA_GAIN        = 4.0
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
LANE_ERROR_OFFSET_PX = 0.0

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
