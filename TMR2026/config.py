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
# Mechanical steering offset, in servo degrees, applied AFTER the inversion in
# SteeringDriver._physical(). Calibrated on the car 2026-07-27 with
# tools/auto_trim.py: the chassis pulls left with the servo centred (confirmed by
# pushing the car by hand, motor off), and no controller can absorb that inside a
# run -- a b-degree bias leaves a standing b/Kp px error, and the integrator would
# need ~40 s against a 3-8 s run.
#
# CAUTION: the convergence run was noisy. Drift went 28, 47, 73, 46, 22, 28, 33,
# 7 px/s as trim walked 0 -> -6.9, i.e. it never crossed zero and the final
# reading may be luck rather than convergence. Re-run auto_trim to confirm, and
# treat a value near the +/-8 limit as a sign the linkage itself needs adjusting.
SERVO_TRIM_DEG      = -7.5

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

# City track, tape-measured 2026-07-28. This REPLACES the 3.46 m track the
# earlier numbers came from; nothing geometric here is inherited.
#
# Two lines, not three: white tape on the RIGHT edge and a dashed centre. The
# opposite lane has no tape -- its edge is the mat against the tile floor. That
# is fine, because with the car correctly inside the right lane the pipeline
# already prefers the `dashed <-> right solid` pair (see LANE_DRIVEN_WIDTH_M).
#
# Display only (tools/track_geometry.py); NOT tape-measured. The sign was
# measured at 1.50 m from the start line and the mat ends just past it.
ROAD_TOTAL_LENGTH_M    = 1.60
ROAD_LEFT_TO_DASHED_M  = 0.35    # dashed centre -> unpainted mat edge
ROAD_DASHED_TO_RIGHT_M = 0.31    # dashed centre -> inner edge of the white tape
ROAD_STOP_SIGN_AT_M    = 1.50    # start line -> sign post (unchanged by luck)
ROAD_PARKING_GAP_M     = 0.290   # stale: parking cannot run, reverse is dead

LANE_WIDTH_M = ROAD_LEFT_TO_DASHED_M + ROAD_DASHED_TO_RIGHT_M

TOF_TIMING_BUDGET_US = 20_000
TOF_MAX_RANGE_MM     = 1_200
TOF_POLL_INTERVAL_S  = 0.020

# Lane-following gains. ONE definition -- main.py, main_simulator.py and
# tools/bench_braking_physical.py all read these now. They used to redefine
# PID_KP=0.08 locally, agreeing with each other but not with this file.
#
# Kp 0.08 deg/px is not arbitrary: pure pursuit at small angles gives
# delta ~= 2*L*x/Ld^2, so with wheelbase L=0.31 m and the BEV's 679.6 px/m,
#     Ld 0.6 m -> 0.145      Ld 0.8 m -> 0.082      Ld 1.0 m -> 0.052
# 0.08 therefore matches a lookahead of ~0.8 m, which is what
# LANE_AIM_WINDOW_FRAC now actually measures at.
#
# Ki and Kd are ZERO on purpose:
#   Ki -- with SERVO_TRIM_DEG calibrated there is no standing bias left to
#         integrate away, and on a 3-8 s run an integrator can only wind up.
#   Kd -- the lookahead already supplies the damping; a derivative on top just
#         amplifies pixel noise into servo chatter.
STEER_KP = 0.08
STEER_KI = 0.0
STEER_KD = 0.0

# Where in the lane the car should sit, as a fraction from the far (unpainted)
# edge across the full road width. The road is 35.0 cm + 31.0 cm = 66.0 cm, so
# the centre of the RIGHT lane is at
#     (0.35 + 0.31/2) / 0.66 = 0.765
# On the city track this is nearly moot: with only two lines the wide pair is
# never found, and the driven-lane pair aims at a hard-coded 0.50. Kept correct
# so the outer-pair branch stays right if a left line is ever taped down.
# 0.50 ON THIS TRACK, not 0.765. Measured 2026-07-28 with the camera finally
# aimed at the road: the dashed centre lands at BEV x=185 and the right tape at
# x=460, a separation of 275 px for a lane that is 31 cm wide. That is 8.9 px/cm,
# where BEV_SCALE_PX_PER_CM claims 384/66 = 5.82.
#
# The scale is DEFINED, not measured: `384 / (LANE_WIDTH_M * 100)` simply asserts
# that the carriageway fills the destination window. That was verified on the
# 56.5 cm track and quietly stopped being true at 66 cm, because the homography
# is a fixed trapezoid in frame coordinates and knows nothing about road width.
#
# The consequence is a misclassification, not a rounding error: lane_px comes out
# 180 and road_px 384, so a real 275 px lane reads as 28 % narrow for the
# carriageway (inside the +/-40 % window) and 53 % wide for the lane (outside
# it). The pipeline then aims at 0.765 ACROSS THE DRIVEN LANE instead of across
# the road -- 8 cm right of centre, every frame, which is the +74 px this reads.
#
# 0.50 is correct under either branch here, because the only pair this track can
# produce is dashed <-> right tape and the car belongs in the middle of it. The
# carriageway pair cannot even form: 66 cm at the true scale is ~585 px against a
# 460 px valid window, so the far edge is outside the view whenever the car is
# actually in its lane.
LANE_RIGHT_BIAS = 0.50

# Width of the lane the car actually drives in (dashed centre -> right tape).
# The sliding windows return one of two pairs:
#   far edge <-> right tape  = 0.660 m -> 384 px, aim at LANE_RIGHT_BIAS
#   dashed   <-> right tape  = 0.310 m -> 180 px, aim at 50 %
# The city track has NO tape on the far edge, so in practice only the 180 px
# pair appears -- which is the one the car needs anyway: a vehicle correctly
# inside the right lane puts the far edge outside the BEV window regardless.
LANE_DRIVEN_WIDTH_M = 0.31

# Pure-pursuit lookahead, as a fraction of the bird's-eye view height measured
# from the bottom. 0.70 aims near the far end of the view (~0.8 m ahead), which
# is what STEER_KP is sized for. Raise if the car weaves, lower if it reacts
# late or cuts corners.
LANE_AIM_WINDOW_FRAC = 0.70

# Heading feedforward (Stanley-style) on top of the lateral PID, in degrees of
# servo per pseudo-degree of lane lean (LaneResult.heading). Why it exists: the
# lateral term only reacts AFTER the car has translated sideways, so any yaw --
# from the launch kick, a floor seam, or steering trim -- turns into a drift the
# PID chases instead of preventing. The integrator cannot help within a run: at
# Ki=0.002, cancelling a 6 deg trim bias needs ~40 s of accumulation, and a
# braking run lasts 3 s. Heading feedback closes that hole.
#
# DEFAULT 0.0 = DISABLED, deliberately: the sign of the heading estimate has not
# been confirmed on the car yet, and if it is inverted this becomes positive
# feedback that throws the car off the track. Enable procedure:
#   1. place the car on the lane rotated ~15 deg with the nose to the LEFT
#   2. python tools/diag_track.py --frames 8  -> heading column must be POSITIVE
#   3. nose to the RIGHT -> heading must be NEGATIVE
#   4. only then set 1.5 here, and raise toward 2.5 if it still weaves
STEER_HEADING_GAIN = 0.0

# Steering deadband: inside this lane-error band the servo holds CENTRE and does
# not chase pixels (with the chassis trim calibrated, centre IS straight). It
# re-engages when the error leaves the band and keeps correcting until it is
# back inside the smaller inner band -- the hysteresis pair prevents chatter at
# the boundary. 25 px = 3.7 cm at the BEV scale of 6.8 px/cm.
LANE_DEADBAND_PX      = 25.0
LANE_DEADBAND_EXIT_PX = 12.0

VEL_STOP_KP = 0.035
VEL_STOP_KI = 0.001
VEL_STOP_KD = 0.008

# Lowest duty that actually advances the car. Measured 2026-07-27 by driving the
# H-bridge directly, 1.4 s per burst, judged by the SIGN DISTANCE (YOLO bbox
# through the pinhole model) -- never by frame differencing, which reported
# motion at every duty on a stationary car because chassis vibration displaces
# each frame by a few pixels at random:
#     20 % -> +20.8 cm    25 % -> +24.6 cm    30 % -> +22.1 cm
#     104 cm to 36 cm of approach in three bursts
#
# WITH the spare battery sitting on the chassis this same car would not move at
# 95 %. The load, not the duty, was the whole problem: every "the motor has no
# torque" conclusion earlier in the day was measured under that extra weight.
# Keep the chassis light.
MOTOR_MIN_MOVE_PWM = 20.0

# REVERSE IS DEAD ON THIS CHASSIS, re-confirmed unladen with sign-distance
# measurement: 25/35/45/60/80/95 % all left the sign at 36.5 cm exactly, zero
# variation, while forward at 20 % advanced 20 cm. The LPWM leg of the H-bridge
# does not drive the motor at any power.
#
# Consequences: ParkingFSM cannot execute (its manoeuvre reverses into the bay),
# and the car cannot back up to a start line on its own. Check the LPWM side of
# the IBT-2 and its wiring before relying on either.
MOTOR_REVERSE_OK = False

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
# HOW TO RE-MEASURE -- and the trap in it. Centre the car in the lane by hand, run
#   python tools/diag_track.py --frames 12
# and use the SETTLED error_px (the EMA ramps over the first frames, so the run mean
# understates it). Positive means the pipeline reads high.
#
# Only trust the reading when the reported line separation is close to the expected
# 384 px. This value moved four times on 2026-07-26 -- +50, +44, +30.6, +58.7 -- and
# the reason was not the camera. The bias depends on WHICH pair of lines the sliding
# windows locked onto: at separation 327 px it was tracking something narrower than
# the lane (most likely the dashed centre line instead of the left solid) and the
# centre of that pair sits somewhere else entirely. Measured back to back, centred
# both times:
#     separation 327 px -> raw error +58.7
#     separation 385 px -> raw error +28.5   <- the true full-lane lock
# So a calibration taken during a partial or mis-paired lock is simply wrong, and no
# single additive constant can serve both lock modes. Check the separation first.
# Re-measured 2026-07-27 with the car centred in the RIGHT LANE and the target
# rule at LANE_RIGHT_BIAS=0.74. The previous 28.5 was taken with the target at
# the centre of the whole ROAD (bias 0.50), which is a different physical place,
# so it no longer applied. Settled reading was +30.0 px on top of that 28.5.
# Re-measured on the city track 2026-07-28. 58.5 was the 3.46 m track's value and
# could not survive this one (road 56.5 -> 66.0 cm, driven lane 29 -> 31 cm).
#
# The run that produced this still had 58.5 loaded and reported a SETTLED
# -35.4 px with the car centred by hand, so the raw standing bias is
#     58.5 - 35.4 = 23.1 px
# Taken from the settled value, not the printed mean of -33.3: the EMA starts at
# -16.1 and converges over ~8 frames, so the mean is contaminated by its own
# transient. Frames 7-19 all read -35.0..-35.7.
#
# That reading transfers even though it was taken under the OLD constants,
# because both lines were found in 20/20 frames and the separation (184 px) put
# BOTH configurations on the same branch -- the driven-lane pair, which aims at
# the measured midpoint with a hard-coded 0.5. Neither LANE_RIGHT_BIAS nor
# lane_px enters that path, so nothing stale leaked into the number.
#
# TO RE-MEASURE: centre the car in the RIGHT lane by hand, then
#     python tools/diag_track.py --frames 20
# Confirm the separation is IN BAND and the mask is not degenerate FIRST -- a
# calibration taken during a partial or mis-paired lock is simply wrong (see the
# 327 px vs 385 px note above). A correct value now reads ~0.0 settled.
# RETRACTED 2026-07-28, same day: 23.1 was measured against FURNITURE, not lines.
# /tmp/diag/2_bev.png from the run that produced it shows the bird's-eye view
# filled with a doorway, a wall and a wooden bench -- no road at all. The camera
# is aimed roughly LEVEL, while this file assumes CAMERA_TILT_DEG = 10 down, so
# the BEV's top edge (row 264 of 480) samples the horizon instead of ~1 m ahead.
# The sliding windows then locked on bright vertical edges of the room, which is
# why three runs "found both lines" at 100 % confidence with separations of 184,
# 316 and 386 px -- three different pairs of furniture.
#
# Fix the camera aim first, then re-measure. Do NOT compensate in software: the
# lens geometry in the paper (blind distance, rows the octagon occupies) all
# assumes 22 cm and 10 deg, and patching the trapezoid would bake the
# misalignment into the numbers being published.
#
# RE-MEASURED after the camera was tilted down, with LANE_RIGHT_BIAS at 0.50:
# +8.5 px settled, std 1.1, 100 % confidence, both lines in 20/20 frames. This
# one is on real lines -- checked against the mask's own column bands, which put
# left_x=151 inside the dashed run (132-169) and right_x=510 inside the tape
# (482-512).
LANE_ERROR_OFFSET_PX = 8.5

STOP_BRAKE_START_MM  = 700
STOP_TARGET_MM       = 270
STOP_TOLERANCE_MM    = 30
STOP_WAIT_SEC        = 5.0
STOP_LED_BLINK_HZ    = 2.0

# Measured on the city track's sign 2026-07-28: the octagon is 9.0 cm across the
# flats, not 8.5. This feeds the pinhole distance directly, so the old value was
# reporting every sign 5.9 % CLOSER than it was (d = h_real*f/h_px), which made
# the brake gate fire ~19 mm early. Re-measure after any change of sign.
STOP_SIGN_REAL_HEIGHT_M  = 0.09
# Post height under the octagon is estimated from the photographs (~6 cm), not
# taped. Only tools/track_geometry.py reads this, for the "leaves the bottom of
# the frame" figure -- it is not in the control path.
STOP_SIGN_TOTAL_HEIGHT_M = 0.15
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
