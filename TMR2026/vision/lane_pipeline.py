"""Lane-detection pipeline: BEV + HSV + Sliding Windows.

Full pipeline:
  1. ROI: crop the lower half of the frame (ignore sky/upper noise).
  2. Bird's-Eye View: perspective transform to a top-down view.
  2b. Discard the columns outside the destination rectangle. The homography is
      only defined between the trapezoid and BEV_DST_RATIO; columns beyond it are
      lateral extrapolations that sample whatever lies past the track edge, so
      they are zeroed before the histogram sees them.
  3. Morphological top-hat: keep bright structures narrower than the kernel, so
     lane lines survive and the off-track floor -- which is BRIGHTER than any
     line -- does not. Replaces an HSV filter that rejected the dashed and left
     lines outright: dim lines read as highly saturated (S=(max-min)/max), so a
     saturation gate discards exactly the ones hardest to see. See _lane_mask().
  4. Morphology: remove speckle noise (specular highlights of black plastic).
  5. Sliding Windows: find left and right lane centres from bottom to top.
  6. Compute the steering error relative to the frame centre.
  7. Temporal smoothing (EMA) to reduce servo oscillation.

BEV calibration:
  The SRC points must be calibrated by placing the car on the lane and
  adjusting until the white lines look vertical in the BEV view.
  Change BEV_SRC_RATIO on your instance or use calibrate_bev().

Performance note:
  At 640x480 this pipeline takes ~8-12 ms on the Pi 5 (no GPU acceleration).
"""

from __future__ import annotations

import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from config import LANE_WIDTH_M
except ImportError:
    LANE_WIDTH_M = 0.54

try:
    from config import LANE_ERROR_OFFSET_PX as _CFG_LANE_ERROR_OFFSET_PX
except ImportError:
    _CFG_LANE_ERROR_OFFSET_PX = 0.0

try:
    from config import (LANE_RIGHT_BIAS as _CFG_RIGHT_BIAS,
                        LANE_AIM_WINDOW_FRAC as _CFG_AIM_FRAC,
                        LANE_DRIVEN_WIDTH_M as _CFG_DRIVEN_WIDTH_M)
except ImportError:
    _CFG_RIGHT_BIAS, _CFG_AIM_FRAC, _CFG_DRIVEN_WIDTH_M = 0.74, 0.70, 0.290


@dataclass
class LaneResult:
    """Result of the lane-detection pipeline."""
    error_px:    float
    confidence:  float
    # Lane lean in pseudo-degrees; >0 = lines lean right going up = the car's
    # nose points left of the road and must steer right. 0.0 when unknown.
    heading:     float = 0.0
    left_x:      Optional[int] = None
    right_x:     Optional[int] = None
    bev_frame:   Optional[np.ndarray] = None
    mask_frame:  Optional[np.ndarray] = None


class LanePipeline:
    """
    Lane detector for a glossy-black track with white lines (~40 cm).

    Key parameters to calibrate on the track:
      bev_src_ratio: perspective-trapezoid points (fraction of the frame)
      hsv_white_s_max: max saturation to accept white (rejects grey reflections)
      hsv_white_v_min: min brightness (rejects shadows)
    """

    BEV_SRC_RATIO = np.float32([
        [0.05, 1.00],
        [0.95, 1.00],
        [0.62, 0.55],
        [0.38, 0.55],
    ])
    BEV_DST_RATIO = np.float32([
        [0.20, 1.00],
        [0.80, 1.00],
        [0.80, 0.00],
        [0.20, 0.00],
    ])
    BEV_SCALE_PX_PER_CM = 384.0 / (LANE_WIDTH_M * 100.0)

    # How far beyond the destination window a column may still be trusted, as a
    # fraction of frame width. See the note where _valid_x0/_valid_x1 are built.
    # 0.0 crops hard at the lane edge and will drop a line whenever the car is a
    # little off centre; too large and the homography's lateral extrapolations
    # come back. 0.06 was chosen from measured line positions.
    BEV_VALID_MARGIN_RATIO = 0.06
    LANE_WIDTH_TOL = 0.40

    HSV_WHITE_LO = np.array([  0,  0, 130])
    HSV_WHITE_HI = np.array([179, 60, 255])

    # Top-hat line extraction. Kernel must be WIDER than a lane line (~25 px in
    # the bird's-eye view) and NARROWER than the off-track floor slab. 41 px was
    # picked by sweep: at 41/30 it returns exactly the three real lines with no
    # spurious floor stripe; 81 px starts splitting a line in two, and a
    # threshold of 20 lets the floor edge back in.
    TOPHAT_ENABLED = True
    TOPHAT_W       = 41
    TOPHAT_MIN     = 30
    TOPHAT_V_FLOOR = 35     # absolute floor, so noise on a black frame is not a mask
    MIN_FILL_FOR_LOCK = 1.5  # % of the view; below this, confidence is capped at 0.5

    # Legacy HSV path, kept behind TOPHAT_ENABLED=False. Adaptive V_min from the
    # view's histogram, with HSV_WHITE_LO as the no-contrast fallback floor.
    ADAPTIVE_WHITE = True
    V_ADAPT_PCTL   = 94.0    # lines cover only a few percent of the bird's-eye view
    V_ADAPT_FLOOR  = 55.0    # never threshold below this, however dark the frame
    V_ADAPT_CEIL   = 210.0   # nor above it, however blown out
    MIN_CONTRAST   = 25.0    # percentile must beat the median by this to be lines

    N_WINDOWS  = 9
    WIN_MARGIN = 70
    MIN_PIX    = 60

    EMA_ALPHA  = 0.45

    RIGHT_BIAS = _CFG_RIGHT_BIAS

    # Which window row the steering error is measured at, as a fraction from the
    # bottom of the bird's-eye view. This is the pure-pursuit lookahead. See the
    # AIM ROW note in _sliding_windows(). Raise it (aim further) if the car
    # weaves; lower it if it cuts corners or reacts too late.
    AIM_WINDOW_FRAC = _CFG_AIM_FRAC

    def __init__(
        self,
        frame_w: int = 640,
        frame_h: int = 480,
        debug: bool = False,
        right_bias: float = RIGHT_BIAS,
        roi_frac: float = 0.5,
        bev_src_ratio=None,
        hsv_white_lo=None,
        hsv_white_hi=None,
        error_offset_px=None,
    ):
        self._w     = frame_w
        self._h     = frame_h
        self._debug = debug
        self._right_bias = max(0.0, min(1.0, float(right_bias)))

        # Standing lateral bias to subtract from the raw error, in BEV pixels.
        # Defaults to config's LANE_ERROR_OFFSET_PX; pass explicitly to override
        # per instance (the simulator has no mechanical offset, so it uses 0).
        self.error_offset_px = (float(error_offset_px)
                                if error_offset_px is not None
                                else float(_CFG_LANE_ERROR_OFFSET_PX))

        if bev_src_ratio is not None:
            self.BEV_SRC_RATIO = np.float32(bev_src_ratio)

        if hsv_white_lo is not None:
            self.HSV_WHITE_LO = np.array(hsv_white_lo)
        if hsv_white_hi is not None:
            self.HSV_WHITE_HI = np.array(hsv_white_hi)

        self._roi_y = int(frame_h * roi_frac)

        src = self.BEV_SRC_RATIO.copy()
        dst = self.BEV_DST_RATIO.copy()
        src[:, 0] *= frame_w;  src[:, 1] *= frame_h
        dst[:, 0] *= frame_w;  dst[:, 1] *= frame_h

        src[:, 1] -= self._roi_y
        src[:, 1]  = np.clip(src[:, 1], 0, frame_h - self._roi_y - 1)

        self._M    = cv2.getPerspectiveTransform(src, dst)
        self._Minv = cv2.getPerspectiveTransform(dst, src)

        self._bev_w = frame_w
        self._bev_h = frame_h - self._roi_y

        # The destination window spans exactly one lane width by construction
        # (BEV_SCALE_PX_PER_CM is derived from LANE_WIDTH_M), so a car centred in
        # its lane puts BOTH lines precisely on the window's edges. Cropping hard
        # at the edge therefore deletes a legitimately-positioned line as soon as
        # the car sits a couple of centimetres off centre.
        #
        # Measured 2026-07-26: left line at BEV x=110, right at x=490 -- a
        # separation of 380 px against the expected 384, so the geometry was very
        # nearly ideal -- yet the left line fell outside [128, 512] and was
        # zeroed, dropping the pipeline to "RIGHT only" and 50% confidence.
        #
        # The margin buys tolerance for that offset while still discarding the far
        # periphery, which is what the crop exists for: at 0.06 the window becomes
        # [90, 550], which keeps the line at 110 and still rejects the sunlit floor
        # measured at x=619. Columns out there are lateral extrapolations of the
        # homography and were the source of the all-white mask that produced a
        # false 100% lock.
        margin = int(round(self.BEV_VALID_MARGIN_RATIO * frame_w))
        lo = self.BEV_DST_RATIO[:, 0].min() * frame_w - margin
        hi = self.BEV_DST_RATIO[:, 0].max() * frame_w + margin
        self._valid_x0 = max(0, int(round(lo)))
        self._valid_x1 = min(frame_w, int(round(hi)))

        self._smooth_error = 0.0
        self._prev_conf    = 0.0

        self._last_good_error = 0.0
        self._last_good_time  = 0.0
        self.LANE_HOLD_S      = 1.0
        self.MAX_ERR_JUMP_PX  = 90.0

        self._morph_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.last_v_min = int(self.HSV_WHITE_LO[2])


    def _lane_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Isolate the lane lines with a morphological top-hat.

        The HSV route this replaces rejected two of the three lines on the real
        track. Measured in the bird's-eye view, 2026-07-27:

            right solid   V p90 196   S  56   <- the only one that passed
            dashed centre V p90  92   S 130   <- rejected by S <= 60
            left solid    V p90 112   S 107   <- rejected by S <= 60
            track asphalt V p90  49   S  75
            floor, off-track  V p90 235       <- brighter than every line

        Two independent failures there. First, saturation is meaningless for a
        dim line: S = (max-min)/max, so a "white" stripe reading (40,50,60) comes
        out at S=85 purely from sensor noise and colour cast. Filtering on it
        throws away exactly the lines that are hardest to see. Second, the
        off-track floor is BRIGHTER than any line, so a global brightness
        percentile is set by the floor and lands above the real lines.

        A top-hat has neither problem. It keeps bright structures NARROWER than
        its kernel and subtracts everything wider, so a 25 px line survives and a
        200 px slab of floor vanishes -- regardless of absolute brightness, and
        without consulting colour at all. On the frame above it recovers all
        three lines at 111 / 286 / 476 px with 6.5 % mask fill and no floor
        artefacts, where the HSV route found only one.

        TOPHAT_V_FLOOR is the one absolute check kept: on a pitch-black frame the
        top-hat would happily amplify sensor noise into a plausible mask, and the
        degenerate-mask guard in tools/diag_track.py needs that to stay visible.
        """
        v = hsv[..., 2]
        if not self.TOPHAT_ENABLED:
            lo, hi = self._white_bounds(hsv)
            self.last_v_min = int(lo[2])
            return cv2.inRange(hsv, lo, hi)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.TOPHAT_W, 1))
        th = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, k)
        mask = np.where((th >= self.TOPHAT_MIN) & (v >= self.TOPHAT_V_FLOOR),
                        np.uint8(255), np.uint8(0))
        self.last_v_min = int(self.TOPHAT_MIN)
        return mask


    def _white_bounds(self, hsv: np.ndarray):
        """Pick V_min from the view's own histogram instead of a fixed number.

        A fixed V_min cannot survive changing light, and the sign detector wants
        the exposure pinned (it needs correct absolute colour -- an overexposed red
        octagon clips to white and becomes invisible), so the lane cannot be fixed
        by opening the aperture. Measured the same afternoon on this track with
        exposure pinned at gain 4.0:

            14:00  lines at V=255, mask fill 9.1 %, confidence 100 %
            15:00  light dropped, mask fill 0.0 %, DEGENERATE, confidence 0 %

        The lines never stopped being the brightest thing on a dark plastic track,
        they just stopped clearing 130. So threshold on that relation instead: take
        a high percentile of V, which is where the lines live because they cover
        only a few percent of the view.

        Guards, both necessary:
          - the percentile must sit MIN_CONTRAST above the median, otherwise there
            is nothing line-like in view and a percentile would happily threshold
            noise into a plausible-looking mask. Failing that, fall back to the
            fixed floor so the mask comes out empty and the degenerate-mask check
            in tools/diag_track.py can still fire. An always-8 %-fill mask would
            defeat that check, which is the trap this guard exists to avoid.
          - the result is clamped to [V_ADAPT_FLOOR, V_ADAPT_CEIL] so a blown-out
            frame cannot drive the threshold to 255, nor a dark one to 0.
        """
        if not self.ADAPTIVE_WHITE:
            return self.HSV_WHITE_LO, self.HSV_WHITE_HI

        v = hsv[..., 2]
        med  = float(np.median(v))
        high = float(np.percentile(v, self.V_ADAPT_PCTL))

        if high - med < self.MIN_CONTRAST:
            return self.HSV_WHITE_LO, self.HSV_WHITE_HI

        v_min = float(np.clip(0.80 * high + 0.20 * med,
                              self.V_ADAPT_FLOOR, self.V_ADAPT_CEIL))
        lo = np.array([self.HSV_WHITE_LO[0], self.HSV_WHITE_LO[1], int(v_min)])
        return lo, self.HSV_WHITE_HI


    def process(self, frame: np.ndarray) -> LaneResult:
        """
        Process a BGR frame and return the steering error.

        Parameters
        ----------
        frame : np.ndarray
            BGR camera frame (already converted with cv2.COLOR_RGB2BGR).

        Returns
        -------
        LaneResult with error_px and confidence.
        """
        roi = frame[self._roi_y:, :]

        bev = cv2.warpPerspective(roi, self._M, (self._bev_w, self._bev_h))

        hsv  = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        mask = self._lane_mask(hsv)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._morph_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_k)

        mask[:, :self._valid_x0] = 0
        mask[:, self._valid_x1:] = 0

        # Exposed for diagnostics: a healthy lane mask covers a few percent of the
        # view. Near 0 means nothing passed the threshold; very high means the
        # track itself is leaking in. tools/tune_exposure.py and diag_track.py both
        # read it to tell those two failures apart from a real lock.
        self.last_mask_fill = float((mask > 0).mean() * 100.0)

        result = self._sliding_windows(mask)

        # Zero out the standing bias before any smoothing, so the EMA, the
        # jump-rejection threshold and every consumer all see the corrected
        # value. Measured 2026-07-26: with the car deliberately centred in its
        # lane the raw error read a stable +50 px, which at BEV_SCALE_PX_PER_CM
        # (6.8 px/cm) is 7 cm of offset the lane follower would have held on
        # purpose. The cause is mechanical, not algorithmic -- the camera is not
        # perfectly on the chassis centreline and the BEV trapezoid is not exactly
        # symmetric about the lens axis -- so it belongs in a calibration constant
        # rather than in the geometry. Re-measure with tools/diag_track.py whenever
        # the camera is remounted.
        result.error_px -= self.error_offset_px

        now = time.monotonic()
        CONF_OK = 0.9

        if result.confidence >= CONF_OK:
            jump = abs(result.error_px - self._last_good_error)
            if (jump > self.MAX_ERR_JUMP_PX
                    and (now - self._last_good_time) <= self.LANE_HOLD_S):
                result.error_px = self._last_good_error
            else:
                smoothed = (self.EMA_ALPHA * result.error_px
                            + (1 - self.EMA_ALPHA) * self._smooth_error)
                self._smooth_error    = smoothed
                result.error_px       = smoothed
                self._last_good_error = smoothed
                self._last_good_time  = now

        elif (now - self._last_good_time) <= self.LANE_HOLD_S:
            result.error_px  = self._last_good_error
            result.confidence = max(result.confidence, CONF_OK)
            self._smooth_error = self._last_good_error

        elif result.confidence > 0.1:
            smoothed = (self.EMA_ALPHA * result.error_px
                        + (1 - self.EMA_ALPHA) * self._smooth_error)
            self._smooth_error = smoothed
            result.error_px    = smoothed

        if self._debug:
            result.bev_frame  = bev
            result.mask_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        return result

    def calibrate_bev(self, src_points: np.ndarray) -> None:
        """
        Update the perspective points at runtime.

        Parameters
        ----------
        src_points : np.ndarray shape (4,2)
            Points in the original frame (absolute pixels).
        """
        dst = self.BEV_DST_RATIO.copy()
        dst[:, 0] *= self._w;  dst[:, 1] *= self._h
        src_roi = src_points.astype(np.float32)
        src_roi[:, 1] -= self._roi_y

        self._M    = cv2.getPerspectiveTransform(src_roi, dst)
        self._Minv = cv2.getPerspectiveTransform(dst, src_roi)


    def _sliding_windows(self, binary: np.ndarray) -> LaneResult:
        """
        Locate the white lane lines using sliding windows.

        Algorithm:
          1. Histogram of the lower half of the BEV.
          2. Left and right peaks as the initial position of each line.
          3. N windows from bottom to top -- recompute the centre per window.
          4. Average the found positions -> lane centre.
          5. Error = lane_centre - frame_centre.
        """
        h, w = binary.shape
        mid  = w // 2

        hist     = np.sum(binary[h // 2:, :], axis=0).astype(np.int32)
        left_x   = int(np.argmax(hist[:mid]))
        right_x  = int(np.argmax(hist[mid:])) + mid

        has_left  = hist[left_x]  > 300
        has_right = hist[right_x] > 300

        if not has_left and not has_right:
            return LaneResult(error_px=self._smooth_error, confidence=0.0)

        win_h        = h // self.N_WINDOWS
        left_centers  = []
        right_centers = []
        # Window index of each accepted centre (0 = bottom). The slope of x over
        # these indices is the line's lean, i.e. the car's heading error --
        # information the windows were already computing and then throwing away.
        left_idx  = []
        right_idx = []

        cur_left  = left_x
        cur_right = right_x

        for i in range(self.N_WINDOWS):
            y_lo = h - (i + 1) * win_h
            y_hi = h - i * win_h

            if has_left:
                xl_lo = max(0, cur_left  - self.WIN_MARGIN)
                xl_hi = min(w, cur_left  + self.WIN_MARGIN)
                win_l = binary[y_lo:y_hi, xl_lo:xl_hi]
                nz_l  = np.count_nonzero(win_l)
                if nz_l >= self.MIN_PIX:
                    pts  = np.where(win_l > 0)[1]
                    cur_left = int(np.mean(pts)) + xl_lo
                    left_centers.append(cur_left)
                    left_idx.append(i)

            if has_right:
                xr_lo = max(0, cur_right - self.WIN_MARGIN)
                xr_hi = min(w, cur_right + self.WIN_MARGIN)
                win_r = binary[y_lo:y_hi, xr_lo:xr_hi]
                nz_r  = np.count_nonzero(win_r)
                if nz_r >= self.MIN_PIX:
                    pts   = np.where(win_r > 0)[1]
                    cur_right = int(np.mean(pts)) + xr_lo
                    right_centers.append(cur_right)
                    right_idx.append(i)

        frame_cx = w / 2.0

        # --- Heading: how much the lines LEAN, from the slope of the window
        # centres. Lateral offset alone cannot damp a heading disturbance: the
        # controller only reacts once the car has already translated sideways,
        # which at cruise speed means it weaves or drifts out before converging.
        # Window index runs bottom -> top, so a POSITIVE slope means the lines
        # lean right going up, i.e. the car's nose points LEFT of the road axis
        # and it must steer RIGHT. Pseudo-degrees: the BEV's vertical scale is
        # not calibrated to its horizontal one, so the magnitude is only
        # proportional to the true angle -- the consumer's gain absorbs that.
        def _lean(idx, xs):
            if len(xs) < 3:
                return None
            return float(np.polyfit(np.asarray(idx, np.float64),
                                    np.asarray(xs, np.float64), 1)[0])

        leans = [s for s in (_lean(left_idx, left_centers),
                             _lean(right_idx, right_centers)) if s is not None]
        heading = (float(np.degrees(np.arctan2(np.mean(leans), float(win_h))))
                   if leans else 0.0)
        # This road has THREE lines -- left solid, dashed centre, right solid --
        # so the pair the sliding windows return is one of two different things,
        # and each needs its own target rule. Measured in the bird's-eye view:
        #
        #   left solid <-> right solid   0.565 m -> 384 px   aim at 74 % across
        #   dashed     <-> right solid   0.290 m -> 197 px   aim at 50 % across
        #
        # Both rules place the car in the same physical spot -- the centre of the
        # right lane -- and agree to 1.5 px, which is the consistency check that
        # says the geometry is right. Recognising only the 384 case (what this did
        # before) threw the dashed line away and dropped to single-line 50 %
        # confidence for the entire run, because with the car correctly inside the
        # right lane the LEFT SOLID sits at BEV x~35, outside the valid window.
        road_px = LANE_WIDTH_M * 100.0 * self.BEV_SCALE_PX_PER_CM
        lane_px = _CFG_DRIVEN_WIDTH_M * 100.0 * self.BEV_SCALE_PX_PER_CM

        l_by_i = dict(zip(left_idx,  left_centers))
        r_by_i = dict(zip(right_idx, right_centers))
        paired = sorted(set(l_by_i) & set(r_by_i))
        widths = [r_by_i[i] - l_by_i[i] for i in paired]
        sep    = float(np.median(widths)) if widths else 0.0

        if widths and abs(sep / road_px - 1.0) <= self.LANE_WIDTH_TOL:
            width_px, bias = sep, self._right_bias          # outer pair
        elif widths and abs(sep / lane_px - 1.0) <= self.LANE_WIDTH_TOL:
            width_px, bias = sep, 0.5                       # the driven lane
        else:
            # Neither: one side is off-track, or the dashed was paired with the
            # left solid (the LEFT lane, ~187 px -- nearly the same width as the
            # right one, so it cannot be told apart by width alone). Fall back to
            # a single line and rebuild from known geometry.
            #
            # Prefer the RIGHT line: it is continuous, the brightest thing in the
            # view, and stays inside the window for the whole right lane, whereas
            # any lone left-of-centre stripe is ambiguous between the left solid
            # and the dashed. Half a lane left of the right solid is the target
            # either way.
            if r_by_i:
                l_by_i = {}
            else:
                r_by_i = {}
            width_px, bias, paired = lane_px, 0.5, []

        # ONE target rule for all three cases, so the estimate cannot jump when a
        # line drops out: every row aims at the same fraction `bias` across the
        # same measured lane width. The old code used magic fractions of the
        # FRAME width (0.20+0.16*bias for left-only, 0.36-0.16*bias for
        # right-only), which placed the three cases at DIFFERENT points in the
        # lane. Measured at bias 0.70 they aimed at 70 %, 52 % and 59 % -- a 69 px
        # (10 cm) step the instant one line flickered, straight into the servo.
        centres_i: list[float] = []
        centres_x: list[float] = []
        for i in sorted(set(l_by_i) | set(r_by_i)):
            if i in l_by_i and i in r_by_i:
                cx = l_by_i[i] + bias * (r_by_i[i] - l_by_i[i])
            elif i in l_by_i:
                cx = l_by_i[i] + bias * width_px
            else:
                cx = r_by_i[i] - (1.0 - bias) * width_px
            centres_i.append(float(i))
            centres_x.append(float(cx))

        if not centres_x:
            return LaneResult(error_px=self._smooth_error, confidence=0.0)

        # AIM ROW -- the lookahead that makes this controllable at all.
        #
        # Steering on the instantaneous cross-track error is a double integrator
        # (steering -> yaw rate -> heading -> lateral position); it oscillates at
        # ANY gain, which is why neither Stanley nor pure pursuit uses cross-track
        # alone. Pure pursuit aims at a point AHEAD, and for small angles
        #     delta ~= 2 * L * x_ahead / Ld^2
        # i.e. the steering command is proportional to the LATERAL OFFSET OF THAT
        # POINT -- exactly what this returns. The lookahead supplies the damping
        # and also covers the ~150 ms sensorimotor delay (33 ms exposure + ~10 ms
        # pipeline + ~100 ms of MG90S travel), which at 0.3 m/s is 4.5 cm of blind
        # motion.
        #
        # Averaging all nine window rows -- what this did before -- discards the
        # lookahead and hands the gain an error measured at no particular
        # distance. With L=0.31 m, Kp=0.08 deg/px corresponds to Ld~0.8 m, so the
        # existing gain is already right for an aim near the top of the view; what
        # was missing was measuring the error THERE.
        aim_i = self.AIM_WINDOW_FRAC * (self.N_WINDOWS - 1)
        if len(centres_x) >= 2:
            # Fit through every available row, then evaluate at the aim row: uses
            # all the evidence (noise averaging) and extrapolates cleanly when the
            # far windows found nothing, which is common -- the lines are thinnest
            # and dimmest up there.
            slope, intercept = np.polyfit(np.asarray(centres_i),
                                          np.asarray(centres_x), 1)
            lane_cx = float(slope * aim_i + intercept)
        else:
            lane_cx = centres_x[0]

        confidence  = 1.0 if paired else 0.5
        # Never report a full lock on almost no evidence. A top-hat will happily
        # turn sensor noise into plausible narrow stripes on a black frame, and a
        # confident wrong error drives the car off the track; a hedged one only
        # slows it. A healthy mask covers a few percent of the view.
        if self.last_mask_fill < self.MIN_FILL_FOR_LOCK:
            confidence = min(confidence, 0.5)
        left_x_avg  = int(np.mean(left_centers))  if l_by_i else None
        right_x_avg = int(np.mean(right_centers)) if r_by_i else None

        error = float(lane_cx - frame_cx)

        return LaneResult(
            error_px   = error,
            confidence = float(confidence),
            heading    = heading,
            left_x     = left_x_avg if left_centers else None,
            right_x    = right_x_avg if right_centers else None,
        )


    def draw_debug(self, frame: np.ndarray, result: LaneResult) -> np.ndarray:
        """
        Draw the detected lane line on top of the original frame.
        Returns an annotated copy.
        """
        vis = frame.copy()
        H, W = vis.shape[:2]

        cv2.line(vis, (W // 2, H), (W // 2, H // 2), (0, 150, 150), 1)

        cx = W // 2 + int(result.error_px)
        cx = max(0, min(W - 1, cx))
        col = (0, 255, 0) if result.confidence >= 0.5 else (0, 80, 255)
        cv2.line(vis, (cx, H), (cx, H // 2), col, 3)

        cv2.putText(vis,
            f"err:{result.error_px:+.0f}px  conf:{result.confidence:.0%}",
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

        return vis
