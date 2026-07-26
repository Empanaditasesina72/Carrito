"""Lane-detection pipeline: BEV + HSV + Sliding Windows.

Full pipeline:
  1. ROI: crop the lower half of the frame (ignore sky/upper noise).
  2. Bird's-Eye View: perspective transform to a top-down view.
  2b. Discard the columns outside the destination rectangle. The homography is
      only defined between the trapezoid and BEV_DST_RATIO; columns beyond it are
      lateral extrapolations that sample whatever lies past the track edge, so
      they are zeroed before the histogram sees them.
  3. Strict HSV filter: isolate white and reject glossy-black reflections.
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


@dataclass
class LaneResult:
    """Result of the lane-detection pipeline."""
    error_px:    float
    confidence:  float
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

    N_WINDOWS  = 9
    WIN_MARGIN = 70
    MIN_PIX    = 60

    EMA_ALPHA  = 0.45

    RIGHT_BIAS = 0.70

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
    ):
        self._w     = frame_w
        self._h     = frame_h
        self._debug = debug
        self._right_bias = max(0.0, min(1.0, float(right_bias)))

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
        mask = cv2.inRange(hsv, self.HSV_WHITE_LO, self.HSV_WHITE_HI)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._morph_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_k)

        mask[:, :self._valid_x0] = 0
        mask[:, self._valid_x1:] = 0

        result = self._sliding_windows(mask)

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

            if has_right:
                xr_lo = max(0, cur_right - self.WIN_MARGIN)
                xr_hi = min(w, cur_right + self.WIN_MARGIN)
                win_r = binary[y_lo:y_hi, xr_lo:xr_hi]
                nz_r  = np.count_nonzero(win_r)
                if nz_r >= self.MIN_PIX:
                    pts   = np.where(win_r > 0)[1]
                    cur_right = int(np.mean(pts)) + xr_lo
                    right_centers.append(cur_right)

        frame_cx = w / 2.0
        bias     = self._right_bias

        if left_centers and right_centers:
            mean_l = float(np.mean(left_centers))
            mean_r = float(np.mean(right_centers))
            expected_px = LANE_WIDTH_M * 100.0 * self.BEV_SCALE_PX_PER_CM
            measured_px = mean_r - mean_l
            ratio = measured_px / max(1.0, expected_px)
            valid_width = abs(ratio - 1.0) <= self.LANE_WIDTH_TOL

            if not valid_width:
                if len(left_centers) >= len(right_centers):
                    lane_cx    = mean_l + w * (0.20 + 0.16 * bias)
                    confidence = 0.5
                    left_x_avg  = int(mean_l)
                    right_x_avg = None
                else:
                    lane_cx    = mean_r - w * (0.36 - 0.16 * bias)
                    confidence = 0.5
                    left_x_avg  = None
                    right_x_avg = int(mean_r)
                error = float(lane_cx - frame_cx)
                return LaneResult(error_px=error, confidence=confidence,
                                  left_x=left_x_avg, right_x=right_x_avg)

            lane_cx    = mean_l + bias * (mean_r - mean_l)
            confidence = 1.0
            left_x_avg  = int(mean_l)
            right_x_avg = int(mean_r)
        elif left_centers:
            lane_cx    = np.mean(left_centers) + w * (0.20 + 0.16 * bias)
            confidence = 0.5
            left_x_avg  = int(np.mean(left_centers))
            right_x_avg = None
        elif right_centers:
            lane_cx    = np.mean(right_centers) - w * (0.36 - 0.16 * bias)
            confidence = 0.5
            left_x_avg  = None
            right_x_avg = int(np.mean(right_centers))
        else:
            return LaneResult(error_px=self._smooth_error, confidence=0.0)

        error = float(lane_cx - frame_cx)

        return LaneResult(
            error_px   = error,
            confidence = float(confidence),
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
