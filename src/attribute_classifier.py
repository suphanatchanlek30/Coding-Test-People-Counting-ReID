from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AttributePrediction:
    category: str
    confidence: float
    blue_ratio: float
    black_ratio: float
    reason: str = ""


class HSVAttributeClassifier:
    def __init__(
        self,
        blue_ratio_threshold: float = 0.18,
        black_ratio_threshold: float = 0.35,
        detect_staff_black: bool = False,
    ) -> None:
        self.blue_ratio_threshold = blue_ratio_threshold
        self.black_ratio_threshold = black_ratio_threshold
        self.detect_staff_black = detect_staff_black

    def classify(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> AttributePrediction:
        crop = self._crop_upper_body(frame, bbox)

        if crop.size == 0:
            return AttributePrediction("unknown", 0.0, 0.0, 0.0, "empty_crop")

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        blue_mask = cv2.inRange(hsv, (88, 55, 45), (132, 255, 255))
        black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 80, 70))
        visible_color_mask = cv2.inRange(hsv, (0, 35, 45), (180, 255, 255))

        total_pixels = float(hsv.shape[0] * hsv.shape[1])
        blue_ratio = float(np.count_nonzero(blue_mask)) / total_pixels
        black_ratio = float(np.count_nonzero(black_mask)) / total_pixels
        visible_color_ratio = float(np.count_nonzero(visible_color_mask)) / total_pixels

        ambiguous_blue_ratio = self.blue_ratio_threshold * 0.45

        if blue_ratio >= self.blue_ratio_threshold and black_ratio < 0.35:
            return AttributePrediction(
                "superai_shirt",
                min(1.0, blue_ratio / max(self.blue_ratio_threshold, 1e-6)),
                blue_ratio,
                black_ratio,
                "blue_ratio_above_superai_threshold",
            )

        if (
            self.detect_staff_black
            and black_ratio >= self.black_ratio_threshold
            and blue_ratio < 0.12
        ):
            return AttributePrediction(
                "staff_black",
                min(1.0, black_ratio / max(self.black_ratio_threshold, 1e-6)),
                blue_ratio,
                black_ratio,
                "black_ratio_above_staff_threshold",
            )

        if blue_ratio >= ambiguous_blue_ratio:
            return AttributePrediction(
                "unknown",
                0.45,
                blue_ratio,
                black_ratio,
                "some_blue_but_not_enough_for_superai",
            )

        if black_ratio >= self.black_ratio_threshold and blue_ratio < 0.12:
            return AttributePrediction(
                "non_superai",
                0.72,
                blue_ratio,
                black_ratio,
                "confident_non_superai_dark_torso",
            )

        if visible_color_ratio >= 0.12 and blue_ratio < ambiguous_blue_ratio:
            confidence = min(0.88, 0.58 + visible_color_ratio)
            return AttributePrediction(
                "non_superai",
                confidence,
                blue_ratio,
                black_ratio,
                "confident_non_blue_torso",
            )

        if max(blue_ratio, black_ratio) < 0.06:
            return AttributePrediction(
                "unknown",
                0.35,
                blue_ratio,
                black_ratio,
                "low_color_evidence",
            )

        return AttributePrediction(
            "unknown",
            0.40,
            blue_ratio,
            black_ratio,
            "ambiguous_torso_color",
        )

    @staticmethod
    def _crop_upper_body(
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=frame.dtype)

        box_w = x2 - x1
        box_h = y2 - y1

        torso_x1 = x1 + int(box_w * 0.18)
        torso_x2 = x2 - int(box_w * 0.18)
        torso_y1 = y1 + int(box_h * 0.12)
        torso_y2 = y1 + int(box_h * 0.58)

        if torso_x2 <= torso_x1 or torso_y2 <= torso_y1:
            return frame[y1:y2, x1:x2]

        return frame[torso_y1:torso_y2, torso_x1:torso_x2]