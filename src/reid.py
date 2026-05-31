from __future__ import annotations

import cv2
import numpy as np


class ColorHistogramReID:
    """Lightweight person ReID from non-biometric multi-region appearance.

    The feature intentionally avoids automatic face recognition and skin-tone
    identity matching. It uses clothing, shoes, hair/head-region color proxy,
    and coarse body-box shape to reduce duplicate counting in a lightweight
    coding-test environment.
    """

    def __init__(
        self,
        hsv_bins: tuple[int, int] = (24, 16),
        lab_bins: tuple[int, int] = (16, 16),
    ) -> None:
        self.hsv_bins = hsv_bins
        self.lab_bins = lab_bins

    def extract(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray:
        regions = self._extract_regions(frame, bbox)
        region_weights = {
            "torso": 0.34,
            "pants": 0.26,
            "shoes": 0.18,
            "head_hair": 0.10,
            "full_body": 0.06,
        }

        parts = []
        for region_name, crop in regions.items():
            weight = region_weights[region_name]
            parts.extend(
                [
                    self._hsv_histogram(crop, self.hsv_bins) * (weight * 0.55),
                    self._lab_histogram(crop, self.lab_bins) * (weight * 0.30),
                    self._color_moments(crop) * (weight * 0.15),
                ]
            )

        # Shape is deliberately weak: it helps reject impossible matches without
        # turning body size into the main identity signal.
        parts.append(self._shape_descriptor(frame.shape, bbox) * 0.06)

        feature = np.concatenate(parts).astype(np.float32)
        norm = np.linalg.norm(feature)
        if norm > 1e-8:
            feature = feature / norm

        return feature

    @staticmethod
    def similarity(feature_a: np.ndarray | None, feature_b: np.ndarray | None) -> float:
        if feature_a is None or feature_b is None:
            return 0.0

        denom = float(np.linalg.norm(feature_a) * np.linalg.norm(feature_b))
        if denom <= 1e-8:
            return 0.0

        return float(np.dot(feature_a, feature_b) / denom)

    @staticmethod
    def _hsv_histogram(crop: np.ndarray, bins: tuple[int, int]) -> np.ndarray:
        if crop.size == 0:
            return np.zeros(bins[0] * bins[1], dtype=np.float32)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 20, 25), (180, 255, 255))
        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            mask,
            bins,
            [0, 180, 0, 256],
        )
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    @staticmethod
    def _lab_histogram(crop: np.ndarray, bins: tuple[int, int]) -> np.ndarray:
        if crop.size == 0:
            return np.zeros(bins[0] * bins[1], dtype=np.float32)

        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist(
            [lab],
            [1, 2],
            None,
            bins,
            [0, 256, 0, 256],
        )
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    @staticmethod
    def _color_moments(crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros(12, dtype=np.float32)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)

        hsv_mean = hsv.reshape(-1, 3).mean(axis=0) / np.array([180, 255, 255])
        hsv_std = hsv.reshape(-1, 3).std(axis=0) / np.array([180, 255, 255])
        lab_mean = lab.reshape(-1, 3).mean(axis=0) / 255.0
        lab_std = lab.reshape(-1, 3).std(axis=0) / 255.0

        return np.concatenate([hsv_mean, hsv_std, lab_mean, lab_std]).astype(
            np.float32
        )

    def _extract_regions(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> dict[str, np.ndarray]:
        return {
            # Head/hair proxy: color/texture only, not face recognition.
            "head_hair": self._crop_region(frame, bbox, 0.00, 0.20, 0.20, 0.20),
            "torso": self._crop_region(frame, bbox, 0.18, 0.56, 0.12, 0.12),
            "pants": self._crop_region(frame, bbox, 0.50, 0.86, 0.10, 0.10),
            "shoes": self._crop_region(frame, bbox, 0.82, 1.00, 0.00, 0.00),
            "full_body": self._crop_region(frame, bbox, 0.05, 0.98, 0.08, 0.08),
        }

    @staticmethod
    def _shape_descriptor(
        frame_shape: tuple[int, ...],
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray:
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        area = box_w * box_h

        descriptor = np.array(
            [
                box_w / box_h,
                box_h / max(1.0, frame_h),
                box_w / max(1.0, frame_w),
                area / max(1.0, frame_w * frame_h),
            ],
            dtype=np.float32,
        )
        return descriptor

    @staticmethod
    def _crop_region(
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        y_start_ratio: float,
        y_end_ratio: float,
        x_left_margin: float,
        x_right_margin: float,
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

        rx1 = x1 + int(box_w * x_left_margin)
        rx2 = x2 - int(box_w * x_right_margin)
        ry1 = y1 + int(box_h * y_start_ratio)
        ry2 = y1 + int(box_h * y_end_ratio)

        if rx2 <= rx1 or ry2 <= ry1:
            return frame[y1:y2, x1:x2]

        return frame[ry1:ry2, rx1:rx2]
