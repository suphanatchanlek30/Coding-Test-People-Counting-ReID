from __future__ import annotations

import cv2
import numpy as np


class ColorHistogramReID:
    """Lightweight person ReID based on multi-region body appearance.

    This is a practical fallback when a trained ReID model such as OSNet or
    FastReID is not available. It intentionally avoids face recognition.
    """

    def __init__(
        self,
        hist_bins: tuple[int, int, int] = (12, 8, 8),
        feature_size: int = 256,
    ) -> None:
        self.hist_bins = hist_bins
        self.feature_size = feature_size

    def extract(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray:
        crop = self._safe_crop(frame, bbox)
        if crop.size == 0:
            return np.zeros(self.feature_size, dtype=np.float32)

        regions = self._body_regions(crop)
        features: list[np.ndarray] = []

        for region_name, region in regions:
            if region.size == 0:
                continue

            weight = self._region_weight(region_name)
            hist = self._hsv_histogram(region) * weight
            color_moments = self._color_moments(region) * weight
            features.extend([hist, color_moments])

        shape_feature = self._shape_feature(bbox)
        features.append(shape_feature)

        if not features:
            return np.zeros(self.feature_size, dtype=np.float32)

        vector = np.concatenate(features).astype(np.float32)
        vector = self._resize_feature(vector, self.feature_size)
        return self._normalize(vector)

    def similarity(
        self,
        feature_a: np.ndarray | None,
        feature_b: np.ndarray | None,
    ) -> float:
        if feature_a is None or feature_b is None:
            return 0.0

        if feature_a.size == 0 or feature_b.size == 0:
            return 0.0

        a = self._normalize(feature_a.astype(np.float32))
        b = self._normalize(feature_b.astype(np.float32))

        return float(np.clip(np.dot(a, b), 0.0, 1.0))

    @staticmethod
    def _safe_crop(
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

        return frame[y1:y2, x1:x2]

    @staticmethod
    def _body_regions(crop: np.ndarray) -> list[tuple[str, np.ndarray]]:
        h, w = crop.shape[:2]

        head = crop[0 : int(h * 0.22), int(w * 0.20) : int(w * 0.80)]
        torso = crop[int(h * 0.18) : int(h * 0.58), int(w * 0.14) : int(w * 0.86)]
        pants = crop[int(h * 0.52) : int(h * 0.86), int(w * 0.12) : int(w * 0.88)]
        shoes = crop[int(h * 0.80) : h, int(w * 0.08) : int(w * 0.92)]
        full = crop

        return [
            ("head", head),
            ("torso", torso),
            ("pants", pants),
            ("shoes", shoes),
            ("full", full),
        ]

    @staticmethod
    def _region_weight(region_name: str) -> float:
        weights = {
            "head": 0.55,
            "torso": 1.25,
            "pants": 0.85,
            "shoes": 0.70,
            "full": 0.55,
        }
        return weights.get(region_name, 1.0)

    def _hsv_histogram(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            self.hist_bins,
            [0, 180, 0, 256, 0, 256],
        )
        hist = cv2.normalize(hist, hist).flatten()
        return hist.astype(np.float32)

    @staticmethod
    def _color_moments(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        moments = []

        for channel in cv2.split(hsv):
            channel = channel.astype(np.float32)
            moments.append(float(np.mean(channel)) / 255.0)
            moments.append(float(np.std(channel)) / 255.0)

        return np.array(moments, dtype=np.float32)

    @staticmethod
    def _shape_feature(bbox: tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        area = width * height

        return np.array(
            [
                width / max(height, 1.0),
                height / max(width, 1.0),
                min(area / 250000.0, 1.0),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _resize_feature(feature: np.ndarray, target_size: int) -> np.ndarray:
        if feature.size == target_size:
            return feature

        if feature.size > target_size:
            return feature[:target_size]

        padded = np.zeros(target_size, dtype=np.float32)
        padded[: feature.size] = feature
        return padded

    @staticmethod
    def _normalize(feature: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(feature))
        if norm <= 1e-8:
            return feature.astype(np.float32)

        return (feature / norm).astype(np.float32)