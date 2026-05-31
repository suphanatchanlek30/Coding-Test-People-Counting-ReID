from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


class ReviewCropExporter:
    """Export full-body crops for classes that should be manually checked."""

    def __init__(
        self,
        output_dir: str = "outputs/review_crops",
        categories: list[str] | None = None,
        max_crops_per_id: int = 4,
        min_frame_gap: int = 24,
        crop_padding: float = 0.08,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.categories = set(categories or ["non_superai", "unknown"])
        self.max_crops_per_id = max_crops_per_id
        self.min_frame_gap = min_frame_gap
        self.crop_padding = crop_padding

        self.saved_counts: dict[tuple[str, int], int] = defaultdict(int)
        self.last_saved_frame: dict[tuple[str, int], int] = defaultdict(lambda: -10**9)
        self.saved_paths: dict[str, list[Path]] = defaultdict(list)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_if_needed(
        self,
        frame: np.ndarray,
        category: str,
        global_id: int,
        track_id: int,
        frame_id: int,
        timestamp: float,
        bbox: tuple[float, float, float, float],
        confidence: float,
        category_confidence: float,
    ) -> None:
        if category not in self.categories:
            return

        key = (category, global_id)

        if self.saved_counts[key] >= self.max_crops_per_id:
            return

        if frame_id - self.last_saved_frame[key] < self.min_frame_gap:
            return

        crop = self._crop_with_padding(frame, bbox)
        if crop.size == 0:
            return

        label = (
            f"G{global_id:03d} T{track_id} "
            f"{category} det={confidence:.2f} attr={category_confidence:.2f}"
        )
        crop = self._draw_crop_label(crop, label)

        output_dir = self.output_dir / category / f"G{global_id:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"frame_{frame_id:06d}_T{track_id}.jpg"
        cv2.imwrite(str(output_path), crop)

        self.saved_counts[key] += 1
        self.last_saved_frame[key] = frame_id
        self.saved_paths[category].append(output_path)

    def write_contact_sheets(self) -> list[str]:
        output_paths = []

        for category, paths in sorted(self.saved_paths.items()):
            thumbnails = []

            for path in paths:
                image = cv2.imread(str(path))
                if image is None:
                    continue

                thumbnails.append(self._make_thumbnail(image, label=category))

            if not thumbnails:
                continue

            sheet = self._tile_images(thumbnails, columns=5)
            output_path = self.output_dir / f"{category}_contact_sheet.jpg"
            cv2.imwrite(str(output_path), sheet)
            output_paths.append(str(output_path))

        return output_paths

    def _crop_with_padding(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray:
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]

        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        pad_x = int(w * self.crop_padding)
        pad_y = int(h * self.crop_padding)

        x1 = max(0, x1 - pad_x)
        x2 = min(frame_w, x2 + pad_x)
        y1 = max(0, y1 - pad_y)
        y2 = min(frame_h, y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=frame.dtype)

        return frame[y1:y2, x1:x2]

    @staticmethod
    def _draw_crop_label(image: np.ndarray, label: str) -> np.ndarray:
        canvas = image.copy()
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 32), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        cv2.putText(
            canvas,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas

    @staticmethod
    def _make_thumbnail(
        image: np.ndarray,
        label: str,
        size: tuple[int, int] = (180, 260),
    ) -> np.ndarray:
        thumb_w, thumb_h = size
        canvas = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)

        h, w = image.shape[:2]
        scale = min(thumb_w / max(w, 1), (thumb_h - 28) / max(h, 1))
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))

        rh, rw = resized.shape[:2]
        x = (thumb_w - rw) // 2
        y = 24 + ((thumb_h - 28 - rh) // 2)
        canvas[y : y + rh, x : x + rw] = resized

        cv2.putText(
            canvas,
            label,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas

    @staticmethod
    def _tile_images(images: list[np.ndarray], columns: int = 5) -> np.ndarray:
        if not images:
            return np.empty((0, 0, 3), dtype=np.uint8)

        h, w = images[0].shape[:2]
        rows = int(np.ceil(len(images) / columns))
        sheet = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)

        for index, image in enumerate(images):
            row = index // columns
            col = index % columns
            sheet[row * h : (row + 1) * h, col * w : (col + 1) * w] = image

        return sheet