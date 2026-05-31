from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


class HeadVerifier:
    """Export head crops for manual identity review.

    This is not face recognition. It only saves approximate head crops so a
    reviewer can manually check whether a Global ID is fragmented or duplicated.
    """

    def __init__(
        self,
        output_dir: str = "outputs/head_crops",
        contact_sheet_path: str = "outputs/head_contact_sheet.jpg",
        max_crops_per_id: int = 6,
        min_frame_gap: int = 18,
        head_height_ratio: float = 0.24,
        head_width_shrink: float = 0.18,
        crop_padding: float = 0.18,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.contact_sheet_path = Path(contact_sheet_path)
        self.max_crops_per_id = max_crops_per_id
        self.min_frame_gap = min_frame_gap
        self.head_height_ratio = head_height_ratio
        self.head_width_shrink = head_width_shrink
        self.crop_padding = crop_padding

        self.saved_counts: dict[int, int] = defaultdict(int)
        self.last_saved_frame: dict[int, int] = defaultdict(lambda: -10**9)
        self.saved_paths: dict[int, list[Path]] = defaultdict(list)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def estimate_head_bbox(
        self,
        bbox: tuple[float, float, float, float],
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]

        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)

        head_h = int(box_h * self.head_height_ratio)
        shrink_x = int(box_w * self.head_width_shrink)

        hx1 = x1 + shrink_x
        hx2 = x2 - shrink_x
        hy1 = y1
        hy2 = y1 + head_h

        return (
            max(0, min(frame_w - 1, hx1)),
            max(0, min(frame_h - 1, hy1)),
            max(0, min(frame_w, hx2)),
            max(0, min(frame_h, hy2)),
        )

    def save_crop_if_needed(
        self,
        frame: np.ndarray,
        global_id: int,
        track_id: int,
        frame_id: int,
        timestamp: float,
        head_bbox: tuple[int, int, int, int],
    ) -> None:
        if self.saved_counts[global_id] >= self.max_crops_per_id:
            return

        if frame_id - self.last_saved_frame[global_id] < self.min_frame_gap:
            return

        crop = self._crop_with_padding(frame, head_bbox)
        if crop.size == 0:
            return

        global_dir = self.output_dir / f"G{global_id:03d}"
        global_dir.mkdir(parents=True, exist_ok=True)

        output_path = global_dir / f"frame_{frame_id:06d}_T{track_id}.jpg"
        cv2.imwrite(str(output_path), crop)

        self.saved_counts[global_id] += 1
        self.last_saved_frame[global_id] = frame_id
        self.saved_paths[global_id].append(output_path)

    def write_contact_sheet(self) -> None:
        thumbnails = []

        for global_id, paths in sorted(self.saved_paths.items()):
            if not paths:
                continue

            image = cv2.imread(str(paths[0]))
            if image is None:
                continue

            thumb = self._make_thumbnail(image, label=f"G{global_id:03d}")
            thumbnails.append(thumb)

        if not thumbnails:
            return

        sheet = self._tile_images(thumbnails, columns=6)
        self.contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.contact_sheet_path), sheet)

    def _crop_with_padding(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

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
    def _make_thumbnail(
        image: np.ndarray,
        label: str,
        size: tuple[int, int] = (140, 180),
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
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas

    @staticmethod
    def _tile_images(images: list[np.ndarray], columns: int = 6) -> np.ndarray:
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