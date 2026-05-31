from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoFrame:
    frame_id: int
    timestamp: float
    frame: np.ndarray


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    total_frames: int


class VideoReader:
    def __init__(
        self,
        video_path: str,
        resize_width: int | None = None,
        process_every_n_frames: int = 1,
    ) -> None:
        self.video_path = Path(video_path)
        self.resize_width = resize_width
        self.process_every_n_frames = max(1, process_every_n_frames)

        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {self.video_path}. "
                "Put entrance.mov in the project root or update config.yaml."
            )

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def info(self) -> VideoInfo:
        return VideoInfo(
            path=str(self.video_path),
            fps=self.fps,
            width=self.width,
            height=self.height,
            total_frames=self.total_frames,
        )

    def frames(self) -> Iterator[VideoFrame]:
        frame_id = 0

        while True:
            ok, frame = self.cap.read()

            if not ok:
                break

            if frame_id % self.process_every_n_frames != 0:
                frame_id += 1
                continue

            if self.resize_width:
                frame = self._resize(frame, self.resize_width)

            timestamp = frame_id / self.fps if self.fps > 0 else 0.0

            yield VideoFrame(
                frame_id=frame_id,
                timestamp=timestamp,
                frame=frame,
            )

            frame_id += 1

    def release(self) -> None:
        self.cap.release()

    @staticmethod
    def _resize(frame: np.ndarray, target_width: int) -> np.ndarray:
        h, w = frame.shape[:2]

        if w == target_width:
            return frame

        scale = target_width / w
        target_height = int(h * scale)

        return cv2.resize(frame, (target_width, target_height))