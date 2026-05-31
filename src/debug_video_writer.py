from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class DebugVideoWriter:
    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            frame_size,
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {self.output_path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()