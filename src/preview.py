from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Protocol

import cv2
import numpy as np

from src.counter import CountSummary


class DrawableObject(Protocol):
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        ...

    @property
    def foot_point(self) -> tuple[float, float]:
        ...


class TrackingPreview:
    def __init__(
        self,
        window_name: str = "People Tracking Preview",
        wait_ms: int = 1,
        max_trajectory_points: int = 30,
    ) -> None:
        self.window_name = window_name
        self.wait_ms = wait_ms
        self.trajectories: dict[int, Deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=max_trajectory_points)
        )

    def draw(
        self,
        frame: np.ndarray,
        tracked_objects: list[DrawableObject],
        frame_id: int,
        count_summary: CountSummary | None = None,
        line_points: list[list[int]] | None = None,
        zones: dict | None = None,
    ) -> np.ndarray:
        canvas = frame.copy()

        if zones is not None:
            self._draw_zones(canvas, zones)

        if line_points is not None:
            self._draw_counting_line(canvas, line_points)

        for obj in tracked_objects:
            self._draw_object(canvas, obj)

        self._draw_overlay(canvas, frame_id, tracked_objects, count_summary)
        return canvas

    def show(self, frame: np.ndarray) -> bool:
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(self.wait_ms) & 0xFF
        return key != ord("q")

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)

    def _draw_object(self, frame: np.ndarray, obj: DrawableObject) -> None:
        x1, y1, x2, y2 = [int(v) for v in obj.bbox]
        global_id = int(getattr(obj, "global_id", obj.track_id))
        color = self._color_for_id(global_id)

        foot_point = (int(obj.foot_point[0]), int(obj.foot_point[1]))
        self.trajectories[global_id].append(foot_point)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        self._draw_trajectory(frame, global_id, color)

        cv2.circle(frame, foot_point, 4, color, -1)
        cv2.circle(frame, foot_point, 7, (8, 8, 8), 2)

        label = f"G{global_id} T{obj.track_id} | det {obj.confidence:.2f}"
        self._draw_label(frame, (x1, max(4, y1 - 26)), label, color)

    def _draw_label(
        self,
        frame: np.ndarray,
        origin: tuple[int, int],
        text: str,
        color: tuple[int, int, int],
    ) -> None:
        x, y = origin
        w, h = 210, 24
        overlay = frame.copy()

        cv2.rectangle(overlay, (x, y), (x + w, y + h), (12, 12, 12), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (x, y), (x + 5, y + h), color, -1)
        cv2.putText(
            frame,
            text,
            (x + 10, y + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_overlay(
        self,
        frame: np.ndarray,
        frame_id: int,
        tracked_objects: list[DrawableObject],
        count_summary: CountSummary | None,
    ) -> None:
        panel_x, panel_y = 14, 14
        panel_w, panel_h = 420, 148

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (12, 12, 12),
            -1,
        )
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (85, 85, 85),
            1,
        )

        cv2.putText(
            frame,
            "IDENTITY STITCHING DEBUG",
            (panel_x + 16, panel_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Frame: {frame_id} | Visible: {len(tracked_objects)}",
            (panel_x + 16, panel_y + 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        if count_summary is not None:
            cv2.putText(
                frame,
                f"Unique(global): {count_summary.total_unique_people}",
                (panel_x + 16, panel_y + 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Enter: {count_summary.enter_count} | Exit: {count_summary.exit_count}",
                (panel_x + 16, panel_y + 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (0, 230, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Currently Visible: {count_summary.currently_visible_people}",
                (panel_x + 16, panel_y + 138),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )

    def _draw_zones(self, frame: np.ndarray, zones: dict) -> None:
        zone_colors = {
            "outside": (0, 180, 255),
            "door": (0, 80, 255),
            "inside": (60, 210, 90),
        }
        overlay = frame.copy()

        for zone_name, zone_cfg in zones.items():
            polygon = np.array(zone_cfg["polygon"], dtype=np.int32)
            color = zone_colors.get(zone_name, (180, 180, 180))
            cv2.fillPoly(overlay, [polygon], color)
            cv2.polylines(frame, [polygon], True, color, 3)

            label_point = tuple(polygon[0])
            cv2.putText(
                frame,
                zone_name.upper(),
                (int(label_point[0]) + 8, int(label_point[1]) + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.addWeighted(overlay, 0.13, frame, 0.87, 0, frame)

    @staticmethod
    def _draw_counting_line(frame: np.ndarray, line_points: list[list[int]]) -> None:
        p1 = tuple(line_points[0])
        p2 = tuple(line_points[1])
        cv2.line(frame, p1, p2, (0, 0, 255), 3)
        cv2.putText(
            frame,
            "COUNTING LINE",
            (p1[0] + 10, p1[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_trajectory(
        self,
        frame: np.ndarray,
        global_id: int,
        color: tuple[int, int, int],
    ) -> None:
        points = list(self.trajectories[global_id])
        if len(points) < 2:
            return

        for i in range(1, len(points)):
            thickness = 1 + int(i / max(1, len(points)) * 2)
            cv2.line(frame, points[i - 1], points[i], color, thickness)

    @staticmethod
    def _color_for_id(person_id: int) -> tuple[int, int, int]:
        colors = [
            (255, 80, 80),
            (80, 255, 80),
            (80, 180, 255),
            (255, 220, 80),
            (220, 80, 255),
            (80, 255, 255),
        ]
        return colors[person_id % len(colors)]