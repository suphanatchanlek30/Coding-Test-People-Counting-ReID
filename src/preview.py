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
    category: str
    category_confidence: float
    head_bbox: tuple[int, int, int, int] | None

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
        category_counts: dict[str, int] | None = None,
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

        self._draw_overlay(
            canvas,
            frame_id,
            tracked_objects,
            count_summary,
            category_counts or {},
        )
        self._draw_legend(canvas)
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
        category = getattr(obj, "category", "unknown")
        category_conf = float(getattr(obj, "category_confidence", 0.0))
        color = self._color_for_category(category, global_id)

        foot_point = (int(obj.foot_point[0]), int(obj.foot_point[1]))
        self.trajectories[global_id].append(foot_point)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        self._draw_head_box(frame, obj)
        self._draw_trajectory(frame, global_id, color)

        cv2.circle(frame, foot_point, 4, color, -1)
        cv2.circle(frame, foot_point, 7, (8, 8, 8), 2)

        label = f"G{global_id} T{obj.track_id}"
        sub_label = f"{category.replace('_', ' ')} {category_conf:.2f}"
        self._draw_label(frame, (x1, max(4, y1 - 48)), label, sub_label, color)

    @staticmethod
    def _draw_head_box(frame: np.ndarray, obj: DrawableObject) -> None:
        head_bbox = getattr(obj, "head_bbox", None)
        if head_bbox is None:
            return

        x1, y1, x2, y2 = [int(v) for v in head_bbox]
        if x2 <= x1 or y2 <= y1:
            return

        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 220), 1)

    def _draw_label(
        self,
        frame: np.ndarray,
        origin: tuple[int, int],
        title: str,
        subtitle: str,
        color: tuple[int, int, int],
    ) -> None:
        x, y = origin
        w, h = 168, 40
        y = max(4, y)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (12, 12, 12), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (x, y), (x + 5, y + h), color, -1)

        cv2.putText(frame, title, (x + 12, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (x + 12, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 190), 1, cv2.LINE_AA)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        frame_id: int,
        tracked_objects: list[DrawableObject],
        count_summary: CountSummary | None,
        category_counts: dict[str, int],
    ) -> None:
        panel_x, panel_y = 14, 14
        panel_w, panel_h = 540, 224

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (12, 12, 12), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (85, 85, 85), 1)

        cv2.putText(frame, "LIVE ENTRANCE ANALYTICS", (panel_x + 16, panel_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Frame {frame_id} | Visible {len(tracked_objects)}", (panel_x + 16, panel_y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (205, 205, 205), 1, cv2.LINE_AA)

        if count_summary is not None:
            self._draw_metric_card(frame, panel_x + 16, panel_y + 72, "Realtime Unique", str(count_summary.total_unique_people), (255, 255, 255))
            self._draw_metric_card(frame, panel_x + 188, panel_y + 72, "Enter", str(count_summary.enter_count), (60, 255, 120))
            self._draw_metric_card(frame, panel_x + 360, panel_y + 72, "Exit", str(count_summary.exit_count), (0, 170, 255))
            self._draw_metric_card(frame, panel_x + 16, panel_y + 124, "Visible", str(count_summary.currently_visible_people), (210, 210, 210))

        self._draw_metric_card(frame, panel_x + 188, panel_y + 124, "SuperAI", str(category_counts.get("superai_shirt", 0)), self._category_colors()["superai_shirt"])
        self._draw_metric_card(frame, panel_x + 360, panel_y + 124, "Non-SuperAI", str(category_counts.get("non_superai", 0)), self._category_colors()["non_superai"])
        self._draw_metric_card(frame, panel_x + 16, panel_y + 176, "Unknown", str(category_counts.get("unknown", 0)), self._category_colors()["unknown"])

    @staticmethod
    def _draw_metric_card(frame: np.ndarray, x: int, y: int, label: str, value: str, color: tuple[int, int, int]) -> None:
        cv2.rectangle(frame, (x, y), (x + 174, y + 42), (28, 28, 28), -1)
        cv2.rectangle(frame, (x, y), (x + 174, y + 42), (70, 70, 70), 1)
        cv2.putText(frame, label, (x + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, value, (x + 10, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)

    def _draw_legend(self, frame: np.ndarray) -> None:
        items = [
            ("SuperAI", self._category_colors()["superai_shirt"]),
            ("Visitor", self._category_colors()["non_superai"]),
            ("Unknown", self._category_colors()["unknown"]),
        ]
        x = frame.shape[1] - 210
        y = 18

        cv2.rectangle(frame, (x - 12, y - 8), (frame.shape[1] - 12, y + 86), (12, 12, 12), -1)

        for label, color in items:
            cv2.rectangle(frame, (x, y), (x + 18, y + 18), color, -1)
            cv2.putText(frame, label, (x + 28, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
            y += 26

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
            cv2.putText(frame, zone_name.upper(), (int(label_point[0]) + 8, int(label_point[1]) + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.13, frame, 0.87, 0, frame)

    @staticmethod
    def _draw_counting_line(frame: np.ndarray, line_points: list[list[int]]) -> None:
        p1 = tuple(line_points[0])
        p2 = tuple(line_points[1])
        cv2.line(frame, p1, p2, (0, 0, 255), 3)

    def _draw_trajectory(self, frame: np.ndarray, global_id: int, color: tuple[int, int, int]) -> None:
        points = list(self.trajectories[global_id])
        if len(points) < 2:
            return

        for i in range(1, len(points)):
            thickness = 1 + int(i / max(1, len(points)) * 2)
            cv2.line(frame, points[i - 1], points[i], color, thickness)

    @staticmethod
    def _category_colors() -> dict[str, tuple[int, int, int]]:
        return {
            "superai_shirt": (255, 120, 20),
            "non_superai": (0, 220, 255),
            "staff_black": (245, 245, 245),
            "unknown": (150, 150, 150),
        }

    @staticmethod
    def _color_for_category(category: str, fallback_id: int) -> tuple[int, int, int]:
        colors = TrackingPreview._category_colors()
        if category in colors:
            return colors[category]

        fallback_colors = [
            (255, 80, 80),
            (80, 255, 80),
            (80, 180, 255),
            (255, 220, 80),
            (220, 80, 255),
            (80, 255, 255),
        ]
        return fallback_colors[fallback_id % len(fallback_colors)]