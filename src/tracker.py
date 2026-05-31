from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np
from ultralytics import YOLO


@dataclass
class TrackedObject:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    frame_id: int
    timestamp: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def foot_point(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


@dataclass
class TrackState:
    track_id: int
    first_frame: int
    last_frame: int
    first_seen: float
    last_seen: float
    hits: int = 0
    avg_confidence: float = 0.0
    history: Deque[dict] = field(default_factory=lambda: deque(maxlen=120))

    def update(self, obj: TrackedObject) -> None:
        self.last_frame = obj.frame_id
        self.last_seen = obj.timestamp
        self.hits += 1
        self.avg_confidence += (obj.confidence - self.avg_confidence) / self.hits
        self.history.append(
            {
                "frame": obj.frame_id,
                "time": obj.timestamp,
                "bbox": list(obj.bbox),
                "center": list(obj.center),
                "confidence": obj.confidence,
            }
        )


class MultiObjectTracker:
    def __init__(
        self,
        model_name: str,
        tracker_type: str = "botsort",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        min_box_area: float = 1000.0,
    ) -> None:
        self.model = YOLO(model_name)
        self.tracker_yaml = self._resolve_tracker_yaml(tracker_type)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.min_box_area = min_box_area
        self.person_class_id = self._find_person_class_id()

        self.track_states: dict[int, TrackState] = {}
        self.active_track_ids: set[int] = set()
        self.track_history = defaultdict(list)

    def update(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
    ) -> list[TrackedObject]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_yaml,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[self.person_class_id],
            verbose=False,
        )

        tracked_objects: list[TrackedObject] = []
        self.active_track_ids = set()

        if not results:
            return tracked_objects

        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return tracked_objects

        boxes = result.boxes

        for i in range(len(boxes)):
            track_id = int(boxes.id[i].detach().cpu().item())
            class_id = int(boxes.cls[i].detach().cpu().item())
            confidence = float(boxes.conf[i].detach().cpu().item())
            xyxy = boxes.xyxy[i].detach().cpu().numpy().astype(float)

            obj = TrackedObject(
                track_id=track_id,
                bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                confidence=confidence,
                class_id=class_id,
                class_name=self.model.names[class_id],
                frame_id=frame_id,
                timestamp=timestamp,
            )

            if not self._is_valid_track(obj, frame.shape):
                continue

            tracked_objects.append(obj)
            self.active_track_ids.add(track_id)
            self._update_state(obj)

        return tracked_objects

    def get_track_states(self) -> dict[int, TrackState]:
        return self.track_states

    def _update_state(self, obj: TrackedObject) -> None:
        if obj.track_id not in self.track_states:
            self.track_states[obj.track_id] = TrackState(
                track_id=obj.track_id,
                first_frame=obj.frame_id,
                last_frame=obj.frame_id,
                first_seen=obj.timestamp,
                last_seen=obj.timestamp,
            )

        self.track_states[obj.track_id].update(obj)
        self.track_history[obj.track_id].append(
            {
                "frame": obj.frame_id,
                "center": obj.center,
                "bbox": obj.bbox,
                "time": obj.timestamp,
            }
        )

    def _find_person_class_id(self) -> int:
        for class_id, class_name in self.model.names.items():
            if class_name == "person":
                return int(class_id)

        raise RuntimeError("YOLO model does not contain class name: person")

    @staticmethod
    def _resolve_tracker_yaml(tracker_type: str) -> str:
        tracker_type = tracker_type.lower().strip()

        if tracker_type in {"botsort", "bot-sort", "bot_sort"}:
            return "botsort.yaml"

        if tracker_type in {"bytetrack", "byte-track", "byte_track"}:
            return "bytetrack.yaml"

        raise ValueError(
            f"Unsupported tracker_type: {tracker_type}. "
            "Use 'botsort' or 'bytetrack'."
        )

    def _is_valid_track(
        self,
        obj: TrackedObject,
        frame_shape: tuple[int, ...],
    ) -> bool:
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = obj.bbox

        if obj.confidence < self.confidence_threshold:
            return False

        if x2 <= x1 or y2 <= y1:
            return False

        if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
            return False

        area = (x2 - x1) * (y2 - y1)

        if area < self.min_box_area:
            return False

        box_w = x2 - x1
        box_h = y2 - y1
        aspect_ratio = box_h / box_w

        if aspect_ratio < 1.0 or aspect_ratio > 5.0:
            return False

        return True