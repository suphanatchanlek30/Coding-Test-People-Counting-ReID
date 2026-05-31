from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import cv2
import numpy as np

Direction = Literal["enter", "exit", "unknown"]


class CountableObject(Protocol):
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    frame_id: int
    timestamp: float

    @property
    def center(self) -> tuple[float, float]:
        ...

    @property
    def foot_point(self) -> tuple[float, float]:
        ...


@dataclass
class CountEvent:
    event_id: int
    person_id: int
    event_type: Direction
    frame_id: int
    timestamp: float
    line_crossed: bool
    confidence: float


@dataclass
class CountSummary:
    total_unique_people: int = 0
    enter_count: int = 0
    exit_count: int = 0
    currently_visible_people: int = 0


@dataclass
class TrackCountingState:
    previous_side: float | None = None
    first_center: tuple[float, float] | None = None
    last_center: tuple[float, float] | None = None
    last_seen_frame: int = 0
    last_seen_timestamp: float = 0.0
    last_confidence: float = 0.0
    last_zone: str | None = None
    lost_event_finalized: bool = False
    counted_events: set[str] = field(default_factory=set)
    zone_history: list[str] = field(default_factory=list)


def object_person_id(obj: CountableObject) -> int:
    return int(getattr(obj, "global_id", obj.track_id))


def object_counting_point(obj: CountableObject) -> tuple[float, float]:
    return getattr(obj, "foot_point", obj.center)


class LineCrossingCounter:
    def __init__(
        self,
        line_points: list[list[int]],
        min_movement_pixels: float = 20.0,
        count_once_per_global_id: bool = True,
    ) -> None:
        if len(line_points) != 2:
            raise ValueError("line_points must contain exactly two points")

        self.line_start = tuple(line_points[0])
        self.line_end = tuple(line_points[1])
        self.min_movement_pixels = min_movement_pixels
        self.count_once_per_global_id = count_once_per_global_id
        self.track_states: dict[int, TrackCountingState] = {}
        self.unique_people: set[int] = set()
        self.events: list[CountEvent] = []
        self.next_event_id = 1

    def update(
        self,
        tracked_objects: list[CountableObject],
        frame_id: int | None = None,
        timestamp: float | None = None,
        max_missing_frames: int = 8,
    ) -> CountSummary:
        visible_ids = set()

        for obj in tracked_objects:
            person_id = object_person_id(obj)
            visible_ids.add(person_id)
            self.unique_people.add(person_id)

            state = self.track_states.setdefault(person_id, TrackCountingState())
            current_center = object_counting_point(obj)
            current_side = self._point_side(current_center)

            if state.first_center is None:
                state.first_center = current_center

            previous_side = state.previous_side
            state.last_center = current_center
            state.previous_side = current_side

            if previous_side is None:
                continue

            if previous_side * current_side >= 0:
                continue

            if not self._has_enough_movement(state.first_center, current_center):
                continue

            direction = self._infer_direction(previous_side, current_side)
            event_key = (
                direction
                if self.count_once_per_global_id
                else f"{direction}:{len(self.events)}"
            )

            if event_key in state.counted_events:
                continue

            state.counted_events.add(event_key)
            self._add_event(
                person_id=person_id,
                event_type=direction,
                frame_id=obj.frame_id,
                timestamp=obj.timestamp,
                confidence=obj.confidence,
            )

        return self.summary(currently_visible_people=len(visible_ids))

    def summary(self, currently_visible_people: int = 0) -> CountSummary:
        enter_count = sum(1 for event in self.events if event.event_type == "enter")
        exit_count = sum(1 for event in self.events if event.event_type == "exit")

        return CountSummary(
            total_unique_people=len(self.unique_people),
            enter_count=enter_count,
            exit_count=exit_count,
            currently_visible_people=currently_visible_people,
        )

    def get_events(self) -> list[CountEvent]:
        return self.events

    def _add_event(
        self,
        person_id: int,
        event_type: Direction,
        frame_id: int,
        timestamp: float,
        confidence: float,
    ) -> None:
        self.events.append(
            CountEvent(
                event_id=self.next_event_id,
                person_id=person_id,
                event_type=event_type,
                frame_id=frame_id,
                timestamp=timestamp,
                line_crossed=True,
                confidence=confidence,
            )
        )
        self.next_event_id += 1

    def _point_side(self, point: tuple[float, float]) -> float:
        x, y = point
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

    @staticmethod
    def _infer_direction(previous_side: float, current_side: float) -> Direction:
        if previous_side < 0 and current_side > 0:
            return "enter"

        if previous_side > 0 and current_side < 0:
            return "exit"

        return "unknown"

    def _has_enough_movement(
        self,
        first_center: tuple[float, float] | None,
        current_center: tuple[float, float],
    ) -> bool:
        if first_center is None:
            return False

        dx = current_center[0] - first_center[0]
        dy = current_center[1] - first_center[1]
        return (dx * dx + dy * dy) ** 0.5 >= self.min_movement_pixels


class ZoneSequenceCounter:
    def __init__(
        self,
        zones: dict,
        min_movement_pixels: float = 20.0,
        count_once_per_global_id: bool = True,
    ) -> None:
        self.zone_polygons = {
            zone_name: np.array(zone_cfg["polygon"], dtype=np.int32)
            for zone_name, zone_cfg in zones.items()
        }
        self.zone_bounds = {
            zone_name: cv2.boundingRect(polygon)
            for zone_name, polygon in self.zone_polygons.items()
        }
        self.door_center_x = self._zone_center_x("door")
        self.inside_max_x = self._zone_max_x("inside")
        self.outside_min_x = self._zone_min_x("outside")
        self.min_movement_pixels = min_movement_pixels
        self.count_once_per_global_id = count_once_per_global_id
        self.track_states: dict[int, TrackCountingState] = {}
        self.unique_people: set[int] = set()
        self.events: list[CountEvent] = []
        self.next_event_id = 1

    def update(
        self,
        tracked_objects: list[CountableObject],
        frame_id: int | None = None,
        timestamp: float | None = None,
        max_missing_frames: int = 8,
    ) -> CountSummary:
        visible_ids = set()

        for obj in tracked_objects:
            person_id = object_person_id(obj)
            visible_ids.add(person_id)
            self.unique_people.add(person_id)

            state = self.track_states.setdefault(person_id, TrackCountingState())
            current_center = object_counting_point(obj)

            if state.first_center is None:
                state.first_center = current_center

            state.last_center = current_center
            state.last_seen_frame = obj.frame_id
            state.last_seen_timestamp = obj.timestamp
            state.last_confidence = obj.confidence
            state.lost_event_finalized = False

            zone_name = self._get_zone(current_center)
            if zone_name is not None and (
                not state.zone_history or state.zone_history[-1] != zone_name
            ):
                state.zone_history.append(zone_name)
                state.zone_history = state.zone_history[-12:]
                state.last_zone = zone_name

            if not self._has_enough_movement(state.first_center, current_center):
                continue

            direction = self._infer_event_from_zones(state.zone_history)
            if direction == "unknown":
                direction = self._infer_event_from_motion(state, current_center)

            if direction == "unknown":
                continue

            event_key = (
                direction
                if self.count_once_per_global_id
                else f"{direction}:{len(self.events)}"
            )

            if event_key in state.counted_events:
                continue

            state.counted_events.add(event_key)
            self._add_event(
                person_id=person_id,
                event_type=direction,
                frame_id=obj.frame_id,
                timestamp=obj.timestamp,
                confidence=obj.confidence,
            )

        if frame_id is not None:
            self._finalize_lost_door_tracks(
                visible_ids=visible_ids,
                frame_id=frame_id,
                timestamp=timestamp,
                max_missing_frames=max_missing_frames,
            )

        return self.summary(currently_visible_people=len(visible_ids))

    def summary(self, currently_visible_people: int = 0) -> CountSummary:
        enter_count = sum(1 for event in self.events if event.event_type == "enter")
        exit_count = sum(1 for event in self.events if event.event_type == "exit")

        return CountSummary(
            total_unique_people=len(self.unique_people),
            enter_count=enter_count,
            exit_count=exit_count,
            currently_visible_people=currently_visible_people,
        )

    def get_events(self) -> list[CountEvent]:
        return self.events

    def _get_zone(self, point: tuple[float, float]) -> str | None:
        x, y = point

        for zone_name in ("door", "outside", "inside"):
            polygon = self.zone_polygons.get(zone_name)
            if polygon is None:
                continue

            if cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0:
                return zone_name

        return None

    def _infer_event_from_motion(
        self,
        state: TrackCountingState,
        current_center: tuple[float, float],
    ) -> Direction:
        if state.first_center is None:
            return "unknown"

        first_x = state.first_center[0]
        current_x = current_center[0]
        dx = current_x - first_x
        min_horizontal_motion = max(self.min_movement_pixels * 3.0, 90.0)

        if abs(dx) < min_horizontal_motion:
            return "unknown"

        first_zone = state.zone_history[0] if state.zone_history else None
        last_zone = state.zone_history[-1] if state.zone_history else None

        if first_zone == "outside" and current_x <= self.door_center_x:
            return "enter"

        if first_zone == "inside" and current_x >= self.door_center_x:
            return "exit"

        if (
            self.outside_min_x is not None
            and self.inside_max_x is not None
            and first_x >= self.outside_min_x
            and current_x <= self.inside_max_x
        ):
            return "enter"

        if (
            self.outside_min_x is not None
            and self.inside_max_x is not None
            and first_x <= self.inside_max_x
            and current_x >= self.outside_min_x
        ):
            return "exit"

        if last_zone == "inside" and dx < -min_horizontal_motion:
            return "enter"

        if last_zone == "outside" and dx > min_horizontal_motion:
            return "exit"

        return "unknown"

    def _finalize_lost_door_tracks(
        self,
        visible_ids: set[int],
        frame_id: int,
        timestamp: float | None,
        max_missing_frames: int,
    ) -> None:
        for person_id, state in self.track_states.items():
            if person_id in visible_ids:
                continue

            if state.lost_event_finalized:
                continue

            if state.last_seen_frame <= 0:
                continue

            missing_frames = frame_id - state.last_seen_frame
            if missing_frames < max_missing_frames:
                continue

            direction = self._infer_lost_event(state)
            if direction == "unknown":
                state.lost_event_finalized = True
                continue

            event_key = (
                direction
                if self.count_once_per_global_id
                else f"{direction}:{len(self.events)}"
            )
            if event_key in state.counted_events:
                state.lost_event_finalized = True
                continue

            state.counted_events.add(event_key)
            state.lost_event_finalized = True
            self._add_event(
                person_id=person_id,
                event_type=direction,
                frame_id=state.last_seen_frame,
                timestamp=state.last_seen_timestamp
                if state.last_seen_timestamp > 0
                else (timestamp or 0.0),
                confidence=state.last_confidence,
            )

    def _infer_lost_event(self, state: TrackCountingState) -> Direction:
        compressed = self._compressed_zones(state.zone_history)
        if not compressed:
            return "unknown"

        first_zone = compressed[0]
        last_zone = compressed[-1]

        if last_zone != "door":
            return "unknown"

        if first_zone == "outside":
            return "enter"

        if first_zone == "inside":
            return "exit"

        if state.first_center is None or state.last_center is None:
            return "unknown"

        dx = state.last_center[0] - state.first_center[0]
        if dx < -self.min_movement_pixels:
            return "enter"

        if dx > self.min_movement_pixels:
            return "exit"

        return "unknown"

    def _zone_center_x(self, zone_name: str) -> float:
        x, _, w, _ = self.zone_bounds.get(zone_name, (0, 0, 0, 0))
        return float(x + (w / 2.0))

    def _zone_min_x(self, zone_name: str) -> float | None:
        if zone_name not in self.zone_bounds:
            return None
        x, _, _, _ = self.zone_bounds[zone_name]
        return float(x)

    def _zone_max_x(self, zone_name: str) -> float | None:
        if zone_name not in self.zone_bounds:
            return None
        x, _, w, _ = self.zone_bounds[zone_name]
        return float(x + w)

    @staticmethod
    def _infer_event_from_zones(zone_history: list[str]) -> Direction:
        compressed = ZoneSequenceCounter._compressed_zones(zone_history)
        history = "->".join(compressed)

        if "outside->door->inside" in history or "outside->inside" in history:
            return "enter"

        if "inside->door->outside" in history or "inside->outside" in history:
            return "exit"

        if len(compressed) >= 2:
            first_zone = compressed[0]
            last_zone = compressed[-1]

            if first_zone == "outside" and last_zone == "inside":
                return "enter"

            if first_zone == "inside" and last_zone == "outside":
                return "exit"

            if first_zone == "door" and last_zone == "inside":
                return "enter"

            if first_zone == "door" and last_zone == "outside":
                return "exit"

        return "unknown"

    @staticmethod
    def _compressed_zones(zone_history: list[str]) -> list[str]:
        compressed = []
        for zone_name in zone_history:
            if not compressed or compressed[-1] != zone_name:
                compressed.append(zone_name)
        return compressed

    def _add_event(
        self,
        person_id: int,
        event_type: Direction,
        frame_id: int,
        timestamp: float,
        confidence: float,
    ) -> None:
        self.events.append(
            CountEvent(
                event_id=self.next_event_id,
                person_id=person_id,
                event_type=event_type,
                frame_id=frame_id,
                timestamp=timestamp,
                line_crossed=False,
                confidence=confidence,
            )
        )
        self.next_event_id += 1

    def _has_enough_movement(
        self,
        first_center: tuple[float, float] | None,
        current_center: tuple[float, float],
    ) -> bool:
        if first_center is None:
            return False

        dx = current_center[0] - first_center[0]
        dy = current_center[1] - first_center[1]
        return (dx * dx + dy * dy) ** 0.5 >= self.min_movement_pixels