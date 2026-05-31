from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from src.attribute_classifier import AttributePrediction
from src.reid import ColorHistogramReID
from src.tracker import TrackedObject


@dataclass
class GlobalObservation:
    global_id: int
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    frame_id: int
    timestamp: float
    category: str
    category_confidence: float
    head_bbox: tuple[int, int, int, int] | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def foot_point(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


@dataclass
class GlobalPersonProfile:
    global_id: int
    track_ids: set[int] = field(default_factory=set)
    feature: np.ndarray | None = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    first_frame: int = 0
    last_frame: int = 0
    last_position: tuple[float, float] = (0.0, 0.0)
    hits: int = 0
    avg_confidence: float = 0.0
    category_votes: Counter[str] = field(default_factory=Counter)
    category_conf_sum: Counter[str] = field(default_factory=Counter)

    def final_category(self) -> tuple[str, float]:
        if not self.category_votes:
            return "unknown", 0.0

        superai_votes = self.category_votes.get("superai_shirt", 0)
        non_superai_votes = self.category_votes.get("non_superai", 0)
        unknown_votes = self.category_votes.get("unknown", 0)
        total_votes = max(1, superai_votes + non_superai_votes + unknown_votes)

        superai_ratio = superai_votes / total_votes
        non_superai_ratio = non_superai_votes / total_votes

        if non_superai_votes > 0:
            non_superai_conf = self.category_conf_sum["non_superai"] / non_superai_votes
            is_confident_non_superai = (
                non_superai_votes >= 2
                and non_superai_ratio >= 0.16
                and non_superai_votes >= max(2, int(superai_votes * 0.45))
                and superai_ratio < 0.42
                and non_superai_conf >= 0.58
            )
            if is_confident_non_superai:
                return "non_superai", float(non_superai_conf)

        if superai_votes > 0:
            superai_conf = self.category_conf_sum["superai_shirt"] / superai_votes
            is_confident_superai = (
                superai_votes >= 4
                or superai_ratio >= 0.32
                or (superai_votes >= 2 and superai_conf >= 0.95)
            )
            if is_confident_superai:
                return "superai_shirt", float(superai_conf)

        if unknown_votes > 0:
            unknown_conf = self.category_conf_sum["unknown"] / unknown_votes
            return "unknown", float(unknown_conf)

        category, votes = self.category_votes.most_common(1)[0]
        confidence_sum = self.category_conf_sum[category]
        return category, float(confidence_sum / max(1, votes))


class GlobalIDManager:
    def __init__(
        self,
        reid: ColorHistogramReID,
        enabled: bool = True,
        appearance_threshold: float = 0.68,
        max_time_gap_seconds: float = 5.0,
        max_spatial_distance: float = 250.0,
        color_weight: float = 0.60,
        spatial_weight: float = 0.25,
        temporal_weight: float = 0.15,
        allow_long_gap_reentry: bool = True,
        category_consistency_bonus: float = 0.04,
    ) -> None:
        self.reid = reid
        self.enabled = enabled
        self.appearance_threshold = appearance_threshold
        self.max_time_gap_seconds = max_time_gap_seconds
        self.max_spatial_distance = max_spatial_distance
        self.color_weight = color_weight
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.allow_long_gap_reentry = allow_long_gap_reentry
        self.category_consistency_bonus = category_consistency_bonus

        self.next_global_id = 1
        self.track_to_global: dict[int, int] = {}
        self.memory: dict[int, GlobalPersonProfile] = {}

    def update(
        self,
        frame,
        tracked_object: TrackedObject,
        attribute: AttributePrediction,
        active_global_ids: set[int] | None = None,
    ) -> GlobalObservation:
        feature = self.reid.extract(frame, tracked_object.bbox)

        if tracked_object.track_id in self.track_to_global:
            global_id = self.track_to_global[tracked_object.track_id]
        else:
            global_id = self._match_or_create_global_id(
                tracked_object=tracked_object,
                feature=feature,
                attribute=attribute,
                active_global_ids=active_global_ids or set(),
            )
            self.track_to_global[tracked_object.track_id] = global_id

        self._update_profile(global_id, tracked_object, feature, attribute)

        return GlobalObservation(
            global_id=global_id,
            track_id=tracked_object.track_id,
            bbox=tracked_object.bbox,
            confidence=tracked_object.confidence,
            class_id=tracked_object.class_id,
            class_name=tracked_object.class_name,
            frame_id=tracked_object.frame_id,
            timestamp=tracked_object.timestamp,
            category=attribute.category,
            category_confidence=attribute.confidence,
        )

    def get_profiles(self) -> dict[int, GlobalPersonProfile]:
        return self.memory

    def _match_or_create_global_id(
        self,
        tracked_object: TrackedObject,
        feature: np.ndarray,
        attribute: AttributePrediction,
        active_global_ids: set[int],
    ) -> int:
        if not self.enabled:
            return self._create_profile_id()

        best_global_id = None
        best_score = 0.0

        for global_id, profile in self.memory.items():
            if global_id in active_global_ids:
                continue

            time_gap = tracked_object.timestamp - profile.last_seen
            if time_gap < 0 or time_gap > self.max_time_gap_seconds:
                continue

            distance = self._distance(tracked_object.center, profile.last_position)

            if (
                not self.allow_long_gap_reentry
                and distance > self.max_spatial_distance
            ):
                continue

            color_score = (
                self.reid.similarity(feature, profile.feature)
                if profile.feature is not None
                else 0.0
            )
            spatial_score = max(0.0, 1.0 - distance / self.max_spatial_distance)
            temporal_score = max(0.0, 1.0 - time_gap / self.max_time_gap_seconds)
            category_score = self._category_consistency_score(profile, attribute)

            score = (
                self.color_weight * color_score
                + self.spatial_weight * spatial_score
                + self.temporal_weight * temporal_score
                + self.category_consistency_bonus * category_score
            )

            if self.allow_long_gap_reentry and distance > self.max_spatial_distance:
                if color_score < self.appearance_threshold + 0.08:
                    continue

            if score > best_score:
                best_score = score
                best_global_id = global_id

        if best_global_id is not None and best_score >= self.appearance_threshold:
            return best_global_id

        return self._create_profile_id()

    def _create_profile_id(self) -> int:
        global_id = self.next_global_id
        self.next_global_id += 1
        return global_id

    def _update_profile(
        self,
        global_id: int,
        tracked_object: TrackedObject,
        feature: np.ndarray,
        attribute: AttributePrediction,
    ) -> None:
        if global_id not in self.memory:
            self.memory[global_id] = GlobalPersonProfile(
                global_id=global_id,
                first_seen=tracked_object.timestamp,
                last_seen=tracked_object.timestamp,
                first_frame=tracked_object.frame_id,
                last_frame=tracked_object.frame_id,
                last_position=tracked_object.center,
            )

        profile = self.memory[global_id]
        profile.track_ids.add(tracked_object.track_id)
        profile.last_seen = tracked_object.timestamp
        profile.last_frame = tracked_object.frame_id
        profile.last_position = tracked_object.center
        profile.hits += 1
        profile.avg_confidence += (
            tracked_object.confidence - profile.avg_confidence
        ) / profile.hits

        if profile.feature is None:
            profile.feature = feature
        else:
            alpha = 0.85
            profile.feature = (
                alpha * profile.feature + (1.0 - alpha) * feature
            ).astype(np.float32)

        profile.category_votes[attribute.category] += 1
        profile.category_conf_sum[attribute.category] += attribute.confidence

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return float((dx * dx + dy * dy) ** 0.5)

    @staticmethod
    def _category_consistency_score(
        profile: GlobalPersonProfile,
        attribute: AttributePrediction,
    ) -> float:
        if not profile.category_votes:
            return 0.0

        category, _ = profile.category_votes.most_common(1)[0]
        if category == attribute.category:
            return 1.0

        if "unknown" in {category, attribute.category}:
            return 0.25

        return -0.5