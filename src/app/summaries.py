from __future__ import annotations

from collections import Counter
from typing import Iterable

from src.counter import CountEvent, CountSummary
from src.global_id_manager import GlobalPersonProfile


def live_valid_profiles(
    profiles: dict[int, GlobalPersonProfile],
    min_track_length: int,
    min_avg_confidence: float,
) -> dict[int, GlobalPersonProfile]:
    return {
        global_id: profile
        for global_id, profile in profiles.items()
        if profile.hits >= min_track_length
        and profile.avg_confidence >= min_avg_confidence
    }


def filter_events_for_profiles(
    events: Iterable[CountEvent],
    profiles: dict[int, GlobalPersonProfile],
) -> list[CountEvent]:
    valid_global_ids = set(profiles)
    return [event for event in events if event.person_id in valid_global_ids]


def build_live_summary(
    *,
    valid_profiles: dict[int, GlobalPersonProfile],
    active_global_ids: set[int],
    events: list[CountEvent],
) -> CountSummary:
    valid_events = filter_events_for_profiles(events, valid_profiles)

    return CountSummary(
        total_unique_people=len(valid_profiles),
        enter_count=sum(1 for event in valid_events if event.event_type == "enter"),
        exit_count=sum(1 for event in valid_events if event.event_type == "exit"),
        currently_visible_people=len(active_global_ids & set(valid_profiles)),
    )


def build_final_summary(
    *,
    valid_profiles: dict[int, GlobalPersonProfile],
    valid_events: list[CountEvent],
    currently_visible_people: int,
) -> CountSummary:
    return CountSummary(
        total_unique_people=len(valid_profiles),
        enter_count=sum(1 for event in valid_events if event.event_type == "enter"),
        exit_count=sum(1 for event in valid_events if event.event_type == "exit"),
        currently_visible_people=currently_visible_people,
    )


def category_counts(
    profiles: dict[int, GlobalPersonProfile],
) -> Counter[str]:
    return Counter(profile.final_category()[0] for profile in profiles.values())