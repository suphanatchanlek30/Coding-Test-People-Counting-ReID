from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.counter import CountEvent, CountSummary
from src.global_id_manager import GlobalPersonProfile


class ResultExporter:
    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_summary(
        self,
        path: str,
        video_path: str,
        method: str,
        count_summary: CountSummary,
        profiles: dict[int, GlobalPersonProfile],
        performance: dict[str, Any],
        total_frames: int,
        processed_frames: int,
    ) -> None:
        category_counts = self._category_counts(profiles)
        data = {
            "video": video_path,
            "method": method,
            "total_unique_people": count_summary.total_unique_people,
            "enter_count": count_summary.enter_count,
            "exit_count": count_summary.exit_count,
            "superai_people": category_counts["superai_shirt"],
            "non_superai_people": category_counts["non_superai"],
            "unknown_people": category_counts["unknown"],
            "average_fps": performance.get("avg_fps", 0.0),
            "total_frames": total_frames,
            "processed_frames": processed_frames,
        }
        self._write_json(path, data)

    def save_tracks(
        self,
        path: str,
        profiles: dict[int, GlobalPersonProfile],
        counted_ids: set[int],
    ) -> None:
        rows = []

        for global_id, profile in sorted(profiles.items()):
            category, category_confidence = profile.final_category()
            rows.append(
                {
                    "global_id": global_id,
                    "track_ids": "|".join(str(track_id) for track_id in sorted(profile.track_ids)),
                    "category": category,
                    "first_seen": profile.first_seen,
                    "last_seen": profile.last_seen,
                    "first_frame": profile.first_frame,
                    "last_frame": profile.last_frame,
                    "counted": global_id in counted_ids,
                    "direction": "event" if global_id in counted_ids else "none",
                    "avg_confidence": profile.avg_confidence,
                    "category_confidence": category_confidence,
                }
            )

        self._write_csv(
            path,
            rows,
            [
                "global_id",
                "track_ids",
                "category",
                "first_seen",
                "last_seen",
                "first_frame",
                "last_frame",
                "counted",
                "direction",
                "avg_confidence",
                "category_confidence",
            ],
        )

    def save_events(
        self,
        path: str,
        events: list[CountEvent],
        profiles: dict[int, GlobalPersonProfile],
    ) -> None:
        rows = []

        for event in events:
            category, category_confidence = self._profile_category(event.person_id, profiles)
            row = asdict(event)
            row["global_id"] = row.pop("person_id")
            row["category"] = category
            row["category_confidence"] = category_confidence
            rows.append(row)

        self._write_csv(
            path,
            rows,
            [
                "event_id",
                "global_id",
                "event_type",
                "category",
                "timestamp",
                "frame_id",
                "line_crossed",
                "confidence",
                "category_confidence",
            ],
        )

    def save_performance(self, path: str, performance: dict[str, Any]) -> None:
        self._write_json(path, performance)

    def _write_json(self, path: str, data: dict[str, Any]) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_csv(
        self,
        path: str,
        rows: list[dict[str, Any]],
        fieldnames: list[str],
    ) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def _category_counts(profiles: dict[int, GlobalPersonProfile]) -> dict[str, int]:
        counts = {
            "superai_shirt": 0,
            "non_superai": 0,
            "unknown": 0,
        }

        for profile in profiles.values():
            category, _ = profile.final_category()
            if category not in counts:
                category = "unknown"
            counts[category] += 1

        return counts

    @staticmethod
    def _profile_category(
        global_id: int,
        profiles: dict[int, GlobalPersonProfile],
    ) -> tuple[str, float]:
        profile = profiles.get(global_id)
        if profile is None:
            return "unknown", 0.0

        return profile.final_category()