from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from statistics import mean


class PerformanceMeter:
    def __init__(self) -> None:
        self.timings_ms: dict[str, list[float]] = defaultdict(list)
        self.processed_frames = 0
        self.start_time = time.perf_counter()

    @contextmanager
    def timer(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.timings_ms[name].append(elapsed_ms)

    def mark_frame(self) -> None:
        self.processed_frames += 1

    def current_fps(self) -> float:
        elapsed = time.perf_counter() - self.start_time
        return self.processed_frames / elapsed if elapsed > 0 else 0.0

    def report(self) -> dict:
        total_elapsed = time.perf_counter() - self.start_time
        total_ms = self.timings_ms.get("total_frame", [])

        report = {
            "processed_frames": self.processed_frames,
            "total_processing_time_seconds": total_elapsed,
            "avg_fps": self.processed_frames / total_elapsed if total_elapsed > 0 else 0.0,
            "avg_total_time_ms": mean(total_ms) if total_ms else 0.0,
            "p95_latency_ms": self._percentile(total_ms, 95),
        }

        for name, values in self.timings_ms.items():
            report[f"avg_{name}_time_ms"] = mean(values) if values else 0.0

        return report

    @staticmethod
    def _percentile(values: list[float], percentile: int) -> float:
        if not values:
            return 0.0

        ordered = sorted(values)
        index = int(round((percentile / 100.0) * (len(ordered) - 1)))
        return float(ordered[index])