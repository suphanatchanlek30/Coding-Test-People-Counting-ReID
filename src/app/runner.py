from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.app.factories import PipelineComponents, build_components
from src.app.summaries import (
    build_final_summary,
    build_live_summary,
    category_counts,
    filter_events_for_profiles,
    live_valid_profiles,
)
from src.config.settings import AppSettings
from src.counter import CountSummary
from src.domain import GlobalObservation
from src.io.video import DebugVideoWriter, VideoReader
from src.utils import ensure_dir


class PeopleAnalyticsApp:
    def __init__(
        self,
        settings: AppSettings,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings
        self.config = settings.data
        self.logger = logger or logging.getLogger("people_analytics")
        self.components: PipelineComponents | None = None

    def run(self) -> None:
        ensure_dir(self.settings.output_dir)

        video_cfg = self.config["video"]
        tracking_cfg = self.config["tracking"]
        counting_cfg = self.config["counting"]
        export_cfg = self.config["export"]
        debug_video_cfg = self.config.get("debug_video", {})

        reader = VideoReader(
            video_path=video_cfg["input_path"],
            resize_width=video_cfg.get("resize_width"),
            process_every_n_frames=video_cfg.get("process_every_n_frames", 1),
        )
        info = reader.info()
        self.components = build_components(self.config)

        runtime = _RuntimeState(
            output_video_path=video_cfg.get("output_path", "outputs/output_video.mp4"),
            save_video=export_cfg.get("save_video", True),
            debug_video_enabled=debug_video_cfg.get("enabled", False),
            debug_video_max_frames=debug_video_cfg.get("max_frames"),
            counting_mode=counting_cfg.get("mode", "line"),
            min_track_length=tracking_cfg["min_track_length"],
            min_avg_confidence=tracking_cfg.get("min_avg_confidence", 0.35),
            disappear_after_frames=counting_cfg.get("disappear_after_frames", 8),
        )

        self._log_run_header(info, runtime)

        try:
            self._process_video(reader, info, runtime)
        finally:
            self._release_runtime_resources(reader, runtime)

        self._finalize_and_export(info, runtime)

    def _process_video(
        self,
        reader: VideoReader,
        info,
        runtime: "_RuntimeState",
    ) -> None:
        assert self.components is not None
        c = self.components
        counting_cfg = self.config["counting"]
        debug_video_cfg = self.config.get("debug_video", {})

        for video_frame in tqdm(reader.frames(), total=info.total_frames):
            with c.meter.timer("total_frame"):
                with c.meter.timer("tracking"):
                    tracked_objects = c.tracker.update(
                        frame=video_frame.frame,
                        frame_id=video_frame.frame_id,
                        timestamp=video_frame.timestamp,
                    )

                observations: list[GlobalObservation] = []
                active_global_ids: set[int] = set()

                with c.meter.timer("reid_attribute"):
                    for tracked_object in tracked_objects:
                        observation = self._build_observation(
                            video_frame,
                            tracked_object,
                            active_global_ids,
                        )
                        observations.append(observation)
                        active_global_ids.add(observation.global_id)

                with c.meter.timer("counting"):
                    c.counter.update(
                        observations,
                        frame_id=video_frame.frame_id,
                        timestamp=video_frame.timestamp,
                        max_missing_frames=runtime.disappear_after_frames,
                    )

                self._apply_stable_categories(observations, runtime)

                runtime.last_count_summary = build_live_summary(
                    valid_profiles=runtime.valid_profiles,
                    active_global_ids=active_global_ids,
                    events=c.counter.get_events(),
                )

                annotated_frame = self._draw_frame(
                    frame=video_frame.frame,
                    observations=observations,
                    frame_id=video_frame.frame_id,
                    runtime=runtime,
                )

                if runtime.save_video and annotated_frame is not None:
                    if runtime.output_writer is None:
                        frame_h, frame_w = annotated_frame.shape[:2]
                        runtime.output_writer = DebugVideoWriter(
                            output_path=runtime.output_video_path,
                            fps=float(debug_video_cfg.get("fps") or info.fps),
                            frame_size=(frame_w, frame_h),
                        )
                    runtime.output_writer.write(annotated_frame)

                if runtime.debug_video_enabled and annotated_frame is not None:
                    should_write_debug = (
                        runtime.debug_video_max_frames is None
                        or runtime.debug_video_frame_count < runtime.debug_video_max_frames
                    )
                    if should_write_debug:
                        if runtime.debug_writer is None:
                            frame_h, frame_w = annotated_frame.shape[:2]
                            runtime.debug_writer = DebugVideoWriter(
                                output_path=debug_video_cfg.get(
                                    "output_path",
                                    "outputs/debug_tracking_preview.mp4",
                                ),
                                fps=float(debug_video_cfg.get("fps") or info.fps),
                                frame_size=(frame_w, frame_h),
                            )
                        runtime.debug_writer.write(annotated_frame)
                        runtime.debug_video_frame_count += 1

                if c.preview is not None and annotated_frame is not None:
                    if not c.preview.show(annotated_frame):
                        self.logger.info("Preview stopped by user.")
                        break

                runtime.mark_frame(active_count=len(observations))
                c.meter.mark_frame()

    def _build_observation(
        self,
        video_frame,
        tracked_object,
        active_global_ids: set[int],
    ) -> GlobalObservation:
        assert self.components is not None
        c = self.components
        head_cfg = self.config.get("head_verification", {})

        attribute = c.attribute_classifier.classify(
            video_frame.frame,
            tracked_object.bbox,
        )
        observation = c.global_id_manager.update(
            frame=video_frame.frame,
            tracked_object=tracked_object,
            attribute=attribute,
            active_global_ids=active_global_ids,
        )

        if c.head_verifier is not None:
            head_bbox = c.head_verifier.estimate_head_bbox(
                tracked_object.bbox,
                video_frame.frame.shape,
            )
            observation.head_bbox = head_bbox

            if head_cfg.get("save_crops", True):
                c.head_verifier.save_crop_if_needed(
                    frame=video_frame.frame,
                    global_id=observation.global_id,
                    track_id=observation.track_id,
                    frame_id=observation.frame_id,
                    timestamp=observation.timestamp,
                    head_bbox=head_bbox,
                )

        if c.review_exporter is not None:
            c.review_exporter.save_if_needed(
                frame=video_frame.frame,
                category=observation.category,
                global_id=observation.global_id,
                track_id=observation.track_id,
                frame_id=observation.frame_id,
                timestamp=observation.timestamp,
                bbox=observation.bbox,
                confidence=observation.confidence,
                category_confidence=observation.category_confidence,
            )

        return observation

    def _apply_stable_categories(
        self,
        observations: list[GlobalObservation],
        runtime: "_RuntimeState",
    ) -> None:
        assert self.components is not None

        profiles = self.components.global_id_manager.get_profiles()
        runtime.valid_profiles = live_valid_profiles(
            profiles,
            min_track_length=runtime.min_track_length,
            min_avg_confidence=runtime.min_avg_confidence,
        )

        for observation in observations:
            profile = runtime.valid_profiles.get(observation.global_id)
            if profile is None:
                continue

            category, category_confidence = profile.final_category()
            observation.category = category
            observation.category_confidence = category_confidence

    def _draw_frame(
        self,
        frame,
        observations: list[GlobalObservation],
        frame_id: int,
        runtime: "_RuntimeState",
    ):
        assert self.components is not None
        c = self.components
        counting_cfg = self.config["counting"]

        if not (runtime.save_video or runtime.debug_video_enabled or c.preview is not None):
            return None

        with c.meter.timer("visualization"):
            return c.renderer.draw(
                frame=frame,
                tracked_objects=observations,
                frame_id=frame_id,
                count_summary=runtime.last_count_summary,
                category_counts=dict(category_counts(runtime.valid_profiles)),
                line_points=(
                    counting_cfg["line_points"]
                    if runtime.counting_mode == "line"
                    else None
                ),
                zones=counting_cfg.get("zones")
                if runtime.counting_mode == "zone"
                else None,
            )

    def _finalize_and_export(self, info, runtime: "_RuntimeState") -> None:
        assert self.components is not None
        c = self.components

        export_cfg = self.config["export"]
        video_cfg = self.config["video"]
        debug_video_cfg = self.config.get("debug_video", {})
        head_cfg = self.config.get("head_verification", {})
        review_cfg = self.config.get("review_export", {})

        profiles = c.global_id_manager.get_profiles()

        if c.head_verifier is not None:
            c.head_verifier.write_contact_sheet()

        review_contact_sheets = []
        if c.review_exporter is not None:
            review_contact_sheets = c.review_exporter.write_contact_sheets()

        valid_profiles = live_valid_profiles(
            profiles,
            min_track_length=runtime.min_track_length,
            min_avg_confidence=runtime.min_avg_confidence,
        )
        valid_events = filter_events_for_profiles(c.counter.get_events(), valid_profiles)

        final_summary = build_final_summary(
            valid_profiles=valid_profiles,
            valid_events=valid_events,
            currently_visible_people=runtime.last_count_summary.currently_visible_people,
        )
        performance = c.meter.report()
        counted_ids = {event.person_id for event in valid_events}

        if export_cfg.get("save_summary_json", True):
            c.exporter.save_summary(
                path="outputs/summary.json",
                video_path=video_cfg["input_path"],
                method=(
                    "YOLO + BoT-SORT + Appearance ReID "
                    "+ Door-Zone Counting + HSV Attributes"
                ),
                count_summary=final_summary,
                profiles=valid_profiles,
                performance=performance,
                total_frames=info.total_frames,
                processed_frames=runtime.processed_frames,
            )

        if export_cfg.get("save_tracks_csv", True):
            c.exporter.save_tracks("outputs/tracks.csv", valid_profiles, counted_ids)

        if export_cfg.get("save_events_csv", True):
            c.exporter.save_events("outputs/events.csv", valid_events, valid_profiles)

        if export_cfg.get("save_performance_report", True):
            c.exporter.save_performance("outputs/performance_report.json", performance)

        self._log_final_report(
            runtime=runtime,
            final_summary=final_summary,
            performance=performance,
            valid_profiles=valid_profiles,
            saved_files=self._saved_files(
                runtime=runtime,
                debug_video_cfg=debug_video_cfg,
                head_cfg=head_cfg,
                review_cfg=review_cfg,
                review_contact_sheets=review_contact_sheets,
            ),
        )

    def _saved_files(
        self,
        runtime: "_RuntimeState",
        debug_video_cfg: dict,
        head_cfg: dict,
        review_cfg: dict,
        review_contact_sheets: list[str],
    ) -> list[str | Path]:
        assert self.components is not None

        saved_files: list[str | Path] = [
            runtime.output_video_path,
            "outputs/summary.json",
            "outputs/tracks.csv",
            "outputs/events.csv",
            "outputs/performance_report.json",
            "outputs/logs/run.log",
        ]

        if runtime.debug_video_enabled:
            saved_files.append(
                debug_video_cfg.get("output_path", "outputs/debug_tracking_preview.mp4")
            )

        if self.components.head_verifier is not None:
            saved_files.append(
                head_cfg.get("contact_sheet_path", "outputs/head_contact_sheet.jpg")
            )
            saved_files.append(head_cfg.get("output_dir", "outputs/head_crops"))

        if self.components.review_exporter is not None:
            saved_files.append(review_cfg.get("output_dir", "outputs/review_crops"))
            saved_files.extend(review_contact_sheets)

        return saved_files

    def _release_runtime_resources(
        self,
        reader: VideoReader,
        runtime: "_RuntimeState",
    ) -> None:
        reader.release()

        if runtime.output_writer is not None:
            runtime.output_writer.release()

        if runtime.debug_writer is not None:
            runtime.debug_writer.release()

        if self.components and self.components.preview is not None:
            try:
                self.components.preview.close()
            except Exception:
                self.logger.exception("Could not close preview window cleanly.")

    def _log_run_header(self, info, runtime: "_RuntimeState") -> None:
        detection_cfg = self.config["detection"]
        tracking_cfg = self.config["tracking"]
        reid_cfg = self.config["reid"]

        self.logger.info("Smart Entrance People Analytics")
        self.logger.info("Processing video: %s", info.path)
        self.logger.info("FPS: %.2f", info.fps)
        self.logger.info("Original size: %sx%s", info.width, info.height)
        self.logger.info("Total frames: %s", info.total_frames)
        self.logger.info("Detector model: %s", detection_cfg["model_name"])
        self.logger.info("Tracker: %s", tracking_cfg["tracker_type"])
        self.logger.info(
            "ReID enabled: %s (%s)",
            reid_cfg.get("enabled", True),
            reid_cfg.get("method"),
        )
        self.logger.info("Counting mode: %s", runtime.counting_mode)
        self.logger.info("Runtime mode: realtime single pass")
        self.logger.info("Output video: %s", runtime.output_video_path)

    def _log_final_report(
        self,
        *,
        runtime: "_RuntimeState",
        final_summary: CountSummary,
        performance: dict,
        valid_profiles: dict,
        saved_files: list[str | Path],
    ) -> None:
        assert self.components is not None

        track_states = self.components.tracker.get_track_states()
        valid_tracks = {
            track_id: state
            for track_id, state in track_states.items()
            if state.hits >= runtime.min_track_length
        }
        avg_tracks_per_frame = (
            runtime.total_active_tracks_seen / runtime.processed_frames
            if runtime.processed_frames > 0
            else 0.0
        )
        counts = category_counts(valid_profiles)

        self.logger.info("")
        self.logger.info("People analytics completed.")
        self.logger.info("Processed frames: %s", runtime.processed_frames)
        self.logger.info("Raw track IDs found: %s", len(track_states))
        self.logger.info(
            "Valid track IDs with length >= %s: %s",
            runtime.min_track_length,
            len(valid_tracks),
        )
        self.logger.info("Confirmed global IDs: %s", len(valid_profiles))
        self.logger.info("Average active tracks per frame: %.2f", avg_tracks_per_frame)
        self.logger.info("Max active tracks in one frame: %s", runtime.max_tracks_in_frame)

        self.logger.info("")
        self.logger.info("Final Result:")
        self.logger.info("* Total Unique People: %s", final_summary.total_unique_people)
        self.logger.info("* Enter Count: %s", final_summary.enter_count)
        self.logger.info("* Exit Count: %s", final_summary.exit_count)
        self.logger.info("* SuperAI People: %s", counts.get("superai_shirt", 0))
        self.logger.info("* Non-SuperAI People: %s", counts.get("non_superai", 0))
        self.logger.info("* Unknown: %s", counts.get("unknown", 0))

        self.logger.info("")
        self.logger.info("Performance:")
        self.logger.info("* Average FPS: %.2f", performance.get("avg_fps", 0.0))
        self.logger.info(
            "* Average Latency: %.2f ms",
            performance.get("avg_total_time_ms", 0.0),
        )
        self.logger.info("* P95 Latency: %.2f ms", performance.get("p95_latency_ms", 0.0))

        self.logger.info("")
        self.logger.info("Saved:")
        for path in saved_files:
            self.logger.info("* %s", Path(path))


class _RuntimeState:
    def __init__(
        self,
        *,
        output_video_path: str,
        save_video: bool,
        debug_video_enabled: bool,
        debug_video_max_frames: int | None,
        counting_mode: str,
        min_track_length: int,
        min_avg_confidence: float,
        disappear_after_frames: int,
    ) -> None:
        self.output_video_path = output_video_path
        self.save_video = save_video
        self.debug_video_enabled = debug_video_enabled
        self.debug_video_max_frames = debug_video_max_frames
        self.counting_mode = counting_mode
        self.min_track_length = min_track_length
        self.min_avg_confidence = min_avg_confidence
        self.disappear_after_frames = disappear_after_frames

        self.processed_frames = 0
        self.total_active_tracks_seen = 0
        self.max_tracks_in_frame = 0
        self.track_count_distribution: Counter[int] = Counter()
        self.debug_video_frame_count = 0
        self.last_count_summary = CountSummary()
        self.valid_profiles = {}
        self.output_writer: DebugVideoWriter | None = None
        self.debug_writer: DebugVideoWriter | None = None

    def mark_frame(self, active_count: int) -> None:
        self.processed_frames += 1
        self.total_active_tracks_seen += active_count
        self.max_tracks_in_frame = max(self.max_tracks_in_frame, active_count)
        self.track_count_distribution[active_count] += 1