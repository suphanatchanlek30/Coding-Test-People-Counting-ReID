from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.config.settings import load_settings
from src.counter import LineCrossingCounter, ZoneSequenceCounter
from src.debug_video_writer import DebugVideoWriter
from src.io.video import VideoReader
from src.utils import ensure_dir
from src.vision.tracking import MultiObjectTracker
from src.visualization.renderer import TrackingPreview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart entrance people analytics.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit for quick tracking tests.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show OpenCV preview window. Press q to stop.",
    )
    return parser.parse_args()


def build_counter(counting_cfg: dict):
    counting_mode = counting_cfg.get("mode", "line")

    if counting_mode == "zone":
        return ZoneSequenceCounter(
            zones=counting_cfg["zones"],
            min_movement_pixels=counting_cfg.get("min_movement_pixels", 20),
            count_once_per_global_id=counting_cfg.get("count_once_per_global_id", True),
        )

    return LineCrossingCounter(
        line_points=counting_cfg["line_points"],
        min_movement_pixels=counting_cfg.get("min_movement_pixels", 20),
        count_once_per_global_id=counting_cfg.get("count_once_per_global_id", True),
    )


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    config = settings.data

    ensure_dir(settings.output_dir)

    video_cfg = config["video"]
    detection_cfg = config["detection"]
    tracking_cfg = config["tracking"]
    counting_cfg = config["counting"]
    debug_video_cfg = config.get("debug_video", {})

    reader = VideoReader(
        video_path=video_cfg["input_path"],
        resize_width=video_cfg.get("resize_width"),
        process_every_n_frames=video_cfg.get("process_every_n_frames", 1),
    )
    tracker = MultiObjectTracker(
        model_name=detection_cfg["model_name"],
        tracker_type=tracking_cfg["tracker_type"],
        confidence_threshold=detection_cfg["confidence_threshold"],
        iou_threshold=detection_cfg["iou_threshold"],
        min_box_area=detection_cfg["min_box_area"],
    )
    counter = build_counter(counting_cfg)
    renderer = TrackingPreview(
        max_trajectory_points=config.get("preview", {}).get("max_trajectory_points", 30),
    )

    info = reader.info()
    output_video_path = debug_video_cfg.get(
        "output_path",
        "outputs/debug_tracking_preview.mp4",
    )
    counting_mode = counting_cfg.get("mode", "line")
    disappear_after_frames = counting_cfg.get("disappear_after_frames", 8)

    print("Smart Entrance People Analytics")
    print("Step 5: door-zone counting.")
    print()
    print(f"Processing video: {info.path}")
    print(f"FPS: {info.fps:.2f}")
    print(f"Original size: {info.width}x{info.height}")
    print(f"Total frames: {info.total_frames}")
    print(f"Detector model: {detection_cfg['model_name']}")
    print(f"Tracker: {tracking_cfg['tracker_type']}")
    print(f"Counting mode: {counting_mode}")
    print(f"Output video: {output_video_path}")
    print()

    writer = None
    processed_frames = 0
    total_active_tracks_seen = 0
    max_tracks_in_frame = 0
    track_count_distribution: Counter[int] = Counter()
    last_count_summary = None

    try:
        total = args.max_frames or info.total_frames

        for video_frame in tqdm(reader.frames(), total=total):
            if args.max_frames is not None and processed_frames >= args.max_frames:
                break

            tracked_objects = tracker.update(
                frame=video_frame.frame,
                frame_id=video_frame.frame_id,
                timestamp=video_frame.timestamp,
            )

            last_count_summary = counter.update(
                tracked_objects,
                frame_id=video_frame.frame_id,
                timestamp=video_frame.timestamp,
                max_missing_frames=disappear_after_frames,
            )

            annotated_frame = renderer.draw(
                frame=video_frame.frame,
                tracked_objects=tracked_objects,
                frame_id=video_frame.frame_id,
                count_summary=last_count_summary,
                line_points=(
                    counting_cfg["line_points"]
                    if counting_mode == "line"
                    else None
                ),
                zones=counting_cfg.get("zones")
                if counting_mode == "zone"
                else None,
            )

            if writer is None:
                frame_h, frame_w = annotated_frame.shape[:2]
                writer = DebugVideoWriter(
                    output_path=output_video_path,
                    fps=float(debug_video_cfg.get("fps") or info.fps),
                    frame_size=(frame_w, frame_h),
                )

            writer.write(annotated_frame)

            if args.show:
                if not renderer.show(annotated_frame):
                    print("Preview stopped by user.")
                    break

            active_count = len(tracked_objects)

            processed_frames += 1
            total_active_tracks_seen += active_count
            max_tracks_in_frame = max(max_tracks_in_frame, active_count)
            track_count_distribution[active_count] += 1

    finally:
        reader.release()

        if writer is not None:
            writer.release()

        if args.show:
            try:
                renderer.close()
            except Exception:
                pass

    track_states = tracker.get_track_states()
    min_track_length = tracking_cfg["min_track_length"]

    valid_tracks = {
        track_id: state
        for track_id, state in track_states.items()
        if state.hits >= min_track_length
    }

    avg_tracks_per_frame = (
        total_active_tracks_seen / processed_frames if processed_frames > 0 else 0.0
    )

    events = counter.get_events()
    enter_count = sum(1 for event in events if event.event_type == "enter")
    exit_count = sum(1 for event in events if event.event_type == "exit")

    print()
    print("Zone counting test completed.")
    print(f"Processed frames: {processed_frames}")
    print(f"Raw track IDs found: {len(track_states)}")
    print(f"Valid track IDs with length >= {min_track_length}: {len(valid_tracks)}")
    print(f"Average active tracks per processed frame: {avg_tracks_per_frame:.2f}")
    print(f"Max active tracks in one frame: {max_tracks_in_frame}")

    print()
    print("Counting result based on temporary track_id:")
    print(f"* Unique tracks seen: {last_count_summary.total_unique_people if last_count_summary else 0}")
    print(f"* Enter Count: {enter_count}")
    print(f"* Exit Count: {exit_count}")
    print(f"* Events: {len(events)}")

    print()
    print("Saved:")
    print(f"* {Path(output_video_path)}")


if __name__ == "__main__":
    main()