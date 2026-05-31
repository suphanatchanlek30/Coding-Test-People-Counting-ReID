from __future__ import annotations

import argparse
from collections import Counter

from tqdm import tqdm

from src.config.settings import load_settings
from src.io.video import VideoReader
from src.utils import ensure_dir
from src.vision.tracking import MultiObjectTracker


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    config = settings.data

    ensure_dir(settings.output_dir)

    video_cfg = config["video"]
    detection_cfg = config["detection"]
    tracking_cfg = config["tracking"]

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

    info = reader.info()

    print("Smart Entrance People Analytics")
    print("Step 3: YOLO multi-object tracking test.")
    print()
    print(f"Processing video: {info.path}")
    print(f"FPS: {info.fps:.2f}")
    print(f"Original size: {info.width}x{info.height}")
    print(f"Total frames: {info.total_frames}")
    print(f"Detector model: {detection_cfg['model_name']}")
    print(f"Tracker: {tracking_cfg['tracker_type']}")
    print()

    processed_frames = 0
    total_active_tracks_seen = 0
    max_tracks_in_frame = 0
    track_count_distribution: Counter[int] = Counter()

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

            active_count = len(tracked_objects)

            processed_frames += 1
            total_active_tracks_seen += active_count
            max_tracks_in_frame = max(max_tracks_in_frame, active_count)
            track_count_distribution[active_count] += 1

    finally:
        reader.release()

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

    print()
    print("Multi-object tracking test completed.")
    print(f"Processed frames: {processed_frames}")
    print(f"Raw track IDs found: {len(track_states)}")
    print(f"Valid track IDs with length >= {min_track_length}: {len(valid_tracks)}")
    print(f"Average active tracks per processed frame: {avg_tracks_per_frame:.2f}")
    print(f"Max active tracks in one frame: {max_tracks_in_frame}")

    print()
    print("Track count distribution:")
    for track_count, frame_count in sorted(track_count_distribution.items()):
        print(f"  {track_count} active track(s): {frame_count} frame(s)")

    print()
    print("Top track summaries:")
    for track_id, state in sorted(
        valid_tracks.items(),
        key=lambda item: item[1].hits,
        reverse=True,
    )[:10]:
        print(
            f"  track_id={track_id}, "
            f"hits={state.hits}, "
            f"first_frame={state.first_frame}, "
            f"last_frame={state.last_frame}, "
            f"avg_conf={state.avg_confidence:.2f}"
        )


if __name__ == "__main__":
    main()