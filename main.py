from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.config.settings import load_settings
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


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    config = settings.data

    ensure_dir(settings.output_dir)

    video_cfg = config["video"]
    detection_cfg = config["detection"]
    tracking_cfg = config["tracking"]
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
    renderer = TrackingPreview(
        max_trajectory_points=config.get("preview", {}).get("max_trajectory_points", 30),
    )

    info = reader.info()
    output_video_path = debug_video_cfg.get(
        "output_path",
        "outputs/debug_tracking_preview.mp4",
    )

    print("Smart Entrance People Analytics")
    print("Step 4: annotated tracking video preview.")
    print()
    print(f"Processing video: {info.path}")
    print(f"FPS: {info.fps:.2f}")
    print(f"Original size: {info.width}x{info.height}")
    print(f"Total frames: {info.total_frames}")
    print(f"Detector model: {detection_cfg['model_name']}")
    print(f"Tracker: {tracking_cfg['tracker_type']}")
    print(f"Output video: {output_video_path}")
    print()

    writer = None
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

            annotated_frame = renderer.draw(
                frame=video_frame.frame,
                tracked_objects=tracked_objects,
                frame_id=video_frame.frame_id,
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

    print()
    print("Annotated tracking video completed.")
    print(f"Processed frames: {processed_frames}")
    print(f"Raw track IDs found: {len(track_states)}")
    print(f"Valid track IDs with length >= {min_track_length}: {len(valid_tracks)}")
    print(f"Average active tracks per processed frame: {avg_tracks_per_frame:.2f}")
    print(f"Max active tracks in one frame: {max_tracks_in_frame}")

    print()
    print("Saved:")
    print(f"* {Path(output_video_path)}")


if __name__ == "__main__":
    main()