from __future__ import annotations

import argparse

from src.config.settings import load_settings
from src.io.video import VideoReader
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart entrance people analytics.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    config = settings.data

    ensure_dir(settings.output_dir)

    video_cfg = config["video"]
    reader = VideoReader(
        video_path=video_cfg["input_path"],
        resize_width=video_cfg.get("resize_width"),
        process_every_n_frames=video_cfg.get("process_every_n_frames", 1),
    )

    try:
        info = reader.info()
        print("Smart Entrance People Analytics")
        print("Step 2: video reader and config loader completed.")
        print()
        print(f"Config: {settings.config_path}")
        print(f"Processing video: {info.path}")
        print(f"FPS: {info.fps:.2f}")
        print(f"Original size: {info.width}x{info.height}")
        print(f"Total frames: {info.total_frames}")

        first_frame = next(reader.frames(), None)
        if first_frame is not None:
            h, w = first_frame.frame.shape[:2]
            print()
            print("First processed frame:")
            print(f"Frame ID: {first_frame.frame_id}")
            print(f"Timestamp: {first_frame.timestamp:.2f}s")
            print(f"Processed size: {w}x{h}")

    finally:
        reader.release()


if __name__ == "__main__":
    main()