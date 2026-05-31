from __future__ import annotations

from dataclasses import dataclass

from src.counter import LineCrossingCounter, ZoneSequenceCounter
from src.io.artifacts import HeadVerifier, ResultExporter, ReviewCropExporter
from src.metrics import PerformanceMeter
from src.visualization.renderer import TrackingPreview
from src.vision.attributes import HSVAttributeClassifier
from src.vision.identity import GlobalIDManager
from src.vision.reid import ColorHistogramReID
from src.vision.tracking import MultiObjectTracker


@dataclass
class PipelineComponents:
    tracker: MultiObjectTracker
    global_id_manager: GlobalIDManager
    attribute_classifier: HSVAttributeClassifier
    counter: LineCrossingCounter | ZoneSequenceCounter
    exporter: ResultExporter
    meter: PerformanceMeter
    renderer: TrackingPreview
    preview: TrackingPreview | None
    head_verifier: HeadVerifier | None
    review_exporter: ReviewCropExporter | None


def build_counter(counting_cfg: dict) -> LineCrossingCounter | ZoneSequenceCounter:
    if counting_cfg.get("mode", "line") == "zone":
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


def build_components(config: dict) -> PipelineComponents:
    detection_cfg = config["detection"]
    tracking_cfg = config["tracking"]
    reid_cfg = config["reid"]
    counting_cfg = config["counting"]
    attribute_cfg = config["attribute"]
    preview_cfg = config.get("preview", {})
    head_cfg = config.get("head_verification", {})
    review_cfg = config.get("review_export", {})

    tracker = MultiObjectTracker(
        model_name=detection_cfg["model_name"],
        tracker_type=tracking_cfg["tracker_type"],
        confidence_threshold=detection_cfg["confidence_threshold"],
        iou_threshold=detection_cfg["iou_threshold"],
        min_box_area=detection_cfg["min_box_area"],
    )

    global_id_manager = GlobalIDManager(
        reid=ColorHistogramReID(),
        enabled=reid_cfg.get("enabled", True),
        appearance_threshold=reid_cfg.get("appearance_threshold", 0.68),
        max_time_gap_seconds=reid_cfg.get("max_time_gap_seconds", 5.0),
        max_spatial_distance=reid_cfg.get("max_spatial_distance", 250),
        color_weight=reid_cfg.get("color_weight", 0.60),
        spatial_weight=reid_cfg.get("spatial_weight", 0.25),
        temporal_weight=reid_cfg.get("temporal_weight", 0.15),
        allow_long_gap_reentry=reid_cfg.get("allow_long_gap_reentry", True),
        category_consistency_bonus=reid_cfg.get("category_consistency_bonus", 0.04),
    )

    attribute_classifier = HSVAttributeClassifier(
        blue_ratio_threshold=attribute_cfg.get("blue_ratio_threshold", 0.18),
        black_ratio_threshold=attribute_cfg.get("black_ratio_threshold", 0.35),
        detect_staff_black=attribute_cfg.get("detect_staff_black", False),
    )

    renderer = TrackingPreview(
        window_name="Realtime Annotated Drawer",
        wait_ms=1,
        max_trajectory_points=preview_cfg.get("max_trajectory_points", 30),
    )

    preview = None
    if preview_cfg.get("enabled", False):
        preview = TrackingPreview(
            window_name=preview_cfg.get("window_name", "People Tracking Preview"),
            wait_ms=preview_cfg.get("wait_ms", 1),
            max_trajectory_points=preview_cfg.get("max_trajectory_points", 30),
        )

    head_verifier = None
    if head_cfg.get("enabled", True):
        head_verifier = HeadVerifier(
            output_dir=head_cfg.get("output_dir", "outputs/head_crops"),
            contact_sheet_path=head_cfg.get(
                "contact_sheet_path",
                "outputs/head_contact_sheet.jpg",
            ),
            max_crops_per_id=head_cfg.get("max_crops_per_id", 6),
            min_frame_gap=head_cfg.get("min_frame_gap", 18),
            head_height_ratio=head_cfg.get("head_height_ratio", 0.24),
            head_width_shrink=head_cfg.get("head_width_shrink", 0.18),
            crop_padding=head_cfg.get("crop_padding", 0.18),
        )

    review_exporter = None
    if review_cfg.get("enabled", True):
        review_exporter = ReviewCropExporter(
            output_dir=review_cfg.get("output_dir", "outputs/review_crops"),
            categories=review_cfg.get("categories", ["non_superai", "unknown"]),
            max_crops_per_id=review_cfg.get("max_crops_per_id", 4),
            min_frame_gap=review_cfg.get("min_frame_gap", 24),
            crop_padding=review_cfg.get("crop_padding", 0.08),
        )

    return PipelineComponents(
        tracker=tracker,
        global_id_manager=global_id_manager,
        attribute_classifier=attribute_classifier,
        counter=build_counter(counting_cfg),
        exporter=ResultExporter("outputs"),
        meter=PerformanceMeter(),
        renderer=renderer,
        preview=preview,
        head_verifier=head_verifier,
        review_exporter=review_exporter,
    )