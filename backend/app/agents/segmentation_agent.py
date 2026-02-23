from __future__ import annotations

import tempfile

from app.core.model_runner import combine_and_filter_predictions
from app.services.coco_utils import filter_coco_by_classes, stats_from_coco
from app.services.inference import run_models_parallel
from app.services.visualization import draw_coco_on_image


def run_single_segmentation(
    image_path: str,
    *,
    confidence: float,
    iou: float,
    selected_classes: list[str],
    annotated_output_path: str,
) -> tuple[dict, dict]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        labels_folders = run_models_parallel(image_path, tmp_dir, conf=confidence, iou=iou)
        coco_json = combine_and_filter_predictions(image_path, labels_folders)

    filtered_coco = filter_coco_by_classes(coco_json, selected_classes)
    draw_coco_on_image(image_path, filtered_coco, selected_classes, output_path=annotated_output_path)
    stats = stats_from_coco(filtered_coco)
    return filtered_coco, stats
