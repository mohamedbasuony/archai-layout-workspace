"""
Model prediction pipeline.

Ported from test_combined_models.py — runs 3 YOLO models and combines
their predictions via ImageBatch.
"""

import os
import shutil
import tempfile
from pathlib import Path

from app.core.constants import coco_class_mapping
from app.core.image_batch_classes import ImageBatch


def combine_and_filter_predictions(image_path: str, labels_folders: dict) -> dict:
    """Combine predictions from all 3 models and filter to coco_class_mapping classes.

    Args:
        image_path: Path to the source image.
        labels_folders: Dict with keys 'catmus', 'emanuskript', 'zone'
                        pointing to directories containing JSON predictions.

    Returns:
        COCO-format dict with only the 25 valid classes.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    for name, folder in labels_folders.items():
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Labels folder not found: {folder} ({name})")

    temp_image_dir = tempfile.mkdtemp()
    try:
        image_filename = os.path.basename(image_path)
        temp_image_path = os.path.join(temp_image_dir, image_filename)
        shutil.copy2(image_path, temp_image_path)

        image_batch = ImageBatch(
            image_folder=temp_image_dir,
            catmus_labels_folder=labels_folders["catmus"],
            emanuskript_labels_folder=labels_folders["emanuskript"],
            zone_labels_folder=labels_folders["zone"],
        )

        image_batch.load_images()
        image_batch.load_annotations()
        image_batch.unify_names()

        coco_json = image_batch.return_coco_file()

        # Filter to only valid classes
        valid_category_ids = set(coco_class_mapping.values())
        coco_json["annotations"] = [
            ann for ann in coco_json["annotations"] if ann["category_id"] in valid_category_ids
        ]
        coco_json["categories"] = [
            cat for cat in coco_json["categories"] if cat["id"] in valid_category_ids
        ]

        return coco_json
    finally:
        shutil.rmtree(temp_image_dir, ignore_errors=True)
