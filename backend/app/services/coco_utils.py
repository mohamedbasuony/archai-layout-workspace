"""COCO JSON utilities — filtering, merging, and statistics."""

from typing import Dict, List

from app.core.constants import coco_class_mapping


def filter_coco_by_classes(coco_json: dict, allowed_classes: List[str]) -> dict:
    """Return a new COCO dict containing only the given class names."""
    id_to_name = {c["id"]: c["name"] for c in coco_json["categories"]}
    allowed_ids = {cid for cid, name in id_to_name.items() if name in allowed_classes}

    filtered_categories = [c for c in coco_json["categories"] if c["id"] in allowed_ids]
    filtered_annotations = [
        a for a in coco_json["annotations"] if a["category_id"] in allowed_ids
    ]

    used_image_ids = {a["image_id"] for a in filtered_annotations}
    filtered_images = [img for img in coco_json["images"] if img["id"] in used_image_ids]

    return {
        **coco_json,
        "categories": filtered_categories,
        "annotations": filtered_annotations,
        "images": filtered_images,
    }


def merge_coco_list(coco_list: List[dict]) -> dict:
    """Merge multiple single-image COCO dicts into one."""
    merged = {
        "info": {"description": "Combined predictions", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": cid, "name": name, "supercategory": ""}
            for name, cid in coco_class_mapping.items()
        ],
    }

    ann_id = 1
    img_id = 1
    for coco in coco_list:
        local_img_id_map = {}
        for img in coco["images"]:
            new_img = dict(img)
            new_img["id"] = img_id
            merged["images"].append(new_img)
            local_img_id_map[img["id"]] = img_id
            img_id += 1

        for ann in coco["annotations"]:
            new_ann = dict(ann)
            new_ann["id"] = ann_id
            new_ann["image_id"] = local_img_id_map.get(ann["image_id"], ann["image_id"])
            merged["annotations"].append(new_ann)
            ann_id += 1

    return merged


def stats_from_coco(coco_json: dict) -> Dict[str, int]:
    """Return {class_name: count} from COCO annotations."""
    id_to_name = {c["id"]: c["name"] for c in coco_json["categories"]}
    counts: Dict[str, int] = {}
    for ann in coco_json["annotations"]:
        name = id_to_name.get(ann["category_id"], f"cls_{ann['category_id']}")
        counts[name] = counts.get(name, 0) + 1
    return counts
