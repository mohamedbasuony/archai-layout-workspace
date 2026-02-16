"""
Core classes for combining YOLO model predictions.

Ported from utils/image_batch_classes.py — handles spatial overlap detection,
class name unification, and COCO format generation.
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from rtree import index
from shapely.geometry import box

from app.core.constants import catmus_zones_mapping, coco_class_mapping


class Annotation:
    def __init__(self, annotation, image):
        self.name = annotation["name"]
        self.cls = annotation["class"]
        self.confidence = annotation["confidence"]
        self.bbox = annotation["box"]
        self.segments = annotation.get("segments")
        self.image = image

    def set_id(self, id):
        self.id = id

    def fix_empty_segments(self, x_coords, y_coords):
        self.segments = {"x": x_coords, "y": y_coords}

    def segments_to_coco_format(self, segment_dict):
        coco_segment = []
        for x, y in zip(segment_dict["x"], segment_dict["y"]):
            coco_segment.append(x)
            coco_segment.append(y)
        return [coco_segment]

    def bbox_to_coco_format(self, box):
        x = box["x1"]
        y = box["y1"]
        width = box["x2"] - box["x1"]
        height = box["y2"] - box["y1"]
        return [x, y, width, height]

    def polygon_area(self, segment_dict):
        x = segment_dict["x"]
        y = segment_dict["y"]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def unify_names(self):
        self.name = catmus_zones_mapping.get(self.name, self.name)

    def to_coco_format(self, current_annotation_id):
        cls_string = catmus_zones_mapping.get(self.name, self.name)
        cls_int = coco_class_mapping[cls_string]

        if self.segments:
            segmentation = self.segments_to_coco_format(self.segments)
            area = self.polygon_area(self.segments)
        else:
            segmentation = []
            width = self.bbox["x2"] - self.bbox["x1"]
            height = self.bbox["y2"] - self.bbox["y1"]
            area = width * height

        return {
            "id": current_annotation_id,
            "image_id": self.image.id,
            "category_id": cls_int,
            "segmentation": segmentation,
            "area": area,
            "bbox": self.bbox_to_coco_format(self.bbox),
            "iscrowd": 0,
            "attributes": {"occluded": False},
        }


class Image:
    def __init__(self, image_path, image_id):
        self.path = image_path
        self.id = image_id
        self.filename = os.path.basename(image_path)
        self.width, self.height = self._get_image_dimensions()
        self.annotations = []
        self.spatial_index = index.Index()
        self.deleted_indices = []
        self.annotations_dict = {}

    def _get_image_dimensions(self):
        with PILImage.open(self.path) as img:
            return img.size

    def process_intersection(
        self, new_box, relevant_classes, overlap_threshold, percentage_dividend, index_to_remove=-1
    ):
        possible_matches = self.spatial_index.intersection(new_box.bounds, objects=True)

        for match in possible_matches:
            if match.object["class"] not in relevant_classes:
                continue

            match_bbox = box(*match.bbox)
            intersection_area = new_box.intersection(match_bbox).area

            if percentage_dividend == "new_box":
                percentage_intersection = intersection_area / new_box.area
            elif percentage_dividend == "match_bbox":
                percentage_intersection = intersection_area / match_bbox.area
            elif percentage_dividend == "symmetric":
                percentage_intersection = min(
                    intersection_area / new_box.area,
                    intersection_area / match_bbox.area,
                )
            else:
                raise ValueError("Invalid percentage_dividend value.")

            if percentage_intersection > overlap_threshold:
                to_remove = index_to_remove if index_to_remove != -1 else match.id
                if to_remove not in self.deleted_indices:
                    self.deleted_indices.append(to_remove)

    def process_defaultline(self, new_box, index):
        possible_matches = list(self.spatial_index.intersection(new_box.bounds, objects=True))
        variant_colored_matches = [
            m
            for m in possible_matches
            if m.object["class"]
            in [
                "Variant script coloured",
                "Variant script black",
                "Main script coloured",
                "NumberingZone",
                "Diagram",
                "MarginTextZone",
                "RunningTitleZone",
                "Table",
                "Quire Mark",
            ]
        ]

        if variant_colored_matches:
            self.deleted_indices.append(index)
        else:
            for match in possible_matches:
                if match.object["class"] == "Main script black":
                    match_bbox = box(*match.bbox)
                    intersection_area = new_box.intersection(match_bbox).area
                    percentage_intersection = intersection_area / match_bbox.area
                    if percentage_intersection > 0.6:
                        self.deleted_indices.append(match.id)

    def add_annotation(self, annotation):
        pos = len(self.annotations)
        minx = annotation.bbox["x1"]
        miny = annotation.bbox["y1"]
        maxx = annotation.bbox["x2"]
        maxy = annotation.bbox["y2"]
        new_box = box(minx, miny, maxx, maxy)

        if annotation.segments:
            if not annotation.segments["x"]:
                x_coords = [minx, minx, maxx, maxx, minx]
                y_coords = [miny, maxy, maxy, miny, miny]
                annotation.fix_empty_segments(x_coords, y_coords)

            if annotation.name in [
                "Main script black",
                "Main script coloured",
                "Variant script black",
                "Variant script coloured",
                "Plain initial- coloured",
                "Plain initial - Highlighted",
                "Plain initial - Black",
            ]:
                self.process_intersection(
                    new_box, ["MarginTextZone", "NumberingZone"], 0.7, "new_box", pos
                )

            if annotation.name in [
                "Embellished",
                "Plain initial- coloured",
                "Plain initial - Highlighted",
                "Plain initial - Black",
                "Inhabited",
            ]:
                self.process_intersection(
                    new_box, ["DropCapitalZone", "GraphicZone"], 0.4, "symmetric"
                )

            if annotation.name == "Page Number":
                self.process_intersection(new_box, ["NumberingZone"], 0.8, "new_box", pos)

            if annotation.name == "Music":
                self.process_intersection(new_box, ["MusicZone", "GraphicZone"], 0.7, "new_box")

            if annotation.name == "Table":
                self.process_intersection(
                    new_box, ["MainZone", "MarginTextZone"], 0.4, "match_bbox"
                )

            if annotation.name in ["Diagram", "Illustrations"]:
                self.process_intersection(new_box, ["GraphicZone"], 0.5, "new_box")

            if annotation.name == "DefaultLine":
                self.process_defaultline(new_box, pos)

        self.annotations.append(annotation)
        annotation.set_id(pos)
        self.spatial_index.insert(pos, new_box.bounds, obj={"class": annotation.name})

    def filter_annotations(self):
        delete_indices_set = set(self.deleted_indices)
        return [
            item
            for idx, item in enumerate(self.annotations)
            if idx not in delete_indices_set
        ]

    def unify_names(self):
        overlapping_classes = ["MainZone", "MarginTextZone"]
        for idx, annotation in enumerate(self.annotations):
            if idx not in self.deleted_indices and annotation.name in overlapping_classes:
                minx = annotation.bbox["x1"]
                miny = annotation.bbox["y1"]
                maxx = annotation.bbox["x2"]
                maxy = annotation.bbox["y2"]
                new_box = box(minx, miny, maxx, maxy)

                possible_matches = self.spatial_index.intersection(
                    new_box.bounds, objects=True
                )

                for match in possible_matches:
                    if (
                        match.id not in self.deleted_indices
                        and match.object["class"] == annotation.name
                        and match.id != idx
                    ):
                        match_bbox = box(*match.bbox)

                        if new_box.area > match_bbox.area:
                            intersection_area = (
                                new_box.intersection(match_bbox).area / match_bbox.area
                            )
                        else:
                            intersection_area = (
                                match_bbox.intersection(new_box).area / new_box.area
                            )

                        if intersection_area > 0.80:
                            delete_index = idx if new_box.area < match_bbox.area else match.id
                            self.deleted_indices.append(delete_index)

            annotation.unify_names()

    def to_coco_image_dict(self):
        return {
            "id": self.id,
            "width": self.width,
            "height": self.height,
            "file_name": self.filename,
        }


class ImageBatch:
    def __init__(self, image_folder, catmus_labels_folder, emanuskript_labels_folder, zone_labels_folder):
        self.image_folder = image_folder
        self.catmus_labels_folder = catmus_labels_folder
        self.emanuskript_labels_folder = emanuskript_labels_folder
        self.zone_labels_folder = zone_labels_folder
        self.images = []

    def load_images(self):
        image_paths = sorted(
            str(p).replace("\\", "/")
            for p in Path(self.image_folder).glob("*")
            if p.is_file()
        )
        for image_id, image_path in enumerate(image_paths, start=1):
            self.images.append(Image(image_path, image_id))

    def load_annotations(self):
        for image in self.images:
            image_basename = os.path.splitext(image.filename)[0]

            catmus_json_path = f"{self.catmus_labels_folder}/{image_basename}.json"
            emanuskript_json_path = f"{self.emanuskript_labels_folder}/{image_basename}.json"
            zone_json_path = f"{self.zone_labels_folder}/{image_basename}.json"

            with open(catmus_json_path) as f:
                catmus_predictions = json.load(f)
            with open(emanuskript_json_path) as f:
                emanuskripts_predictions = json.load(f)
            with open(zone_json_path) as f:
                zone_predictions = json.load(f)

            for annotation_data in zone_predictions + emanuskripts_predictions + catmus_predictions:
                if (
                    annotation_data["name"] == "Variant script black"
                    and len(annotation_data["segments"]["x"]) < 3
                ):
                    continue
                annotation = Annotation(annotation_data, image)
                image.add_annotation(annotation)

    def unify_names(self):
        for image in self.images:
            image.unify_names()

    def create_coco_dict(self):
        return {
            "licenses": [{"name": "", "id": 0, "url": ""}],
            "info": {
                "contributor": "",
                "date_created": "",
                "description": "",
                "url": "",
                "version": "",
                "year": "",
            },
            "categories": [
                {"id": coco_id, "name": cls_name, "supercategory": ""}
                for cls_name, coco_id in coco_class_mapping.items()
            ],
            "annotations": [
                annotation.to_coco_format(annotation_id)
                for image in self.images
                for annotation_id, annotation in enumerate(image.filter_annotations(), start=1)
            ],
            "images": [image.to_coco_image_dict() for image in self.images],
        }

    def return_coco_file(self):
        return self.create_coco_dict()
