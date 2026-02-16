"""Draw COCO annotations on images using matplotlib."""

import hashlib
import os
import tempfile

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from PIL import Image

from app.core.constants import CLASS_COLOR_MAP
from app.services.coco_utils import filter_coco_by_classes

matplotlib.use("Agg")


def _get_class_color(class_name: str) -> str:
    if class_name in CLASS_COLOR_MAP:
        return CLASS_COLOR_MAP[class_name]
    hash_val = int(hashlib.md5(class_name.encode()).hexdigest()[:8], 16)
    hue = (hash_val % 360) / 360.0
    rgb = mcolors.hsv_to_rgb([hue, 0.9, 0.7])
    return mcolors.rgb2hex(rgb)


def _find_label_position(x, y, w, h, existing_positions, img_width, img_height):
    label_w, label_h = 150, 30
    candidates = [
        (x, y - label_h - 5),
        (x, y),
        (x + w - label_w, y),
        (x, y + h + 5),
    ]

    for pos_x, pos_y in candidates:
        if pos_x < 0 or pos_y < 0 or pos_x + label_w > img_width or pos_y + label_h > img_height:
            continue
        overlap = False
        for ex_x, ex_y in existing_positions:
            if abs(pos_x - ex_x) < label_w * 0.8 and abs(pos_y - ex_y) < label_h * 0.8:
                overlap = True
                break
        if not overlap:
            return pos_x, pos_y

    return x, y


def draw_coco_on_image(
    image_path: str,
    coco_json: dict,
    allowed_classes: list[str],
    output_path: str | None = None,
) -> str:
    """Render COCO annotations onto an image and return the output file path.

    Args:
        image_path: Source image path.
        coco_json: Full COCO dict.
        allowed_classes: Class names to draw.
        output_path: Where to save. Auto-generated if None.

    Returns:
        Path to the saved annotated image.
    """
    coco_filtered = filter_coco_by_classes(coco_json, allowed_classes)
    id_to_name = {c["id"]: c["name"] for c in coco_filtered["categories"]}

    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.imshow(img)
    ax.axis("off")

    if not coco_filtered["images"]:
        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), "annotated.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    img_info = coco_filtered["images"][0]
    anns = [a for a in coco_filtered["annotations"] if a["image_id"] == img_info["id"]]

    label_positions = []
    img_width, img_height = img.size

    for ann in anns:
        name = id_to_name[ann["category_id"]]
        color_hex = _get_class_color(name)
        color_rgb = mcolors.hex2color(color_hex)

        segs = ann.get("segmentation", [])
        if segs and isinstance(segs, list) and len(segs[0]) >= 6:
            coords = segs[0]
            xs = coords[0::2]
            ys = coords[1::2]

            poly = Polygon(
                list(zip(xs, ys)),
                closed=True,
                edgecolor=color_rgb,
                facecolor=color_rgb,
                linewidth=2.5,
                alpha=0.3,
            )
            ax.add_patch(poly)
            poly_edge = Polygon(
                list(zip(xs, ys)),
                closed=True,
                edgecolor=color_rgb,
                facecolor="none",
                linewidth=2.5,
                alpha=0.8,
            )
            ax.add_patch(poly_edge)

            min_x, min_y = min(xs), min(ys)
            label_x, label_y = _find_label_position(
                min_x, min_y, max(xs) - min_x, max(ys) - min_y,
                label_positions, img_width, img_height,
            )
        else:
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            rect = Rectangle(
                (x, y), w, h,
                edgecolor=color_rgb, facecolor=color_rgb,
                linewidth=2.5, alpha=0.3,
            )
            ax.add_patch(rect)
            rect_edge = Rectangle(
                (x, y), w, h,
                edgecolor=color_rgb, facecolor="none",
                linewidth=2.5, alpha=0.8,
            )
            ax.add_patch(rect_edge)

            label_x, label_y = _find_label_position(
                x, y, w, h, label_positions, img_width, img_height,
            )

        label_positions.append((label_x, label_y))
        ax.text(
            label_x, label_y, name,
            color="black", fontsize=9, fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=color_rgb,
                linewidth=2,
                alpha=0.7,
            ),
            zorder=10,
        )

    plt.tight_layout()
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), "annotated.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
