from __future__ import annotations

import base64
import io

from PIL import Image, ImageDraw

from app.schemas.agents_ocr import OCRRegionInput


class CropAgentError(RuntimeError):
    """Raised when a region crop cannot be generated."""


def decode_image_bytes(image_b64: str) -> bytes:
    payload = image_b64.strip()
    if payload.startswith("data:image/") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=False)
    except Exception as exc:
        raise CropAgentError("Invalid image_b64 payload.") from exc


def encode_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _bbox_from_polygon(polygon: list[list[float]]) -> tuple[int, int, int, int]:
    xs = [int(round(point[0])) for point in polygon]
    ys = [int(round(point[1])) for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _looks_line_like(label: str | None) -> bool:
    value = str(label or "").strip().lower()
    return any(token in value for token in ("line", "main script", "variant script", "defaultlines", "gloss"))


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    line_like: bool,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    region_w = max(1, x2 - x1)
    region_h = max(1, y2 - y1)
    if line_like:
        pad_x = max(10, int(round(region_w * 0.08)))
        pad_y = max(10, int(round(region_h * 0.45)))
    else:
        pad_x = max(8, int(round(region_w * 0.04)))
        pad_y = max(8, int(round(region_h * 0.12)))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def crop_region(image_b64: str, region: OCRRegionInput, upscale_factor: int = 2) -> tuple[str, str]:
    image_bytes = decode_image_bytes(image_b64)
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            source = image.convert("RGB")
    except Exception as exc:
        raise CropAgentError("Could not decode source image.") from exc

    width, height = source.size

    line_like = _looks_line_like(getattr(region, "label", None))

    if region.polygon is not None:
        # Apply polygon mask first, then crop to polygon bounding box.
        x1, y1, x2, y2 = _bbox_from_polygon(region.polygon)
        mask = Image.new("L", source.size, 0)
        draw = ImageDraw.Draw(mask)
        points = [(int(round(px)), int(round(py))) for px, py in region.polygon]
        draw.polygon(points, fill=255)
        masked = Image.new("RGB", source.size, (255, 255, 255))
        masked.paste(source, mask=mask)
        crop_source = masked
    elif region.bbox_xyxy is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in region.bbox_xyxy]
        crop_source = source
    else:
        raise CropAgentError("Region must include bbox_xyxy or polygon.")

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    x1, y1, x2, y2 = _expand_bbox((x1, y1, x2, y2), width=width, height=height, line_like=line_like)

    crop = crop_source.crop((x1, y1, x2, y2))
    if upscale_factor > 1:
        crop = crop.resize((crop.width * upscale_factor, crop.height * upscale_factor), Image.Resampling.LANCZOS)
    region_id = region.region_id or f"region-{x1}-{y1}-{x2}-{y2}"
    return region_id, encode_png_base64(crop)
