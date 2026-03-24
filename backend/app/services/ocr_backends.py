from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
import io
import math
from pathlib import Path
import re
from typing import Any, Sequence

from PIL import Image, ImageFilter, ImageOps

from app.agents.crop_agent import decode_image_bytes
from app.config import settings


_PATH_ROOTS = tuple(
    root
    for root in (
        Path(__file__).resolve().parents[6],
        Path(__file__).resolve().parents[2],
        Path(__file__).resolve().parents[5],
        Path.cwd(),
    )
    if root.exists()
)
_FRENCH_LANGUAGE_HINTS = {"old_french", "middle_french", "anglo_norman", "french"}
_IBERIAN_LANGUAGE_HINTS = {"spanish", "portuguese", "catalan", "iberian"}
_ITALIAN_LANGUAGE_HINTS = {"italian"}
_LATIN_LANGUAGE_HINTS = {"latin"}
_GERMAN_LANGUAGE_HINTS = {"german", "old_high_german", "middle_high_german"}
_DUTCH_LANGUAGE_HINTS = {"dutch", "old_dutch", "middle_dutch", "flemish"}
_ENGLISH_LANGUAGE_HINTS = {"english", "old_english", "middle_english"}
_ROMANCE_LANGUAGE_HINTS = {"occitan", "old_occitan", "provencal", "romance"}


class OCRBackendError(RuntimeError):
    """Raised when a configured OCR backend cannot recognize a crop."""


@dataclass(frozen=True)
class OCRRecognitionMetadata:
    page_id: str | None
    image_id: str | None
    region_id: str
    label: str | None = None
    bbox_xyxy: list[float] | None = None
    polygon: list[list[float]] | None = None
    language_hint: str | None = None
    script_hint_seed: str | None = None
    year: str | None = None
    place_or_origin: str | None = None
    script_family: str | None = None
    document_type: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class OCRBackendResult:
    text: str
    confidence: float | None
    backend_name: str
    model_name: str
    raw_metadata: dict[str, Any]
    region_id: str
    page_id: str | None = None
    script_hint: str | None = None


@dataclass(frozen=True)
class OCRBackendPlan:
    primary_backend: str
    attempt_backends: tuple[str, ...]
    comparison_backends: tuple[str, ...]


class OCRBackend(ABC):
    backend_name: str

    @abstractmethod
    def recognize(self, crop_b64: str, metadata: OCRRecognitionMetadata) -> OCRBackendResult:
        raise NotImplementedError


def _mean(values: Sequence[float]) -> float | None:
    items = [float(value) for value in values if value is not None]
    if not items:
        return None
    return max(0.0, min(1.0, sum(items) / len(items)))


def _normalize_language_hint(value: str | None) -> str:
    hint = str(value or "").strip().lower()
    if not hint:
        return "unknown"
    compact = re.sub(r"[^a-z0-9]+", "_", hint).strip("_")
    if not compact:
        return "unknown"
    aliases = {
        "fro": "old_french",
        "ancien_francais": "old_french",
        "ancien_french": "old_french",
        "frm": "middle_french",
        "moyen_francais": "middle_french",
        "medieval_french": "middle_french",
        "la": "latin",
        "lat": "latin",
        "medieval_latin": "latin",
        "ecclesiastical_latin": "latin",
        "es": "spanish",
        "spa": "spanish",
        "castilian": "spanish",
        "it": "italian",
        "ita": "italian",
        "pt": "portuguese",
        "por": "portuguese",
        "ca": "catalan",
        "cat": "catalan",
        "nl": "dutch",
        "nld": "dutch",
        "dut": "dutch",
        "ang": "old_english",
        "enm": "middle_english",
        "de": "german",
        "deu": "german",
        "goh": "old_high_german",
        "gmh": "middle_high_german",
        "oci": "occitan",
    }
    return aliases.get(compact, compact)


def select_backend_plan(
    *,
    explicit_backend: str | None,
    language_hint: str | None,
    compare_backends: Sequence[str] | None = None,
) -> OCRBackendPlan:
    backend = str(explicit_backend or "auto").strip().lower() or "auto"
    compare = [str(item).strip().lower() for item in (compare_backends or []) if str(item).strip()]
    if backend != "auto":
        deduped_compare = tuple(item for item in dict.fromkeys(compare) if item != backend)
        return OCRBackendPlan(
            primary_backend=backend,
            attempt_backends=(backend,),
            comparison_backends=deduped_compare,
        )

    hint = _normalize_language_hint(language_hint)
    if hint in _LATIN_LANGUAGE_HINTS:
        attempt_backends = ("kraken_cremma_lat", "kraken_catmus", "kraken_mccatmus")
    elif hint in _FRENCH_LANGUAGE_HINTS:
        attempt_backends = ("kraken_cremma_medieval", "kraken_catmus", "kraken_mccatmus")
    elif hint in _GERMAN_LANGUAGE_HINTS or hint in _DUTCH_LANGUAGE_HINTS or hint in _ENGLISH_LANGUAGE_HINTS:
        attempt_backends = ("kraken_mccatmus", "kraken_catmus")
    elif hint in _IBERIAN_LANGUAGE_HINTS or hint in _ITALIAN_LANGUAGE_HINTS or hint in _ROMANCE_LANGUAGE_HINTS:
        attempt_backends = ("kraken_catmus", "kraken_mccatmus")
    else:
        attempt_backends = ("kraken_catmus", "kraken_mccatmus")

    primary = attempt_backends[0]
    deduped_compare = tuple(item for item in dict.fromkeys(compare) if item not in attempt_backends)
    return OCRBackendPlan(
        primary_backend=primary,
        attempt_backends=attempt_backends,
        comparison_backends=deduped_compare,
    )


def _resolve_model_path(raw_path: str | None, *fallbacks: str | None) -> Path:
    candidates = [str(raw_path or "").strip(), *[str(item or "").strip() for item in fallbacks]]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_absolute():
            return path
        last_candidate: Path | None = None
        for root in _PATH_ROOTS:
            rooted = (root / path).resolve()
            last_candidate = rooted
            if rooted.exists():
                return rooted
        if last_candidate is not None:
            return last_candidate
    raise OCRBackendError("No Kraken model path configured.")


@lru_cache(maxsize=2)
def _load_glmocr_instance(cuda_devices: str):
    """Load and cache a GLM-OCR instance in self-hosted mode."""
    try:
        from glmocr import GlmOcr  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise OCRBackendError("GLM-OCR is not installed: pip install glmocr") from exc
    return GlmOcr(mode="selfhosted", cuda_visible_devices=cuda_devices)


@lru_cache(maxsize=8)
def _load_kraken_model(model_path: str, device: str):
    try:
        from kraken.lib import models
    except Exception as exc:  # pragma: no cover - dependency/runtime issue
        raise OCRBackendError(f"Kraken is not available: {exc}") from exc
    return models.load_any(model_path, train=False, device=device)


def _preprocess_kraken_crop(image: Image.Image) -> Image.Image:
    processed, _meta = _preprocess_kraken_crop_with_metadata(image)
    return processed


def _preprocess_kraken_crop_with_metadata(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    grayscale = ImageOps.grayscale(image)
    source_width, source_height = grayscale.size
    deskew_angle = 0.0
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]

        arr = np.array(ImageOps.autocontrast(grayscale), dtype=np.uint8)
        arr = cv2.fastNlMeansDenoising(arr, None, h=12, templateWindowSize=7, searchWindowSize=21)
        _, probe = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(probe < 250))
        if coords.shape[0] > 32:
            angle = cv2.minAreaRect(coords.astype("float32"))[-1]
            angle = 90 + angle if angle < -45 else angle
            angle = -angle
            if abs(angle) <= 8.0:
                deskew_angle = float(angle)
                pil_work = Image.fromarray(arr).convert("L")
                pil_work = pil_work.rotate(
                    deskew_angle,
                    resample=Image.Resampling.BICUBIC,
                    fillcolor=255,
                )
                arr = np.array(pil_work, dtype=np.uint8)
        arr = cv2.adaptiveThreshold(
            arr,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15,
        )
        processed = Image.fromarray(arr).convert("L")
    except Exception:
        processed = ImageOps.autocontrast(grayscale).filter(ImageFilter.MedianFilter(size=3))
        processed = processed.point(lambda px: 255 if px > 180 else 0).convert("L")

    if processed.height < 96:
        scale = max(2, int(round(96 / max(1, processed.height))))
        processed = processed.resize(
            (processed.width * scale, processed.height * scale),
            Image.Resampling.LANCZOS,
        )
    return processed, {
        "deskew_angle": deskew_angle,
        "source_size": [source_width, source_height],
        "processed_size": [processed.width, processed.height],
        "scale_x": processed.width / max(1, source_width),
        "scale_y": processed.height / max(1, source_height),
    }


def _preprocess_calamari_crop_with_metadata(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    processed, meta = _preprocess_kraken_crop_with_metadata(image)
    if processed.height < 80:
        scale = max(2, int(round(80 / max(1, processed.height))))
        processed = processed.resize(
            (processed.width * scale, processed.height * scale),
            Image.Resampling.LANCZOS,
        )
        meta = {
            **meta,
            "processed_size": [processed.width, processed.height],
            "scale_x": processed.width / max(1, meta["source_size"][0]),
            "scale_y": processed.height / max(1, meta["source_size"][1]),
        }
    return processed, meta


def _preprocess_glmocr_crop(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    rgb = image.convert("RGB")
    source_width, source_height = rgb.size
    processed = ImageOps.autocontrast(rgb)
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]

        arr = np.array(processed)
        arr = cv2.fastNlMeansDenoisingColored(arr, None, 8, 8, 7, 21)
        processed = Image.fromarray(arr).convert("RGB")
    except Exception:
        processed = processed.filter(ImageFilter.MedianFilter(size=3)).convert("RGB")
    short_side = min(processed.width, processed.height)
    if short_side < 256:
        scale = 256 / max(1, short_side)
        processed = processed.resize(
            (max(1, int(round(processed.width * scale))), max(1, int(round(processed.height * scale)))),
            Image.Resampling.LANCZOS,
        )
    return processed, {
        "source_size": [source_width, source_height],
        "processed_size": [processed.width, processed.height],
        "scale_x": processed.width / max(1, source_width),
        "scale_y": processed.height / max(1, source_height),
    }


def _metadata_bbox(metadata: OCRRecognitionMetadata) -> tuple[float, float, float, float]:
    if metadata.polygon:
        xs = [float(point[0]) for point in metadata.polygon]
        ys = [float(point[1]) for point in metadata.polygon]
        return min(xs), min(ys), max(xs), max(ys)
    if metadata.bbox_xyxy:
        x1, y1, x2, y2 = [float(value) for value in metadata.bbox_xyxy[:4]]
        return x1, y1, x2, y2
    return 0.0, 0.0, 1.0, 1.0


def _clip_point(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    max_x = max(0, width - 1)
    max_y = max(0, height - 1)
    return (
        int(round(max(0.0, min(float(max_x), x)))),
        int(round(max(0.0, min(float(max_y), y)))),
    )


def _close_boundary(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []
    if points[0] != points[-1]:
        return [*points, points[0]]
    return points


def _vertical_intersections(boundary: list[tuple[float, float]], sample_x: float) -> list[float]:
    intersections: list[float] = []
    for index in range(len(boundary) - 1):
        x1, y1 = boundary[index]
        x2, y2 = boundary[index + 1]
        if x1 == x2:
            if abs(sample_x - x1) <= 1.0:
                intersections.extend([y1, y2])
            continue
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        if sample_x < min_x or sample_x > max_x:
            continue
        t = (sample_x - x1) / (x2 - x1)
        if 0.0 <= t <= 1.0:
            intersections.append(y1 + t * (y2 - y1))
    return intersections


def _rotate_points(points: list[tuple[float, float]], angle_deg: float, width: int, height: int) -> list[tuple[float, float]]:
    if not points or abs(angle_deg) < 0.01:
        return points
    theta = math.radians(angle_deg)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    rotated: list[tuple[float, float]] = []
    for x, y in points:
        dx = x - center_x
        dy = y - center_y
        rotated.append(
            (
                center_x + dx * cos_theta - dy * sin_theta,
                center_y + dx * sin_theta + dy * cos_theta,
            )
        )
    return rotated


def _local_boundary_from_metadata(metadata: OCRRecognitionMetadata, crop_width: int, crop_height: int) -> list[tuple[float, float]]:
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = _metadata_bbox(metadata)
    bbox_width = max(1.0, bbox_x2 - bbox_x1)
    bbox_height = max(1.0, bbox_y2 - bbox_y1)
    scale_x = crop_width / bbox_width
    scale_y = crop_height / bbox_height

    if metadata.polygon:
        points = [
            ((float(point[0]) - bbox_x1) * scale_x, (float(point[1]) - bbox_y1) * scale_y)
            for point in metadata.polygon
        ]
    else:
        points = [
            (0.0, 0.0),
            (float(crop_width - 1), 0.0),
            (float(crop_width - 1), float(crop_height - 1)),
            (0.0, float(crop_height - 1)),
        ]
    return _close_boundary(points)


def _baseline_from_boundary(boundary: list[tuple[float, float]], crop_width: int, crop_height: int) -> list[tuple[float, float]]:
    if not boundary:
        baseline_y = max(0.0, (crop_height - 1) * 0.8)
        return [(0.0, baseline_y), (float(crop_width - 1), baseline_y)]

    xs = [point[0] for point in boundary]
    min_x = max(0.0, min(xs))
    max_x = min(float(crop_width - 1), max(xs))
    if max_x - min_x < 5.0:
        baseline_y = max(point[1] for point in boundary) - max(1.0, crop_height * 0.12)
        return [(min_x, baseline_y), (max_x, baseline_y)]

    samples = 7
    baseline: list[tuple[float, float]] = []
    for index in range(samples):
        x = min_x + (max_x - min_x) * (index / max(1, samples - 1))
        ys = _vertical_intersections(boundary, x)
        if len(ys) >= 2:
            top = min(ys)
            bottom = max(ys)
            y = top + 0.78 * (bottom - top)
        elif ys:
            y = ys[0]
        else:
            y = max(point[1] for point in boundary) - max(1.0, crop_height * 0.12)
        baseline.append((x, y))
    return baseline


def _build_segmentation_for_crop(metadata: OCRRecognitionMetadata, crop_width: int, crop_height: int, transform: dict[str, Any]):
    from kraken import containers

    boundary = _local_boundary_from_metadata(metadata, crop_width, crop_height)
    baseline = _baseline_from_boundary(boundary, crop_width, crop_height)
    boundary = _rotate_points(boundary, float(transform.get("deskew_angle") or 0.0), crop_width, crop_height)
    baseline = _rotate_points(baseline, float(transform.get("deskew_angle") or 0.0), crop_width, crop_height)

    scale_x = float(transform.get("scale_x") or 1.0)
    scale_y = float(transform.get("scale_y") or 1.0)
    processed_width = int(transform.get("processed_size", [crop_width, crop_height])[0] or crop_width)
    processed_height = int(transform.get("processed_size", [crop_width, crop_height])[1] or crop_height)

    scaled_boundary = [
        _clip_point(x * scale_x, y * scale_y, processed_width, processed_height)
        for x, y in boundary
    ]
    scaled_baseline = [
        _clip_point(x * scale_x, y * scale_y, processed_width, processed_height)
        for x, y in baseline
    ]

    if len(scaled_boundary) < 4:
        scaled_boundary = [
            (0, 0),
            (processed_width - 1, 0),
            (processed_width - 1, processed_height - 1),
            (0, processed_height - 1),
            (0, 0),
        ]
    elif scaled_boundary[0] != scaled_boundary[-1]:
        scaled_boundary.append(scaled_boundary[0])

    if len(scaled_baseline) < 2:
        base_y = max(0, min(processed_height - 1, int(round((processed_height - 1) * 0.8))))
        scaled_baseline = [(0, base_y), (processed_width - 1, base_y)]
    elif math.dist(scaled_baseline[0], scaled_baseline[-1]) < 5.0:
        base_y = scaled_baseline[0][1]
        scaled_baseline = [(0, base_y), (processed_width - 1, base_y)]

    return containers.Segmentation(
        type="baselines",
        imagename=metadata.page_id or metadata.image_id or metadata.region_id,
        text_direction="horizontal-lr",
        script_detection=False,
        lines=[
            containers.BaselineLine(
                id=metadata.region_id,
                baseline=scaled_baseline,
                boundary=scaled_boundary,
            )
        ],
    )


class SAIABackend(OCRBackend):
    backend_name = "saia"

    def __init__(self, *, client: Any, quality_floor: float = 0.60, max_fallbacks: int = 2) -> None:
        self.client = client
        self.quality_floor = quality_floor
        self.max_fallbacks = max_fallbacks

    def recognize(self, crop_b64: str, metadata: OCRRecognitionMetadata) -> OCRBackendResult:
        from app.agents import ocr_agent as ocr_agent_module

        try:
            with Image.open(io.BytesIO(decode_image_bytes(crop_b64))) as image:
                width, height = image.size
        except Exception as exc:  # pragma: no cover - crop already validated upstream
            raise OCRBackendError(f"Could not decode OCR crop for SAIA backend: {exc}") from exc
        try:
            available_models = self.client.list_models()
            preferred_models = ocr_agent_module.resolve_model_preferences()
            candidate_models = ocr_agent_module.choose_models(available_models, preferred_models)
            max_attempts = min(len(candidate_models), 1 + self.max_fallbacks)
            selected_models = candidate_models[:max_attempts]
            if not selected_models:
                raise OCRBackendError("No image-capable SAIA OCR model is available.")

            tile = ocr_agent_module.OCRTile(
                tile_id=metadata.region_id,
                bbox_xyxy=list(metadata.bbox_xyxy or [0.0, 0.0, float(width), float(height)]),
                image_b64=crop_b64,
                width=width,
                height=height,
            )
            latin_lock = ocr_agent_module._latin_lock_from_hint(metadata.language_hint)
            tile_result, fallback_records, fallbacks_used = ocr_agent_module._ocr_tile_with_model_candidates(
                tile=tile,
                saia=self.client,
                selected_models=selected_models,
                quality_floor=self.quality_floor,
                latin_lock=latin_lock,
            )
            return OCRBackendResult(
                text=tile_result.text,
                confidence=tile_result.confidence,
                backend_name=self.backend_name,
                model_name=tile_result.model,
                raw_metadata={
                    "warnings": list(tile_result.warnings),
                    "flags": list(tile_result.flags),
                    "fallbacks": [item.model for item in fallback_records],
                    "fallback_reasons": [item.reason for item in fallback_records],
                    "fallbacks_used": list(fallbacks_used),
                    "script_hint": tile_result.script_hint,
                },
                region_id=metadata.region_id,
                page_id=metadata.page_id,
                script_hint=tile_result.script_hint,
            )
        except OCRBackendError:
            raise
        except Exception as exc:
            raise OCRBackendError(f"SAIA backend failed: {exc}") from exc


class KrakenBackend(OCRBackend):
    def __init__(
        self,
        *,
        backend_name: str,
        model_name: str,
        configured_path: str | None,
        fallback_paths: Sequence[str | None] = (),
    ) -> None:
        self.backend_name = backend_name
        self.model_name = model_name
        self.model_path = _resolve_model_path(configured_path, *fallback_paths)
        self.device = str(settings.kraken_device or "cpu").strip() or "cpu"
        self.pad = max(0, int(settings.kraken_line_padding))

    def recognize(self, crop_b64: str, metadata: OCRRecognitionMetadata) -> OCRBackendResult:
        try:
            from kraken import containers, rpred
        except Exception as exc:  # pragma: no cover - dependency/runtime issue
            raise OCRBackendError(f"Kraken is not available: {exc}") from exc

        if not self.model_path.exists():
            raise OCRBackendError(
                f"Kraken model for backend {self.backend_name} not found: {self.model_path}"
            )

        try:
            with Image.open(io.BytesIO(decode_image_bytes(crop_b64))) as image:
                crop_image = image.convert("RGB")
                crop_width, crop_height = crop_image.size
                processed, transform = _preprocess_kraken_crop_with_metadata(crop_image)
                width, height = processed.size
                network = _load_kraken_model(str(self.model_path), self.device)
                if getattr(network, "seg_type", None) == "baselines":
                    bounds = _build_segmentation_for_crop(metadata, crop_width, crop_height, transform)
                else:
                    bounds = containers.Segmentation(
                        type="bbox",
                        imagename=metadata.page_id or metadata.image_id or metadata.region_id,
                        text_direction="horizontal-lr",
                        script_detection=False,
                        lines=[
                            containers.BBoxLine(
                                id=metadata.region_id,
                                bbox=(0, 0, width, height),
                                text_direction="horizontal-lr",
                            )
                        ],
                    )
                records = list(rpred.rpred(network, processed, bounds, pad=self.pad))
        except Exception as exc:
            raise OCRBackendError(f"Kraken recognition failed for {self.backend_name}: {exc}") from exc

        lines = [str(getattr(record, "prediction", "") or str(record) or "").strip() for record in records]
        lines = [line for line in lines if line]
        confidences = [
            float(score)
            for record in records
            for score in list(getattr(record, "confidences", []) or [])
        ]
        confidence = _mean(confidences)
        return OCRBackendResult(
            text="\n".join(lines).strip(),
            confidence=confidence,
            backend_name=self.backend_name,
            model_name=self.model_name,
            raw_metadata={
                "engine": "kraken",
                "model_path": str(self.model_path),
                "preprocess": "grayscale+denoise+deskew+binarize",
                "deskew_angle": float(transform.get("deskew_angle") or 0.0),
                "segmentation_type": getattr(bounds, "type", "unknown"),
                "record_count": len(records),
                "confidence_count": len(confidences),
                "notes": ["recommended manuscript OCR path using segmentation-driven line crops"],
            },
            region_id=metadata.region_id,
            page_id=metadata.page_id,
            script_hint=None,
        )


class KrakenCatmusBackend(KrakenBackend):
    def __init__(self) -> None:
        super().__init__(
            backend_name="kraken_catmus",
            model_name="CATMuS Medieval",
            configured_path=settings.kraken_catmus_model_path,
            fallback_paths=(settings.kraken_default_recognition_model_path,),
        )


class KrakenMcCatmusBackend(KrakenBackend):
    def __init__(self) -> None:
        super().__init__(
            backend_name="kraken_mccatmus",
            model_name="McCATMuS Medieval",
            configured_path=settings.kraken_mccatmus_model_path,
            fallback_paths=(settings.kraken_catmus_model_path, settings.kraken_default_recognition_model_path),
        )


class KrakenCremmaMedievalBackend(KrakenBackend):
    def __init__(self) -> None:
        super().__init__(
            backend_name="kraken_cremma_medieval",
            model_name="CREMMA Medieval",
            configured_path=settings.kraken_cremma_medieval_model_path,
            fallback_paths=(),
        )


class KrakenCremmaLatBackend(KrakenBackend):
    def __init__(self) -> None:
        super().__init__(
            backend_name="kraken_cremma_lat",
            model_name="CREMMA-Medieval-LAT",
            configured_path=settings.kraken_cremma_lat_model_path,
            fallback_paths=(),
        )


def _looks_like_bastard(script_family: str | None) -> bool:
    value = str(script_family or "").strip().lower()
    return any(token in value for token in ("bastard", "bastarda"))


def _looks_like_gothic(script_family: str | None, document_type: str | None) -> bool:
    script_value = str(script_family or "").strip().lower()
    type_value = str(document_type or "").strip().lower()
    return any(token in script_value for token in ("gothic", "textualis", "bookhand", "rotunda")) or "manuscript" in type_value


def _resolve_calamari_checkpoints(configured_dir: str | None) -> tuple[Path, tuple[str, ...]]:
    model_dir = _resolve_model_path(configured_dir)
    if not model_dir.exists():
        raise OCRBackendError(f"Calamari model directory not found: {model_dir}")
    checkpoint_jsons = sorted(model_dir.glob("*.ckpt.json"))
    checkpoints: list[str] = []
    for checkpoint_json in checkpoint_jsons:
        checkpoint = checkpoint_json.with_suffix("")
        if checkpoint.exists():
            checkpoints.append(str(checkpoint))
    if not checkpoints and model_dir.suffix == ".json":
        checkpoint = model_dir.with_suffix("")
        if checkpoint.exists():
            checkpoints.append(str(checkpoint))
            model_dir = checkpoint.parent
    if not checkpoints:
        raise OCRBackendError(f"No Calamari checkpoints found in {model_dir}")
    return model_dir, tuple(checkpoints)


@lru_cache(maxsize=4)
def _load_calamari_predictor(checkpoints: tuple[str, ...]):
    try:
        from calamari_ocr.ocr.predict.params import PredictorParams  # type: ignore[import-untyped]
        from calamari_ocr.ocr.predict.predictor import MultiPredictor  # type: ignore[import-untyped]
    except ImportError as exc:
        raise OCRBackendError("Calamari OCR is not installed: pip install calamari-ocr") from exc
    params = PredictorParams()
    params.progress_bar = False
    return MultiPredictor.from_paths(
        checkpoints=list(checkpoints),
        predictor_params=params,
    )


class CalamariBackend(OCRBackend):
    backend_name = "calamari"

    def _select_model_dir(self, metadata: OCRRecognitionMetadata) -> str:
        language_hint = _normalize_language_hint(metadata.language_hint)
        script_value = str(metadata.script_family or "").strip().lower()
        document_type = str(metadata.document_type or "").strip().lower()
        if language_hint in _FRENCH_LANGUAGE_HINTS:
            return settings.calamari_historical_french_model_dir
        if language_hint in _LATIN_LANGUAGE_HINTS or language_hint in _IBERIAN_LANGUAGE_HINTS or language_hint in _ITALIAN_LANGUAGE_HINTS or language_hint in _ROMANCE_LANGUAGE_HINTS:
            return settings.calamari_gt4histocr_model_dir
        if language_hint in _GERMAN_LANGUAGE_HINTS:
            if "fraktur" in script_value or "blackletter" in script_value:
                return settings.calamari_fraktur_historical_model_dir
            if "antiqua" in script_value or "humanistic" in script_value or "roman" in script_value:
                return settings.calamari_antiqua_historical_model_dir
            return settings.calamari_gt4histocr_model_dir
        if language_hint in _DUTCH_LANGUAGE_HINTS or language_hint in _ENGLISH_LANGUAGE_HINTS:
            if "modern" in document_type and language_hint == "english":
                return settings.calamari_english_model_dir
            return settings.calamari_gt4histocr_model_dir
        if _looks_like_bastard(metadata.script_family):
            return settings.calamari_bastard_model_dir
        if "antiqua" in script_value or "humanistic" in script_value or "roman" in script_value:
            return settings.calamari_antiqua_historical_model_dir
        if "fraktur" in script_value or "blackletter" in script_value:
            return settings.calamari_fraktur_historical_model_dir
        if _looks_like_gothic(metadata.script_family, metadata.document_type):
            return settings.calamari_gothic_model_dir
        return settings.calamari_default_model_dir

    def recognize(self, crop_b64: str, metadata: OCRRecognitionMetadata) -> OCRBackendResult:
        try:
            import numpy as np  # type: ignore[import-not-found]
        except ImportError as exc:
            raise OCRBackendError("numpy is required for Calamari OCR") from exc

        configured_dir = self._select_model_dir(metadata)
        model_dir, checkpoints = _resolve_calamari_checkpoints(configured_dir)
        try:
            with Image.open(io.BytesIO(decode_image_bytes(crop_b64))) as image:
                processed, transform = _preprocess_calamari_crop_with_metadata(image.convert("RGB"))
            predictor = _load_calamari_predictor(checkpoints)
            sample = next(iter(predictor.predict_raw([np.array(processed)], size=1)))
            outputs = getattr(sample, "outputs", None)
            prediction = None
            if isinstance(outputs, tuple) and outputs:
                prediction = outputs[-1]
            elif isinstance(outputs, list) and outputs:
                prediction = outputs[-1]
            else:
                prediction = outputs
            text = str(getattr(prediction, "sentence", "") or "").strip()
            confidence_value = getattr(prediction, "avg_char_probability", None)
            confidence = float(confidence_value) if confidence_value is not None else None
        except OCRBackendError:
            raise
        except StopIteration:
            text = ""
            confidence = None
            transform = {"processed_size": [0, 0]}
        except Exception as exc:
            raise OCRBackendError(f"Calamari recognition failed: {exc}") from exc

        return OCRBackendResult(
            text=text,
            confidence=confidence,
            backend_name=self.backend_name,
            model_name=f"Calamari {model_dir.name}",
            raw_metadata={
                "engine": "calamari",
                "model_dir": str(model_dir),
                "checkpoints": list(checkpoints),
                "preprocess": "grayscale+denoise+deskew+binarize+upscale",
                "processed_size": transform.get("processed_size"),
                "language_hint": metadata.language_hint or "unknown",
                "normalized_language_hint": _normalize_language_hint(metadata.language_hint),
                "script_family": metadata.script_family or "",
                "document_type": metadata.document_type or "",
                "notes": ["historical print baseline; not the strongest default for medieval handwritten manuscripts"],
            },
            region_id=metadata.region_id,
            page_id=metadata.page_id,
            script_hint=None,
        )


class GlmOcrBackend(OCRBackend):
    """GLM-OCR backend — 0.9B vision-language model for manuscript HTR."""

    backend_name = "glmocr"

    @staticmethod
    def _extract_text(result: Any) -> str:
        if hasattr(result, "markdown_result") and result.markdown_result:
            return str(result.markdown_result).strip()
        if hasattr(result, "json_result"):
            jr = result.json_result
            if isinstance(jr, str):
                return jr.strip()
            if isinstance(jr, dict):
                return str(jr.get("text", "") or jr.get("markdown", "") or "").strip()
            return str(jr).strip()
        return str(result).strip()

    def recognize(self, crop_b64: str, metadata: OCRRecognitionMetadata) -> OCRBackendResult:
        try:
            from glmocr import GlmOcr  # type: ignore[import-untyped]
        except ImportError as exc:
            raise OCRBackendError(
                "GLM-OCR is not installed. Install with: pip install glmocr"
            ) from exc

        import tempfile

        try:
            with Image.open(io.BytesIO(decode_image_bytes(crop_b64))) as image:
                crop_image = image.convert("RGB")
            processed, preprocess_meta = _preprocess_glmocr_crop(crop_image)

            # GLM-OCR accepts file paths; write to temp file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                processed.save(tmp, format="PNG")
                tmp_path = tmp.name

            try:
                cuda_devices = str(settings.glmocr_device or "0").strip() or "0"
                ocr = _load_glmocr_instance(cuda_devices)
                prompt = "Text Recognition:"
                if hasattr(ocr, "_pipeline") and not getattr(ocr, "_use_maas", False):
                    request_data = {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"file://{Path(tmp_path).absolute()}"}},
                                ],
                            }
                        ]
                    }
                    results = list(
                        ocr._pipeline.process(  # type: ignore[attr-defined]
                            request_data,
                            save_layout_visualization=False,
                            layout_vis_output_dir=None,
                        )
                    )
                    result = results[0] if results else None
                else:
                    result = ocr.parse(tmp_path, save_layout_visualization=False)
                text = self._extract_text(result) if result is not None else ""
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

        except OCRBackendError:
            raise
        except Exception as exc:
            raise OCRBackendError(f"GLM-OCR recognition failed: {exc}") from exc

        return OCRBackendResult(
            text=text,
            confidence=None,  # GLM-OCR doesn't provide confidence scores
            backend_name=self.backend_name,
            model_name="GLM-OCR-0.9B",
            raw_metadata={
                "engine": "glmocr",
                "device": str(settings.glmocr_device or "0"),
                "preprocess": "rgb+denoise+autocontrast+upscale",
                "prompt": "Text Recognition:",
                "processed_size": preprocess_meta.get("processed_size"),
                "language_hint": metadata.language_hint or "unknown",
                "script_family": metadata.script_family or "",
                "document_type": metadata.document_type or "",
                "notes": ["experimental document-level OCR; not specialized for medieval manuscript line transcription"],
            },
            region_id=metadata.region_id,
            page_id=metadata.page_id,
            script_hint=None,
        )


def build_backend_runtime(
    backend_ids: Sequence[str],
    *,
    saia_client: Any,
    quality_floor: float,
    max_fallbacks: int,
) -> dict[str, OCRBackend]:
    runtime: dict[str, OCRBackend] = {}
    for backend_id in dict.fromkeys(str(item).strip().lower() for item in backend_ids if str(item).strip()):
        if backend_id == "saia":
            runtime[backend_id] = SAIABackend(
                client=saia_client,
                quality_floor=quality_floor,
                max_fallbacks=max_fallbacks,
            )
        elif backend_id == "kraken":
            runtime[backend_id] = KrakenCatmusBackend()
        elif backend_id == "kraken_mccatmus":
            runtime[backend_id] = KrakenMcCatmusBackend()
        elif backend_id == "kraken_catmus":
            runtime[backend_id] = KrakenCatmusBackend()
        elif backend_id == "kraken_cremma_medieval":
            runtime[backend_id] = KrakenCremmaMedievalBackend()
        elif backend_id == "kraken_cremma_lat":
            runtime[backend_id] = KrakenCremmaLatBackend()
        elif backend_id == "calamari":
            runtime[backend_id] = CalamariBackend()
        elif backend_id == "glmocr":
            runtime[backend_id] = GlmOcrBackend()
        else:
            raise OCRBackendError(f"Unsupported OCR backend: {backend_id}")
    return runtime
