from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import io
import json
import os
import re
from typing import Sequence

from PIL import Image

from app.agents.base import BaseAgent
from app.agents.crop_agent import decode_image_bytes, encode_png_base64
from app.agents.ocr_proofreader_agent import OcrProofreaderAgent
from app.config import settings
from app.schemas.agents_ocr import (
    OCRExtractAnyResponse,
    OCRExtractRequest,
    OCRExtractResponse,
    OCRExtractSimpleResponse,
    OCRFallback,
    OCRProvenance,
    OCRRawOCRPayload,
    OCRRegionResult,
)
from app.services.ocr_evidence import (
    OcrEvidenceRecord,
    now_iso,
    sha256_bytes,
    write_ocr_evidence_jsonl,
)
from app.services.saia_client import SaiaClient, is_model_not_found_error

AGENT_VERSION = "5.0.0"
OCR_PROMPT_VERSION = "paleo_ocr_tiled_v1"
LOCKED_OCR_MODEL = "internvl3.5-30b-a3b"
PIPELINE_VERSION = str(
    os.getenv("ARCHAI_PIPELINE_VERSION")
    or os.getenv("ARCHAI_GIT_SHA")
    or os.getenv("GIT_SHA")
    or "dev"
)
PROMPT_VERSION = OCR_PROMPT_VERSION
OCR_DECODING_TEMPERATURE = 0.0
OCR_DECODING_TOP_P = 1.0
OCR_DECODING_MAX_TOKENS = 3000
OCR_MAX_PIXELS_PER_TILE = max(
    1,
    int(settings.ocr_max_pixels_per_tile or os.getenv("OCR_MAX_PIXELS_PER_TILE", "160000000") or "160000000"),
)
OCR_MAX_LONG_EDGE = max(
    512,
    int(settings.ocr_max_long_edge or os.getenv("OCR_MAX_LONG_EDGE", "12000") or "12000"),
)
OCR_IMAGE_SIZE_RETRY_LIMIT = max(
    0,
    int(settings.ocr_image_size_retry_limit or os.getenv("OCR_IMAGE_SIZE_RETRY_LIMIT", "2") or "2"),
)
OCR_IMAGE_RETRY_SHRINK = float(
    settings.ocr_image_retry_shrink or os.getenv("OCR_IMAGE_RETRY_SHRINK", "0.82") or "0.82"
)
if OCR_IMAGE_RETRY_SHRINK <= 0.0 or OCR_IMAGE_RETRY_SHRINK >= 1.0:
    OCR_IMAGE_RETRY_SHRINK = 0.82

PALEO_OCR_SYSTEM_PROMPT = """You are a professional paleographer producing diplomatic OCR for historical manuscripts.

Core rule: Transcribe ONLY glyphs that are clearly visible in the provided image. NEVER invent, continue, normalize, or “repair” text from memory, context, or known sources. If uncertain, keep what you see and mark uncertainty.

Reading order
- If the page is two-column, output left column top→bottom, then right column top→bottom.
- If single-column, output top→bottom.
- Ignore decorations/frames except when they contain letters.
- Treat a large decorated initial as the first letter(s) of the line’s first word when clearly visible.

Uncertainty markers (MUST be exact)
- Use ? for unclear character(s) inside a word.
- Use […] for unclear spans (one or more words / longer missing sequence).
- NEVER replace […] with guessed content.
- NEVER remove ?/[…] unless the image itself makes the reading deterministic.

Script / character discipline (STRICT)
First decide script_hint from this set:
- insular_old_english
- latin_medieval
- unknown

If script_hint=insular_old_english, avoid drifting into modern Latin-looking substitutions. Use these conventions consistently when clearly visible:
- Use þ for thorn, ð for eth, ƿ for wynn.
- Use 7 for the tironian et / “and” sign when written as that symbol.
- If you see a macron/overline and can represent it, use combining macron (e.g., ā = a + U+0304). If you cannot do it reliably, keep the base letter and add ? or […] rather than inventing expansions.

Do not output characters from other scripts (Greek/Cyrillic/CJK). If you catch yourself about to output non-Latin characters, STOP and use ? / […] instead.

Output contract (NON-NEGOTIABLE)
Return exactly one JSON object with keys: lines, text, script_hint, confidence, warnings.
- lines is an array of strings, one manuscript line per entry.
- text MUST equal "\\n".join(lines).
- confidence is a float 0–1 reflecting how readable the page is overall.
- warnings is an array of strings (may be empty).

If almost nothing is readable: return lines=[] and text="".

If you recognize a famous passage (scripture/boilerplate): treat that recognition as a hallucination risk and only transcribe what is visibly legible; put […] everywhere else."""

PALEO_OCR_USER_PROMPT = '''Return JSON only.

Schema rules:
- Keys must be exactly: lines, text, script_hint, confidence, warnings
- One manuscript line per entry in lines
- text must equal "\\n".join(lines)
- Preserve original spelling, abbreviations, punctuation, capitalization, and line breaks
- Do not translate, normalize, modernize, or expand abbreviations
- Do not output coordinates, markdown, commentary, or extra keys
- Use uncertainty markers exactly: ? and […] (no alternatives)

Quality rules:
- Prefer ? / […] over guessing.
- Do NOT introduce characters from other scripts (Greek/Cyrillic/CJK). If uncertain, use ? / […].
- If page is two-column: output left column fully first, then right column.

If nothing readable is visible, return lines=[] and text="".'''

OCR_JSON_REPAIR_PROMPT = (
    "Return VALID JSON only. Keys must be exactly: lines, text, script_hint, confidence, warnings. "
    "No extra keys. text must equal \"\\n\".join(lines). "
    "script_hint must be one of insular_old_english, latin_medieval, unknown. "
    "Use only ? and […] for uncertainty."
)

ALLOWED_SCRIPT_HINTS = {"insular_old_english", "latin_medieval", "unknown"}
REQUIRED_JSON_KEYS = ("lines", "text", "script_hint", "confidence", "warnings")
REQUIRED_JSON_KEY_SET = set(REQUIRED_JSON_KEYS)
META_TEXT_PATTERNS = (
    r"\bi[' ]?m sorry\b",
    r"\bcannot transcribe\b",
    r"\bunclear image\b",
    r"\bnot clear enough\b",
    r"\bas an ai\b",
)
UNWANTED_SCRIPT_RE = re.compile(r"[\u0370-\u03FF\u0400-\u04FF\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


@dataclass(frozen=True)
class OCRTile:
    tile_id: str
    bbox_xyxy: list[float]
    image_b64: str
    width: int
    height: int
    resized_for_limits: bool = False


@dataclass
class OCRTileResult:
    tile_id: str
    bbox_xyxy: list[float]
    lines: list[str]
    text: str
    confidence: float
    script_hint: str
    warnings: list[str]
    flags: list[str]
    model: str


class OCRAgentError(RuntimeError):
    """Raised when OCR extraction fails."""


class OcrAgent(BaseAgent):
    name = "ocr-agent"

    def __init__(self) -> None:
        self.client = SaiaClient()
        self.upscale_factor = max(
            1,
            int(settings.ocr_crop_upscale or os.getenv("OCR_CROP_UPSCALE", "2") or "2"),
        )

    def run(self, payload: OCRExtractRequest) -> OCRExtractAnyResponse:
        return run_ocr_extraction(payload, client=self.client, upscale_factor=self.upscale_factor)


def _parse_preference_env(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            return []
    return [item.strip() for item in text.split(",") if item.strip()]


def resolve_model_preferences(explicit: Sequence[str] | None = None, prefer_model: str | None = None) -> list[str]:
    _ = explicit
    _ = prefer_model
    return [LOCKED_OCR_MODEL]


def is_vision_model(model_id: str) -> bool:
    key = (model_id or "").lower()
    return any(token in key for token in ("vl", "vision", "internvl", "gemma", "mistral", "omni"))


def choose_models(available_models: Sequence[str], preferred_models: Sequence[str]) -> list[str]:
    by_lower = {model.lower(): model for model in available_models}
    ordered: list[str] = []
    seen: set[str] = set()

    for preferred in preferred_models:
        resolved = by_lower.get(preferred.lower())
        if not resolved:
            continue
        key = resolved.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(resolved)

    if preferred_models:
        return ordered

    if ordered:
        return ordered

    for model in available_models:
        key = model.lower()
        if key in seen:
            continue
        if is_vision_model(model):
            seen.add(key)
            ordered.append(model)

    return ordered


def sanitize_ocr_text(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    lower = text.lower()
    if any(marker in lower for marker in ("no readable text", "nothing readable", "return an empty string")):
        return ""

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, str):
                return parsed.strip()
            if isinstance(parsed, dict):
                for key in ("text", "transcription", "content", "output"):
                    value = parsed.get(key)
                    if isinstance(value, str):
                        return value.strip()
                return ""
            return ""
        except Exception:
            pass

    return text.strip()


def _has_meta_text(text: str) -> bool:
    value = (text or "").lower()
    return any(re.search(pattern, value) for pattern in META_TEXT_PATTERNS)


def _normalize_uncertainty_tokens(value: str) -> str:
    text = value.replace("[...]", "[…]").replace("...", "[…]")
    text = text.replace("…", "[…]")
    text = text.replace("[[…]]", "[…]")
    return text


def _looks_like_pattern_junk(value: str) -> bool:
    text = (value or "").strip().lower()
    if not text:
        return False
    if "ccooocoo" in text:
        return True
    if re.fullmatch(r"\?{6,}", text):
        return True
    if re.search(r"(.)\1{6,}", text):
        return True
    compact = re.sub(r"\s+", "", text)
    if len(compact) >= 10 and len(set(ch for ch in compact if ch.isalpha())) <= 2:
        return True
    return False


def _latin_lock_from_hint(language_hint: str | None) -> bool:
    hint = str(language_hint or "").lower()
    return any(token in hint for token in ("la", "latin", "insular", "old_english", "old english"))


def _sanitize_line_for_latin(line: str, *, latin_lock: bool) -> tuple[str, int]:
    text = NON_PRINTABLE_RE.sub("", str(line or ""))
    text = _normalize_uncertainty_tokens(text)
    text = re.sub(r"\?{6,}", "[… ]", text).replace("[… ]", "[…]")
    replaced = 0

    if latin_lock:
        chars: list[str] = []
        for ch in text:
            if UNWANTED_SCRIPT_RE.match(ch):
                chars.append("?")
                replaced += 1
            else:
                chars.append(ch)
        text = "".join(chars)

    text = re.sub(r"\s+", " ", text).strip()
    return text, replaced


def _normalize_lines(raw_lines: list[str], *, latin_lock: bool) -> tuple[list[str], list[str], float]:
    lines: list[str] = []
    warnings: list[str] = []
    replaced_non_latin = 0
    total_chars = 0

    for raw_line in raw_lines:
        if _has_meta_text(raw_line):
            warnings.append("meta_text_removed")
            continue
        line, replaced = _sanitize_line_for_latin(raw_line, latin_lock=latin_lock)
        replaced_non_latin += replaced
        total_chars += max(1, len(line))
        if not line:
            continue
        if _looks_like_pattern_junk(line):
            warnings.append("pattern_junk_removed")
            continue
        lines.append(line)

    ratio = replaced_non_latin / max(1, total_chars)
    if replaced_non_latin > 0:
        warnings.append("non_latin_chars_replaced")
    return lines, warnings, ratio


def _extract_json_candidate(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return "{}"
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def _detect_script_hint(text: str) -> str:
    value = text or ""
    if UNWANTED_SCRIPT_RE.search(value):
        return "unknown"
    if re.search(r"[þðƿÞÐǷ]", value) or re.search(r"(?<!\d)7(?!\d)", value):
        return "insular_old_english"
    if re.search(r"[A-Za-zÀ-ÿ]", value):
        return "latin_medieval"
    return "unknown"


def _contains_script_drift(text: str) -> bool:
    return bool(UNWANTED_SCRIPT_RE.search(text or ""))


def _has_strict_schema_error(payload: dict[str, object]) -> bool:
    warnings = {str(item) for item in payload.get("warnings") or []}
    return bool({"INVALID_JSON_RESPONSE", "INVALID_SCHEMA_KEYS"} & warnings)


def _has_script_drift_warning(payload: dict[str, object]) -> bool:
    warnings = {str(item) for item in payload.get("warnings") or []}
    return "SCRIPT_DRIFT" in warnings


def _normalize_script_hint_value(value: object) -> tuple[str, bool]:
    hint = str(value or "").strip().lower()
    if hint in ALLOWED_SCRIPT_HINTS:
        return hint, False
    return "unknown", True


def parse_ocr_json(
    raw: str,
    *,
    expected_region_id: str | None = None,
    latin_lock: bool = True,
    strict_keys: bool = False,
) -> dict[str, object]:
    candidate = _extract_json_candidate(raw)
    data: dict[str, object] | None = None

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            data = parsed
    except Exception:
        data = None

    if data is None:
        sanitized = sanitize_ocr_text(raw)
        base_lines = [line for line in sanitized.splitlines() if line.strip()] if sanitized else []
        lines, normalized_warnings, non_latin_ratio = _normalize_lines(base_lines, latin_lock=latin_lock)
        text = "\n".join(lines)
        script_hint = _detect_script_hint(text)
        warnings: list[str] = ["INVALID_JSON_RESPONSE", *normalized_warnings]
        if _contains_script_drift(text):
            warnings.append("SCRIPT_DRIFT")
        return {
            "lines": lines,
            "text": text,
            "script_hint": script_hint,
            "confidence": 0.0 if not text else 0.6,
            "warnings": list(dict.fromkeys(warnings)),
            "non_latin_ratio": non_latin_ratio,
        }

    lines: list[str]
    text: str
    script_hint: str
    confidence_value: object
    warnings_value: object

    warnings: list[str] = []
    if strict_keys:
        incoming_keys = set(data.keys())
        if incoming_keys != REQUIRED_JSON_KEY_SET:
            warnings.append("INVALID_SCHEMA_KEYS")

    if not strict_keys and isinstance(data.get("regions"), list):
        regions = [item for item in data.get("regions", []) if isinstance(item, dict)]
        target: dict[str, object] = {}
        if expected_region_id:
            for item in regions:
                if str(item.get("region_id") or "") == str(expected_region_id):
                    target = item
                    break
        if not target and regions:
            target = regions[0]

        maybe_lines = target.get("lines") if isinstance(target, dict) else None
        lines = [str(item) for item in maybe_lines] if isinstance(maybe_lines, list) else []
        maybe_text = target.get("text") if isinstance(target, dict) else None
        if isinstance(maybe_text, str):
            text = maybe_text
        else:
            full_text = data.get("full_text")
            text = str(full_text) if isinstance(full_text, str) else "\n".join(lines)
        script_hint = str(data.get("document_script_hint") or "unknown").strip().lower()
        confidence_value = target.get("confidence", 0.0) if isinstance(target, dict) else 0.0
        warnings_value = target.get("warnings", []) if isinstance(target, dict) else []
    else:
        maybe_lines = data.get("lines")
        lines = [str(item) for item in maybe_lines] if isinstance(maybe_lines, list) else []
        maybe_text = data.get("text")
        text = str(maybe_text) if isinstance(maybe_text, str) else "\n".join(lines)
        script_hint = str(data.get("script_hint") or "unknown").strip().lower()
        confidence_value = data.get("confidence", 0.0)
        warnings_value = data.get("warnings", [])

    if not lines and text:
        lines = [line for line in text.splitlines() if line.strip()]

    lines, normalized_warnings, non_latin_ratio = _normalize_lines(lines, latin_lock=latin_lock)
    recomputed_text = "\n".join(lines)
    if text != recomputed_text:
        warnings.append("TEXT_JOIN_FIXED")
    text = recomputed_text

    script_hint, script_hint_invalid = _normalize_script_hint_value(script_hint)
    if script_hint_invalid:
        warnings.append("SCRIPT_HINT_INVALID")
    if not script_hint:
        script_hint = "unknown"
    if script_hint == "unknown" and text and not script_hint_invalid:
        detected = _detect_script_hint(text)
        if detected in ALLOWED_SCRIPT_HINTS:
            script_hint = detected

    try:
        confidence = float(confidence_value)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    model_warnings = [str(item).strip() for item in warnings_value] if isinstance(warnings_value, list) else []
    model_warnings = [item for item in model_warnings if item]
    warnings.extend(model_warnings)
    warnings.extend(normalized_warnings)

    if _has_meta_text(text):
        text = ""
        lines = []
        confidence = 0.0
        warnings.append("meta_text_removed")

    if _contains_script_drift(text):
        warnings.append("SCRIPT_DRIFT")

    return {
        "lines": lines,
        "text": text,
        "script_hint": script_hint,
        "confidence": confidence,
        "warnings": list(dict.fromkeys(warnings)),
        "non_latin_ratio": non_latin_ratio,
    }


def score_text_quality(text: str) -> float:
    content = (text or "").strip()
    if not content:
        return 0.0

    chars = list(content)
    total = len(chars)
    letters = sum(1 for ch in chars if ch.isalpha())
    non_latin = sum(1 for ch in chars if UNWANTED_SCRIPT_RE.match(ch))
    printable = sum(1 for ch in chars if ch.isprintable())
    ellipsis = content.count("…")

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        lines = [content]
    short_line_ratio = sum(1 for line in lines if len(line) <= 2) / len(lines)
    token_count = len(re.findall(r"[A-Za-zÀ-ÿ]+", content))

    score = (
        0.40 * min(1.0, letters / max(total * 0.6, 1))
        + 0.30 * (printable / max(total, 1))
        + 0.20 * (1.0 - short_line_ratio)
        + 0.10 * min(1.0, token_count / 20.0)
    )
    score -= min(0.20, ellipsis / max(total, 1))
    score -= min(0.50, non_latin / max(total, 1))
    return max(0.0, min(1.0, score))


def _clip_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    cx1 = max(0, min(x1, width - 1))
    cy1 = max(0, min(y1, height - 1))
    cx2 = max(cx1 + 1, min(x2, width))
    cy2 = max(cy1 + 1, min(y2, height))
    return cx1, cy1, cx2, cy2


def _detect_text_block_bbox(source: Image.Image) -> tuple[int, int, int, int]:
    width, height = source.size
    gray = source.convert("L")
    ink = gray.point(lambda p: 255 if p < 225 else 0, mode="L")
    bbox = ink.getbbox()
    if bbox is None:
        return 0, 0, width, height

    x1, y1, x2, y2 = bbox
    ink_area = max(1, (x2 - x1) * (y2 - y1))
    full_area = max(1, width * height)

    if ink_area < full_area * 0.03:
        return 0, 0, width, height

    xpad = max(12, int(width * 0.04))
    ypad = max(10, int(height * 0.02))
    return _clip_box(x1 - xpad, y1 - ypad, x2 + xpad, y2 + ypad, width, height)


def _choose_band_count(block_height: int) -> int:
    if block_height < 280:
        return 1
    bands = int(round(block_height / 260.0))
    bands = max(5, min(8, bands))
    if block_height // max(1, bands) < 90:
        bands = max(1, block_height // 90)
    return max(1, bands)


def _build_tile_boxes(source: Image.Image) -> list[tuple[int, int, int, int]]:
    width, height = source.size
    block_x1, block_y1, block_x2, block_y2 = _detect_text_block_bbox(source)
    block_w = block_x2 - block_x1
    block_h = block_y2 - block_y1

    boxes: list[tuple[int, int, int, int]] = [(block_x1, block_y1, block_x2, block_y2)]

    if min(block_w, block_h) < 120:
        return boxes

    bands = _choose_band_count(block_h)
    if bands <= 1:
        return boxes

    band_height = max(1, int(round(block_h / bands)))
    overlap_y = max(8, int(round(band_height * 0.12)))
    split_columns = block_w >= 1100
    overlap_x = max(18, int(round(block_w * 0.05)))

    for band_idx in range(bands):
        base_top = block_y1 + band_idx * band_height
        base_bottom = block_y1 + (band_idx + 1) * band_height
        band_top = block_y1 if band_idx == 0 else max(block_y1, base_top - overlap_y)
        band_bottom = block_y2 if band_idx == bands - 1 else min(block_y2, base_bottom + overlap_y)

        if not split_columns:
            boxes.append((block_x1, band_top, block_x2, band_bottom))
            continue

        mid_x = block_x1 + block_w // 2
        left_box = _clip_box(block_x1, band_top, mid_x + overlap_x, band_bottom, width, height)
        right_box = _clip_box(mid_x - overlap_x, band_top, block_x2, band_bottom, width, height)
        boxes.append(left_box)
        boxes.append(right_box)

    return boxes


def _resize_crop_for_ocr(crop: Image.Image, *, requested_scale: float) -> tuple[Image.Image, bool]:
    width, height = crop.size
    if width <= 0 or height <= 0:
        return crop, False

    target_scale = max(0.05, float(requested_scale))
    base_pixels = float(width * height)
    if base_pixels * (target_scale**2) > float(OCR_MAX_PIXELS_PER_TILE):
        target_scale = min(target_scale, (float(OCR_MAX_PIXELS_PER_TILE) / max(base_pixels, 1.0)) ** 0.5)

    long_edge = float(max(width, height))
    if long_edge * target_scale > float(OCR_MAX_LONG_EDGE):
        target_scale = min(target_scale, float(OCR_MAX_LONG_EDGE) / max(long_edge, 1.0))

    target_scale = max(0.05, target_scale)
    new_w = max(1, int(round(width * target_scale)))
    new_h = max(1, int(round(height * target_scale)))
    if new_w == width and new_h == height:
        return crop, False

    resample = Image.Resampling.LANCZOS if target_scale >= 1.0 else Image.Resampling.BICUBIC
    return crop.resize((new_w, new_h), resample), True


def _is_image_too_large_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        ("image size" in text and "exceeds limit" in text)
        or "decompression bomb" in text
        or ("too large" in text and "pixels" in text)
    )


def _downscale_tile_for_retry(tile: OCRTile) -> OCRTile | None:
    try:
        raw = decode_image_bytes(tile.image_b64)
        with Image.open(io.BytesIO(raw)) as decoded:
            image = decoded.convert("RGB")
    except Exception:
        return None

    target_w = max(1, int(round(image.width * OCR_IMAGE_RETRY_SHRINK)))
    target_h = max(1, int(round(image.height * OCR_IMAGE_RETRY_SHRINK)))
    if target_w >= image.width and target_h >= image.height:
        return None

    resized = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
    resized, _ = _resize_crop_for_ocr(resized, requested_scale=1.0)
    if resized.width >= image.width and resized.height >= image.height:
        return None

    return OCRTile(
        tile_id=tile.tile_id,
        bbox_xyxy=list(tile.bbox_xyxy),
        image_b64=encode_png_base64(resized),
        width=resized.width,
        height=resized.height,
        resized_for_limits=True,
    )


def _boxes_to_tiles(source: Image.Image, boxes: list[tuple[int, int, int, int]], *, upscale_factor: int) -> list[OCRTile]:
    unique: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for box in boxes:
        if box in seen:
            continue
        seen.add(box)
        unique.append(box)

    tiles: list[OCRTile] = []
    requested_scale = float(max(1, int(upscale_factor)))
    for idx, (x1, y1, x2, y2) in enumerate(unique):
        crop = source.crop((x1, y1, x2, y2))
        crop, resized_for_limits = _resize_crop_for_ocr(crop, requested_scale=requested_scale)
        tile_id = "tile_full" if idx == 0 else f"tile_{idx:03d}"
        tiles.append(
            OCRTile(
                tile_id=tile_id,
                bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
                image_b64=encode_png_base64(crop),
                width=crop.width,
                height=crop.height,
                resized_for_limits=resized_for_limits,
            )
        )
    return tiles


def _make_tiles_from_payload(payload: OCRExtractRequest, *, upscale_factor: int) -> tuple[list[OCRTile], str]:
    source_b64 = str(payload.cropped_image_b64 or payload.image_b64 or "").strip()
    if not source_b64:
        raise OCRAgentError("image_b64 or cropped_image_b64 is required.")

    image_bytes = decode_image_bytes(source_b64)
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            source = image.convert("RGB")
    except Exception as exc:  # pragma: no cover - decode helper already validated bytes
        raise OCRAgentError("Could not decode source image for tiled OCR.") from exc

    boxes = _build_tile_boxes(source)
    tiles = _boxes_to_tiles(source, boxes, upscale_factor=upscale_factor)
    if not tiles:
        raise OCRAgentError("No OCR tiles were generated from input image.")

    return tiles, source_b64


def _call_ocr_model(
    client: SaiaClient,
    model: str,
    tile_b64: str,
    *,
    latin_lock: bool,
) -> dict[str, object]:
    messages = build_ocr_messages(tile_b64)
    response = _chat_completion_with_optional_json_format(client=client, model=model, messages=messages)
    parsed = parse_ocr_json(str(response.get("text") or ""), latin_lock=latin_lock, strict_keys=True)

    needs_repair = _has_strict_schema_error(parsed) or _has_script_drift_warning(parsed)
    if not needs_repair:
        return parsed

    repair_messages = build_ocr_messages(
        tile_b64,
        user_prompt=(
            OCR_JSON_REPAIR_PROMPT
            + (
                " Remove Greek/Cyrillic/CJK drift from output; if uncertain, use ? or […]."
                if _has_script_drift_warning(parsed)
                else ""
            )
        ),
    )
    repair_response = _chat_completion_with_optional_json_format(client=client, model=model, messages=repair_messages)
    repaired = parse_ocr_json(str(repair_response.get("text") or ""), latin_lock=latin_lock, strict_keys=True)

    if _has_strict_schema_error(repaired):
        if not _has_strict_schema_error(parsed):
            chosen = parsed
        else:
            parsed_quality = score_text_quality(str(parsed.get("text") or ""))
            repaired_quality = score_text_quality(str(repaired.get("text") or ""))
            chosen = repaired if repaired_quality > parsed_quality else parsed
    else:
        chosen = repaired

    if _has_script_drift_warning(parsed) or _has_script_drift_warning(repaired):
        warnings = [str(item) for item in (chosen.get("warnings") or [])]
        if "SCRIPT_DRIFT" not in warnings:
            warnings.append("SCRIPT_DRIFT")
        chosen["warnings"] = warnings
    return chosen


def _chat_completion_with_optional_json_format(
    *,
    client: SaiaClient,
    model: str,
    messages: list[dict[str, object]],
) -> dict[str, object]:
    try:
        return client.chat_completion(
            model=model,
            temperature=OCR_DECODING_TEMPERATURE,
            top_p=OCR_DECODING_TOP_P,
            max_tokens=OCR_DECODING_MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=messages,
        )
    except Exception as exc:
        text = str(exc).lower()
        if "response_format" not in text and "json_object" not in text:
            raise
        return client.chat_completion(
            model=model,
            temperature=OCR_DECODING_TEMPERATURE,
            top_p=OCR_DECODING_TOP_P,
            max_tokens=OCR_DECODING_MAX_TOKENS,
            messages=messages,
        )


def build_ocr_messages(tile_b64: str, *, user_prompt: str = PALEO_OCR_USER_PROMPT) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": PALEO_OCR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tile_b64}"}},
            ],
        },
    ]


def _normalize_for_dedupe(line: str) -> str:
    value = _normalize_uncertainty_tokens(str(line or ""))
    value = value.lower().strip()
    value = re.sub(r"[\u2018\u2019\u201c\u201d`'\".,;:!(){}<>]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _uncertainty_score(line: str) -> int:
    return line.count("?") + (line.count("[…]") * 3)


def _prefer_line(candidate: str, candidate_conf: float, current: str, current_conf: float) -> bool:
    cand_uncertainty = _uncertainty_score(candidate)
    cur_uncertainty = _uncertainty_score(current)
    if cand_uncertainty != cur_uncertainty:
        return cand_uncertainty < cur_uncertainty
    if abs(candidate_conf - current_conf) > 1e-6:
        return candidate_conf > current_conf
    return len(candidate.strip()) > len(current.strip())


def _merge_tile_lines(tile_results: list[OCRTileResult]) -> tuple[list[str], int]:
    merged_lines: list[str] = []
    merged_meta: list[tuple[str, float]] = []
    deduped_count = 0

    for tile in tile_results:
        for raw_line in tile.lines:
            line = str(raw_line or "").strip()
            if not line:
                continue
            norm = _normalize_for_dedupe(line)
            if not norm:
                continue

            duplicate_idx: int | None = None
            best_ratio = 0.0
            for idx, (existing_norm, existing_conf) in enumerate(merged_meta):
                ratio = 1.0 if norm == existing_norm else SequenceMatcher(None, norm, existing_norm).ratio()
                threshold = 0.96 if min(len(norm), len(existing_norm)) < 12 else 0.92
                if ratio >= threshold and ratio > best_ratio:
                    duplicate_idx = idx
                    best_ratio = ratio

            if duplicate_idx is None:
                merged_lines.append(line)
                merged_meta.append((norm, tile.confidence))
                continue

            deduped_count += 1
            existing_norm, existing_conf = merged_meta[duplicate_idx]
            if _prefer_line(line, tile.confidence, merged_lines[duplicate_idx], existing_conf):
                merged_lines[duplicate_idx] = line
                merged_meta[duplicate_idx] = (existing_norm, max(existing_conf, tile.confidence))

    return merged_lines, deduped_count


def _merge_script_hint(tile_results: list[OCRTileResult], merged_text: str) -> str:
    weights: dict[str, float] = {}
    for tile in tile_results:
        hint = tile.script_hint if tile.script_hint in ALLOWED_SCRIPT_HINTS else "unknown"
        if hint == "unknown":
            continue
        weights[hint] = weights.get(hint, 0.0) + max(0.05, tile.confidence)

    if weights:
        return sorted(weights.items(), key=lambda item: item[1], reverse=True)[0][0]

    detected = _detect_script_hint(merged_text)
    return detected if detected in ALLOWED_SCRIPT_HINTS else "unknown"


def _merge_confidence(tile_results: list[OCRTileResult]) -> float:
    if not tile_results:
        return 0.0

    numerator = 0.0
    denominator = 0.0
    for tile in tile_results:
        weight = float(max(1, len(tile.lines)))
        numerator += float(tile.confidence) * weight
        denominator += weight

    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def _source_sha256(image_b64: str) -> str:
    return hashlib.sha256(decode_image_bytes(image_b64)).hexdigest()


def _summarize_final_changes(raw_text: str, final_text: str) -> list[str] | None:
    if raw_text == final_text:
        return None

    raw_lines = raw_text.splitlines()
    final_lines = final_text.splitlines()
    changes: list[str] = []
    for idx in range(max(len(raw_lines), len(final_lines))):
        raw_line = raw_lines[idx] if idx < len(raw_lines) else ""
        final_line = final_lines[idx] if idx < len(final_lines) else ""
        if raw_line == final_line:
            continue
        changes.append(f"line_{idx + 1}")
        if len(changes) >= 50:
            break

    return changes or None


def _ocr_tile_with_model_candidates(
    *,
    tile: OCRTile,
    saia: SaiaClient,
    selected_models: list[str],
    quality_floor: float,
    latin_lock: bool,
) -> tuple[OCRTileResult, list[OCRFallback], list[str]]:
    fallback_records: list[OCRFallback] = []
    fallbacks_used: list[str] = []

    best_lines: list[str] = []
    best_text = ""
    best_quality = -1.0
    best_flags: list[str] = []
    best_model = selected_models[0]
    best_script_hint = "unknown"
    best_confidence = 0.0
    best_warnings: list[str] = []

    for idx, model in enumerate(selected_models):
        current_tile = tile
        resize_retry_count = 0
        resized_after_limit = False
        ocr_payload: dict[str, object] | None = None

        while True:
            try:
                ocr_payload = _call_ocr_model(saia, model, current_tile.image_b64, latin_lock=latin_lock)
                break
            except Exception as exc:
                if _is_image_too_large_error(exc) and resize_retry_count < OCR_IMAGE_SIZE_RETRY_LIMIT:
                    shrunk_tile = _downscale_tile_for_retry(current_tile)
                    if shrunk_tile is not None:
                        current_tile = shrunk_tile
                        resize_retry_count += 1
                        resized_after_limit = True
                        continue

                if idx + 1 < len(selected_models):
                    reason = "MODEL_NOT_FOUND" if is_model_not_found_error(exc) else f"MODEL_ERROR:{exc}"
                    fallback_records.append(OCRFallback(model=model, reason=reason))
                    fallbacks_used.append(model)
                    break
                raise OCRAgentError(f"OCR failed on model {model}: {exc}") from exc

        if ocr_payload is None:
            continue

        lines = [str(item) for item in (ocr_payload.get("lines") or []) if str(item).strip()]
        text = str(ocr_payload.get("text") or "").strip()
        if not text and lines:
            text = "\n".join(lines)

        quality = score_text_quality(text)
        model_warnings = [str(item) for item in (ocr_payload.get("warnings") or []) if str(item).strip()]
        if tile.resized_for_limits or current_tile.resized_for_limits:
            model_warnings.append("auto_resized_for_limit")
        if resized_after_limit:
            model_warnings.append("auto_retry_downscale")
        model_warnings = list(dict.fromkeys(model_warnings))
        model_confidence = float(ocr_payload.get("confidence") or 0.0)
        model_script_hint = str(ocr_payload.get("script_hint") or "unknown").lower()
        non_latin_ratio = float(ocr_payload.get("non_latin_ratio") or 0.0)

        flags: list[str] = []
        if tile.resized_for_limits or current_tile.resized_for_limits:
            flags.append("AUTO_RESIZED_FOR_LIMIT")
        if resized_after_limit:
            flags.append("AUTO_RETRY_DOWNSCALE")
        if not text:
            flags.append("EMPTY_TEXT")
        if quality < quality_floor:
            flags.append("LOW_TEXT_QUALITY")
        if "INVALID_JSON_RESPONSE" in model_warnings or "INVALID_SCHEMA_KEYS" in model_warnings:
            flags.append("INVALID_JSON_RESPONSE")
        if "meta_text_removed" in model_warnings:
            flags.append("META_TEXT_DETECTED")
            quality *= 0.1
        if "pattern_junk_removed" in model_warnings:
            flags.append("PATTERN_JUNK_REMOVED")
            quality *= 0.6
        if non_latin_ratio > 0.0 or _contains_script_drift(text):
            flags.append("NON_LATIN_CHARS_DETECTED")
            quality *= 0.2
        if "SCRIPT_DRIFT" in model_warnings:
            flags.append("SCRIPT_DRIFT")
            quality *= 0.2
        if latin_lock and model_script_hint == "unknown" and _contains_script_drift(text):
            flags.append("SCRIPT_MISMATCH")
            quality *= 0.1
        if model_confidence < 0.35 and text:
            flags.append("LOW_MODEL_CONFIDENCE")
            quality *= 0.6
        if text:
            uncertainty_ratio = (text.count("?") + text.count("[…]")) / max(1, len(text))
            if uncertainty_ratio > 0.35:
                flags.append("HIGH_UNCERTAINTY")
                quality *= 0.7

        if quality > best_quality:
            best_lines = lines
            best_text = text
            best_quality = quality
            best_flags = list(flags)
            best_model = model
            best_script_hint = model_script_hint if model_script_hint in ALLOWED_SCRIPT_HINTS else "unknown"
            best_confidence = max(0.0, min(1.0, model_confidence))
            best_warnings = list(model_warnings)

        force_retry = any(
            flag in {"INVALID_JSON_RESPONSE", "NON_LATIN_CHARS_DETECTED", "SCRIPT_MISMATCH", "SCRIPT_DRIFT"}
            for flag in flags
        )
        if quality >= quality_floor and not force_retry:
            break

        if idx + 1 < len(selected_models):
            fallback_records.append(OCRFallback(model=model, reason=f"LOW_QUALITY:{quality:.2f}"))
            fallbacks_used.append(model)

    if {"NON_LATIN_CHARS_DETECTED", "SCRIPT_MISMATCH", "SCRIPT_DRIFT"} & set(best_flags):
        best_lines = []
        best_text = ""
        best_flags.append("REGION_TEXT_DROPPED")

    return (
        OCRTileResult(
            tile_id=tile.tile_id,
            bbox_xyxy=tile.bbox_xyxy,
            lines=best_lines,
            text=best_text,
            confidence=round(max(0.0, min(1.0, best_confidence)), 4),
            script_hint=best_script_hint,
            warnings=best_warnings,
            flags=best_flags,
            model=best_model,
        ),
        fallback_records,
        fallbacks_used,
    )


def run_ocr_extraction(
    payload: OCRExtractRequest,
    *,
    client: SaiaClient | None = None,
    upscale_factor: int = 2,
) -> OCRExtractAnyResponse:
    saia = client or SaiaClient()

    available_models = saia.list_models()
    preferred_models = resolve_model_preferences(
        explicit=payload.options.model_preference,
        prefer_model=payload.prefer_model,
    )
    candidate_models = choose_models(available_models, preferred_models)
    if not candidate_models:
        raise OCRAgentError("No image-capable model is available from SAIA /models for OCR extraction.")

    max_attempts = min(len(candidate_models), 1 + payload.options.max_fallbacks)
    selected_models = candidate_models[:max_attempts]

    tiles, source_b64 = _make_tiles_from_payload(payload, upscale_factor=upscale_factor)
    latin_lock = _latin_lock_from_hint(payload.options.language_hint)

    tile_results: list[OCRTileResult] = []
    fallback_records: list[OCRFallback] = []
    fallbacks_used: list[str] = []

    for tile in tiles:
        tile_result, tile_fallbacks, tile_fallback_models = _ocr_tile_with_model_candidates(
            tile=tile,
            saia=saia,
            selected_models=selected_models,
            quality_floor=payload.options.quality_floor,
            latin_lock=latin_lock,
        )
        tile_results.append(tile_result)
        fallback_records.extend(tile_fallbacks)
        fallbacks_used.extend(tile_fallback_models)

    merged_lines, deduped_count = _merge_tile_lines(tile_results)
    merged_text = "\n".join(merged_lines).strip()
    merged_script_hint = _merge_script_hint(tile_results, merged_text)
    merged_confidence = round(_merge_confidence(tile_results), 4)

    warnings: list[str] = []
    if deduped_count > 0:
        warnings.append("DEDUPLICATED_OVERLAPS")
    for tile in tile_results:
        for warning in tile.warnings:
            warnings.append(f"{warning}:{tile.tile_id}")
        if "LOW_TEXT_QUALITY" in tile.flags:
            warnings.append(f"LOW_TEXT_QUALITY:{tile.tile_id}")
        if "INVALID_JSON_RESPONSE" in tile.flags:
            warnings.append(f"INVALID_JSON_RESPONSE:{tile.tile_id}")
        if "NON_LATIN_CHARS_DETECTED" in tile.flags:
            warnings.append(f"NON_LATIN_CHARS_DETECTED:{tile.tile_id}")
        if "SCRIPT_DRIFT" in tile.flags:
            warnings.append(f"SCRIPT_DRIFT:{tile.tile_id}")
        if "SCRIPT_MISMATCH" in tile.flags:
            warnings.append(f"SCRIPT_MISMATCH:{tile.tile_id}")
        if "META_TEXT_DETECTED" in tile.flags:
            warnings.append(f"META_TEXT_DETECTED:{tile.tile_id}")
        if "LOW_MODEL_CONFIDENCE" in tile.flags:
            warnings.append(f"LOW_MODEL_CONFIDENCE:{tile.tile_id}")

    if not merged_text:
        warnings.append("NO_READABLE_TEXT")

    raw_ocr = OCRRawOCRPayload(
        lines=merged_lines,
        text=merged_text,
        script_hint=merged_script_hint if merged_script_hint in ALLOWED_SCRIPT_HINTS else "unknown",
        confidence=merged_confidence,
        warnings=sorted(set(warnings)),
    )

    model_counts = Counter([tile.model for tile in tile_results if tile.model])
    final_model = model_counts.most_common(1)[0][0] if model_counts else selected_models[0]

    proofreader = OcrProofreaderAgent(client=saia, model_override=final_model)
    final_text = proofreader.proofread(raw_ocr.text, raw_ocr.script_hint)
    if raw_ocr.text and not final_text:
        final_text = raw_ocr.text

    source_image_bytes = decode_image_bytes(source_b64)
    evidence_id = sha256_bytes(source_image_bytes)
    image_ref = str(payload.image_id or payload.page_id or "inline_image")
    evidence_record = OcrEvidenceRecord(
        created_at=now_iso(),
        image_sha256=evidence_id,
        image_ref=image_ref,
        model=final_model,
        decoding={
            "temperature": OCR_DECODING_TEMPERATURE,
            "top_p": OCR_DECODING_TOP_P,
            "max_tokens": OCR_DECODING_MAX_TOKENS,
        },
        prompt_version=OCR_PROMPT_VERSION,
        pipeline_version=PIPELINE_VERSION,
        script_hint=raw_ocr.script_hint,
        confidence=float(raw_ocr.confidence),
        warnings=list(raw_ocr.warnings),
        raw_ocr_text=raw_ocr.text,
        final_text=final_text or None,
        final_changes=_summarize_final_changes(raw_ocr.text, final_text),
    )
    try:
        write_ocr_evidence_jsonl(evidence_record)
    except Exception:
        pass

    if payload.mode == "simple":
        return OCRExtractSimpleResponse(
            text=final_text,
            script_hint=raw_ocr.script_hint,
            evidence_id=evidence_id,
            is_evidence=True,
            is_verified=False,
        )

    status: str = "FULL"
    if not final_text:
        status = "FAILED"
    elif raw_ocr.warnings or fallback_records or any(tile.flags for tile in tile_results):
        status = "PARTIAL"

    provenance = OCRProvenance(
        crop_sha256=_source_sha256(source_b64),
        prompt_version=PROMPT_VERSION,
        agent_version=AGENT_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    regions = [
        OCRRegionResult(
            region_id=tile.tile_id,
            text=tile.text,
            quality=round(max(0.0, min(1.0, tile.confidence)), 4),
            flags=tile.flags,
            bbox_xyxy=tile.bbox_xyxy,
            polygon=None,
        )
        for tile in tile_results
    ]

    return OCRExtractResponse(
        status=status,  # type: ignore[arg-type]
        model=final_model,
        fallbacksUsed=list(dict.fromkeys(fallbacks_used)),
        warnings=sorted(set(warnings)),
        text=final_text,
        script_hint=raw_ocr.script_hint,
        final_text=final_text,
        page_id=payload.page_id,
        image_id=payload.image_id,
        fallbacks=fallback_records,
        regions=regions,
        provenance=provenance,
        raw_ocr=raw_ocr,
        evidence_id=evidence_id,
        is_evidence=True,
        is_verified=False,
    )
