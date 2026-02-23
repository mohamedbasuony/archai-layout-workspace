from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any, Sequence

from PIL import Image

from app.agents.ocr_proofreader_agent import OcrProofreaderAgent
from app.config import settings
from app.schemas.agents_ocr import (
    DEFAULT_SAIA_OCR_MODEL_PREFS,
    SaiaOCRFallback,
    SaiaOCRLocationSuggestion,
    SaiaOCRRequest,
    SaiaOCRResponse,
)
from app.services.saia_client import SaiaClient, is_model_not_found_error

ALLOWED_SCRIPT_HINTS = {"latin", "greek", "cyrillic", "mixed", "unknown"}
ALLOWED_DETECTED_LANGUAGES = {
    "latin",
    "old_english",
    "middle_english",
    "french",
    "old_french",
    "middle_french",
    "anglo_norman",
    "occitan",
    "old_high_german",
    "middle_high_german",
    "german",
    "dutch",
    "italian",
    "spanish",
    "portuguese",
    "catalan",
    "church_slavonic",
    "greek",
    "hebrew",
    "arabic",
    "mixed",
    "unknown",
}
OLD_ENGLISH_MARKERS_RE = re.compile(r"[þðƿÞÐǷ]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
LATIN_TOKEN_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿĀ-žſƀ-ɏ]+$")
VISION_MODEL_HINTS = ("vl", "vision", "internvl", "gemma", "mistral", "omni")
INTERNVL_MODEL_HINT = "internvl"
REQUIRED_JSON_KEYS = ("lines", "text", "script_hint", "detected_language", "confidence", "warnings")
FRENCH_STRONG_ANCHORS = {
    "tres",
    "tresreuerend",
    "pere",
    "euesque",
    "eusque",
    "prince",
    "lausanne",
    "laufanne",
    "grace",
    "considerat",
    "desirs",
    "chapellain",
    "seruiteur",
    "approchant",
}
FRENCH_FUNCTION_WORDS = {"que", "et", "de", "la", "le", "des"}
FRENCH_OLD_MARKERS = {"pur", "ki", "mei", "sun", "uostre", "nostre", "cil", "cest", "dunc"}
FRENCH_MIDDLE_MARKERS = {
    "approchant",
    "considerat",
    "grace",
    "mille",
    "cens",
    "chapellain",
    "seruiteur",
    "desirs",
    "lausanne",
    "prince",
    "pere",
}
ANGLO_NORMAN_MARKERS = {"anglo", "norman", "normand", "engleterre"}

SAIA_OCR_PROMPT_VERSION = "saia_ocr_full_page_v1"
SAIA_OCR_SYSTEM_PROMPT = (
    "You are a professional paleographer producing diplomatic OCR for historical manuscripts.\n"
    "Transcribe ONLY what is visibly written in the provided full page image in natural reading order.\n"
    "NEVER invent, continue, normalize, or complete missing text from memory or external context.\n"
    "Use uncertainty markers exactly: ? for unclear characters and […] for unclear spans.\n"
    "Treat decorated initials as letters when they clearly form the first letter(s) of a line/word.\n"
    "\n"
    "Language selection rubric (detected_language)\n"
    "- detected_language must reflect the transcription language of the visible text, not modern normalization.\n"
    "- If non-Latin script is visible, choose the matching enum: greek, hebrew, arabic, or church_slavonic (for Cyrillic).\n"
    "- If Latin script:\n"
    "  - Choose latin when morphology is clearly Latin (for example many -us/-um/-ae endings; forms like domini, anno, gratia, episcopus).\n"
    "  - If text is clearly French-family (Oïl), choose among old_french, middle_french, french conservatively:\n"
    "    - Choose middle_french for general historical French spellings typical of c. 1400-1600, or whenever uncertain among French variants.\n"
    "    - Choose old_french only when earlier Old French morphology/spelling is strongly explicit.\n"
    "    - Choose french only when spelling is clearly modern French.\n"
    "  - anglo_norman MUST ONLY be used with explicit Anglo-Norman evidence (insular/Norman-specific orthography or unmistakable Anglo-Norman lexical items).\n"
    "    If unsure, do NOT use anglo_norman; use middle_french.\n"
    "- Strong Middle French anchors for language classification only (do not insert unless visible): tres, tresreuerend, pere, euesque/evesque, prince, lausanne, approchant, an de grace, considerat, desirs, chapellain, seruiteur, inuentions, poethiques.\n"
    "- Confidence guidance:\n"
    "  - confidence is a float 0..1.\n"
    "  - Long readable text with strong language cues: 0.80-0.95.\n"
    "  - Noisy or short text with plausible cues: 0.55-0.75.\n"
    "  - Truly ambiguous or mixed-language text: choose mixed and keep confidence <= 0.60.\n"
    "\n"
    "Return exactly one JSON object with keys: lines, text, script_hint, detected_language, confidence, warnings.\n"
    "text MUST equal \"\\n\".join(lines).\n"
    "detected_language MUST be one of: latin, old_english, middle_english, french, old_french, middle_french, anglo_norman, occitan, old_high_german, middle_high_german, german, dutch, italian, spanish, portuguese, catalan, church_slavonic, greek, hebrew, arabic, mixed, unknown.\n"
    "If you feel tempted to output a known passage (e.g., scripture / standard Latin boilerplate) that is not clearly visible, STOP and output lines=[] and text=\"\" instead."
)
SAIA_OCR_USER_PROMPT = (
    "Return JSON only.\n"
    "Schema rules:\n"
    "- Keys must be exactly: lines, text, script_hint, detected_language, confidence, warnings.\n"
    "- One manuscript line per entry in lines.\n"
    "- text must equal \"\\n\".join(lines).\n"
    "- Preserve original spelling, abbreviations, punctuation, capitalization, and line breaks.\n"
    "- Do not translate, normalize, modernize, or expand abbreviations.\n"
    "- Use uncertainty markers exactly: ? and […].\n"
    "- No markdown, no explanations, no extra keys.\n"
    "\n"
    "Language choice rules for detected_language:\n"
    "- Allowed values only: latin, old_english, middle_english, french, old_french, middle_french, anglo_norman, occitan, old_high_german, middle_high_german, german, dutch, italian, spanish, portuguese, catalan, church_slavonic, greek, hebrew, arabic, mixed, unknown.\n"
    "- For non-Latin script, choose the matching enum.\n"
    "- For Latin script: choose latin for clearly Latin morphology.\n"
    "- For French-family historical text, prefer middle_french when uncertain between french/old_french/middle_french.\n"
    "- Use anglo_norman only with explicit Anglo-Norman evidence; if unsure, use middle_french.\n"
    "- Middle French anchors (classification only, never insertion): tres, tresreuerend, pere, euesque/evesque, prince, lausanne, approchant, an de grace, considerat, desirs, chapellain, seruiteur, inuentions, poethiques.\n"
    "\n"
    "If location suggestions are provided, treat them as hints only for where text is likely located on the full page.\n"
    "Do not output coordinates in the response.\n"
    "If nothing readable is visible, return lines=[] and text=\"\"."
)
SAIA_OCR_JSON_REPAIR_PROMPT = (
    "Return VALID JSON only. Keys must be exactly: lines, text, script_hint, detected_language, confidence, warnings. "
    "No markdown. No extra keys."
)

DEFAULT_MODEL_PREFS_CSV = ",".join(DEFAULT_SAIA_OCR_MODEL_PREFS)
DEFAULT_OCR_TEMPERATURE = 0.0
DEFAULT_OCR_MAX_TOKENS = 4096
SAIA_OCR_MAX_PIXELS = max(
    1,
    int(settings.ocr_max_pixels_per_tile or os.getenv("OCR_MAX_PIXELS_PER_TILE", "160000000") or "160000000"),
)
SAIA_OCR_MAX_LONG_EDGE = max(
    1024,
    int(settings.ocr_max_long_edge or os.getenv("OCR_MAX_LONG_EDGE", "12000") or "12000"),
)
SAIA_OCR_IMAGE_SIZE_RETRY_LIMIT = max(
    0,
    int(settings.ocr_image_size_retry_limit or os.getenv("OCR_IMAGE_SIZE_RETRY_LIMIT", "2") or "2"),
)
SAIA_OCR_IMAGE_RETRY_SHRINK = float(
    settings.ocr_image_retry_shrink or os.getenv("OCR_IMAGE_RETRY_SHRINK", "0.82") or "0.82"
)
if SAIA_OCR_IMAGE_RETRY_SHRINK <= 0.0 or SAIA_OCR_IMAGE_RETRY_SHRINK >= 1.0:
    SAIA_OCR_IMAGE_RETRY_SHRINK = 0.82


class SaiaOCRAgentError(RuntimeError):
    """Raised when SAIA OCR extraction fails."""


def _extract_base64_payload(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise SaiaOCRAgentError("image_b64 is required.")
    if text.startswith("data:"):
        marker = text.find(",")
        if marker == -1:
            raise SaiaOCRAgentError("image_b64 data URL is malformed.")
        text = text[marker + 1 :]
    try:
        base64.b64decode(text, validate=True)
    except Exception as exc:  # pragma: no cover - validation guard
        raise SaiaOCRAgentError("image_b64 is not valid base64.") from exc
    return text


def _open_rgb_image_from_b64(image_b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(image_b64, validate=False)
    except Exception as exc:
        raise SaiaOCRAgentError("image_b64 is not valid base64.") from exc

    previous_max_pixels = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(io.BytesIO(raw)) as decoded:
            image = decoded.convert("RGB")
    except Exception as exc:
        raise SaiaOCRAgentError("Unable to decode image payload for OCR.") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_max_pixels
    return image


def _encode_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _resize_for_limits(image: Image.Image) -> tuple[Image.Image, bool]:
    width, height = image.size
    if width <= 0 or height <= 0:
        return image, False

    pixels = float(width * height)
    long_edge = float(max(width, height))
    scale = 1.0

    if pixels > float(SAIA_OCR_MAX_PIXELS):
        scale = min(scale, (float(SAIA_OCR_MAX_PIXELS) / max(pixels, 1.0)) ** 0.5)
    if long_edge > float(SAIA_OCR_MAX_LONG_EDGE):
        scale = min(scale, float(SAIA_OCR_MAX_LONG_EDGE) / max(long_edge, 1.0))

    if scale >= 0.999:
        return image, False

    target_w = max(1, int(round(width * scale)))
    target_h = max(1, int(round(height * scale)))
    resized = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
    return resized, True


def _prepare_image_for_ocr(image_b64: str) -> tuple[str, list[str]]:
    image = _open_rgb_image_from_b64(image_b64)
    original_w, original_h = image.size
    resized, changed = _resize_for_limits(image)
    if not changed:
        return image_b64, []

    warning = (
        "AUTO_RESIZED_FOR_LIMIT:"
        f"{original_w}x{original_h}->{resized.width}x{resized.height}"
    )
    return _encode_png_base64(resized), [warning]


def _downscale_image_b64_for_retry(image_b64: str) -> str | None:
    image = _open_rgb_image_from_b64(image_b64)
    target_w = max(1, int(round(image.width * SAIA_OCR_IMAGE_RETRY_SHRINK)))
    target_h = max(1, int(round(image.height * SAIA_OCR_IMAGE_RETRY_SHRINK)))
    if target_w >= image.width and target_h >= image.height:
        return None

    resized = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
    resized, _ = _resize_for_limits(resized)
    if resized.width >= image.width and resized.height >= image.height:
        return None
    return _encode_png_base64(resized)


def _is_image_too_large_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        ("image size" in text and "exceeds limit" in text)
        or "decompression bomb" in text
        or ("too large" in text and "pixels" in text)
    )


def _compact_error_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text[:180] if len(text) > 180 else text


def _parse_csv_or_json_list(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(",") if item.strip()]


def _filter_internvl_models(models: Sequence[str]) -> list[str]:
    return [model for model in models if INTERNVL_MODEL_HINT in str(model).lower()]


def _strip_fences(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    return text


def _extract_json_object(raw: str) -> str:
    text = _strip_fences(raw)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def _detect_script_hint(text: str) -> str:
    value = str(text or "")
    has_greek = bool(re.search(r"[\u0370-\u03FF]", value))
    has_cyrillic = bool(re.search(r"[\u0400-\u04FF]", value))
    has_latin = bool(re.search(r"[A-Za-zÀ-ÿ]", value))
    if has_latin and not has_greek and not has_cyrillic:
        return "latin"
    if has_greek and not has_latin and not has_cyrillic:
        return "greek"
    if has_cyrillic and not has_latin and not has_greek:
        return "cyrillic"
    if sum([has_latin, has_greek, has_cyrillic]) >= 2:
        return "mixed"
    return "unknown"


def _normalize_detected_language(value: str | None) -> str:
    language = str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "la": "latin",
        "fr": "french",
        "fro": "old_french",
        "frm": "middle_french",
        "enm": "middle_english",
        "ang": "old_english",
        "de": "german",
        "it": "italian",
        "es": "spanish",
        "pt": "portuguese",
        "nl": "dutch",
        "ca": "catalan",
        "oc": "occitan",
        "he": "hebrew",
        "ar": "arabic",
        "el": "greek",
    }
    language = aliases.get(language, language)
    if language in ALLOWED_DETECTED_LANGUAGES:
        return language
    return "unknown"


def _normalize_language_token(token: str) -> str:
    return str(token or "").lower().replace("ſ", "s").replace("v", "u").replace("j", "i")


def _resolve_french_family_language(text: str) -> str | None:
    value = str(text or "")
    if not value.strip():
        return None

    tokens: list[str] = []
    for raw_token in re.split(r"\s+", value.lower()):
        if not raw_token:
            continue
        cleaned = "".join(ch for ch in raw_token if ch.isalpha())
        if not cleaned:
            continue
        if LATIN_TOKEN_RE.match(cleaned) is None:
            continue
        tokens.append(_normalize_language_token(cleaned))

    if not tokens:
        return None

    strong_hits = 0
    function_hits = 0
    old_hits = 0
    middle_hits = 0
    anglo_hits = 0
    for token in tokens:
        if any(token == anchor or token.startswith(anchor) for anchor in FRENCH_STRONG_ANCHORS):
            strong_hits += 1
        if token in FRENCH_FUNCTION_WORDS:
            function_hits += 1
        if token in FRENCH_OLD_MARKERS:
            old_hits += 1
        if token in FRENCH_MIDDLE_MARKERS:
            middle_hits += 1
        if token in ANGLO_NORMAN_MARKERS:
            anglo_hits += 1

    if not (strong_hits >= 2 or (strong_hits >= 1 and function_hits >= 2)):
        return None
    if anglo_hits >= 2:
        return "anglo_norman"
    if old_hits >= 2 and middle_hits == 0:
        return "old_french"
    return "middle_french"


def _fallback_detected_language(script_hint: str, text: str) -> str:
    value = str(text or "")
    script = str(script_hint or "unknown").lower()

    if not value.strip():
        return "unknown"

    if OLD_ENGLISH_MARKERS_RE.search(value):
        return "old_english"
    if ARABIC_RE.search(value):
        return "arabic"
    if HEBREW_RE.search(value):
        return "hebrew"
    french_family = _resolve_french_family_language(value)
    if french_family is not None:
        return french_family
    if script == "greek":
        return "greek"
    if script == "cyrillic":
        return "church_slavonic"
    if script == "mixed":
        return "mixed"
    return "latin"


def _normalize_lines(lines: Sequence[Any], text: str) -> list[str]:
    normalized = [str(item).strip() for item in lines if str(item).strip()]
    if not normalized and text:
        normalized = [line.strip() for line in text.splitlines() if line.strip()]
    return normalized


def _parse_ocr_payload(raw: str) -> dict[str, Any] | None:
    candidate = _extract_json_object(raw)
    try:
        payload = json.loads(candidate)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if set(payload.keys()) != set(REQUIRED_JSON_KEYS):
        return None

    lines = payload.get("lines")
    text = payload.get("text")
    script_hint = str(payload.get("script_hint") or "unknown").strip().lower()
    detected_language = _normalize_detected_language(str(payload.get("detected_language") or "unknown"))
    confidence = payload.get("confidence", 0.0)
    warnings = payload.get("warnings", [])

    if not isinstance(lines, list):
        return None
    if not isinstance(text, str):
        return None
    if not isinstance(warnings, list):
        return None

    normalized_lines = _normalize_lines(lines, text)
    normalized_text = "\n".join(normalized_lines).strip()
    normalized_warnings: list[str] = []
    if str(text).strip() != normalized_text:
        normalized_warnings.append("TEXT_JOIN_FIXED")
    if script_hint not in ALLOWED_SCRIPT_HINTS:
        script_hint = _detect_script_hint(normalized_text)
    if script_hint not in ALLOWED_SCRIPT_HINTS:
        script_hint = "unknown"
    if detected_language == "unknown" and normalized_text:
        script_guess = _detect_script_hint(normalized_text)
        if script_guess == "greek":
            detected_language = "greek"
    if detected_language == "unknown":
        detected_language = _fallback_detected_language(script_hint, normalized_text)
    try:
        confidence_value = max(0.0, min(1.0, float(confidence)))
    except Exception:
        confidence_value = 0.0
    normalized_warnings.extend([str(item).strip() for item in warnings if str(item).strip()])

    return {
        "lines": normalized_lines,
        "text": normalized_text,
        "script_hint": script_hint,
        "detected_language": detected_language,
        "confidence": confidence_value,
        "warnings": list(dict.fromkeys(normalized_warnings)),
    }


def _format_location_suggestions(suggestions: Sequence[SaiaOCRLocationSuggestion]) -> str:
    if not suggestions:
        return ""
    lines = ["Location suggestions (x,y,w,h in full-page coordinates; hints only):", ""]
    for idx, suggestion in enumerate(suggestions, start=1):
        bbox = [float(v) for v in (suggestion.bbox_xywh or [])]
        if len(bbox) != 4:
            continue
        region_id = str(suggestion.region_id or f"r{idx}")
        category = str(suggestion.category or "text")
        x, y, w, h = bbox
        lines.append(f"{region_id} [{category}]: ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")
    if len(lines) <= 2:
        return ""
    return "\n".join(lines)


def build_saia_ocr_messages(
    image_b64: str,
    *,
    repair_json: bool = False,
    location_suggestions: Sequence[SaiaOCRLocationSuggestion] | None = None,
) -> list[dict[str, Any]]:
    text_prompt = SAIA_OCR_USER_PROMPT
    location_block = _format_location_suggestions(location_suggestions or [])
    if location_block:
        text_prompt = f"{text_prompt}\n\n{location_block}"
    if repair_json:
        text_prompt = f"{text_prompt}\n\n{SAIA_OCR_JSON_REPAIR_PROMPT}"
    return [
        {"role": "system", "content": SAIA_OCR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        },
    ]


def _line_dup_ratio(lines: list[str]) -> float:
    normalized = [re.sub(r"\s+", " ", line.lower()).strip() for line in lines if line.strip()]
    if len(normalized) < 2:
        return 0.0
    return 1.0 - (len(set(normalized)) / len(normalized))


def _garbage_ratio(text: str) -> float:
    value = str(text or "")
    if not value:
        return 1.0
    repeated_runs = sum(len(match.group(0)) - 4 for match in re.finditer(r"(.)\1{4,}", value))
    non_printable = sum(1 for ch in value if not ch.isprintable())
    weird = repeated_runs + non_printable
    return max(0.0, min(1.0, weird / max(1, len(value))))


def _script_drift_ratio(text: str, script_hint: str) -> float:
    if script_hint != "latin":
        return 0.0
    letters = re.findall(r"[A-Za-zÀ-ÿ\u0370-\u03FF\u0400-\u04FF]", text or "")
    if not letters:
        return 0.0
    cross = re.findall(r"[\u0370-\u03FF\u0400-\u04FF]", text or "")
    return len(cross) / len(letters)


def _looks_hallucinated_passage(text: str, lines: list[str]) -> bool:
    if len(text) < 900 or len(lines) < 12:
        return False
    duplicate_ratio = _line_dup_ratio(lines)
    uncertainty_ratio = (text.count("?") + text.count("[…]")) / max(1, len(text))
    words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", text.lower())
    if len(words) < 140:
        return False

    chunk_counts: dict[str, int] = {}
    for idx in range(len(words) - 5):
        chunk = " ".join(words[idx : idx + 6])
        chunk_counts[chunk] = chunk_counts.get(chunk, 0) + 1
    max_repeat = max(chunk_counts.values()) if chunk_counts else 0

    return duplicate_ratio >= 0.28 and max_repeat >= 4 and uncertainty_ratio <= 0.01


def _is_low_text_quality(text: str, lines: list[str], script_hint: str) -> bool:
    if not text.strip():
        return True
    if _garbage_ratio(text) >= 0.22:
        return True
    if _script_drift_ratio(text, script_hint) >= 0.20:
        return True
    if _line_dup_ratio(lines) >= 0.40 and len(lines) >= 8:
        return True
    return False


class SaiaOCRAgent:
    def __init__(
        self,
        *,
        client: SaiaClient | None = None,
        model_prefs: Sequence[str] | None = None,
    ) -> None:
        self.client = client or SaiaClient()
        self.model_prefs = (
            _filter_internvl_models(model_prefs) if model_prefs is not None else self._resolve_model_prefs()
        )
        self.temperature = self._resolve_temperature()
        self.max_tokens = self._resolve_max_tokens()

    @staticmethod
    def _resolve_model_prefs() -> list[str]:
        prefs = _filter_internvl_models(
            _parse_csv_or_json_list(
            settings.saia_ocr_models
            or os.getenv("SAIA_OCR_MODELS", "")
            or settings.saia_ocr_model_prefs
            or os.getenv("SAIA_OCR_MODEL_PREFS", "")
            or settings.saia_ocr_model_preferences
            or DEFAULT_MODEL_PREFS_CSV
            )
        )
        return prefs or _filter_internvl_models(DEFAULT_SAIA_OCR_MODEL_PREFS)

    @staticmethod
    def _resolve_temperature() -> float:
        raw = str(
            settings.saia_ocr_temperature
            if settings.saia_ocr_temperature is not None
            else os.getenv("SAIA_OCR_TEMPERATURE", str(DEFAULT_OCR_TEMPERATURE))
        )
        try:
            return float(raw)
        except Exception:
            return DEFAULT_OCR_TEMPERATURE

    @staticmethod
    def _resolve_max_tokens() -> int:
        raw = str(
            settings.saia_ocr_max_tokens
            if settings.saia_ocr_max_tokens is not None
            else os.getenv("SAIA_OCR_MAX_TOKENS", str(DEFAULT_OCR_MAX_TOKENS))
        )
        try:
            return max(256, int(raw))
        except Exception:
            return DEFAULT_OCR_MAX_TOKENS

    def _select_candidate_models(self, available_models: Sequence[str]) -> list[str]:
        by_lower = {model.lower(): model for model in available_models}
        ordered: list[str] = []
        seen: set[str] = set()

        for preferred in self.model_prefs:
            resolved = by_lower.get(preferred.lower())
            if not resolved:
                continue
            key = resolved.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(resolved)

        if ordered:
            return ordered

        for model in available_models:
            key = model.lower()
            if key in seen:
                continue
            if INTERNVL_MODEL_HINT in key:
                seen.add(key)
                ordered.append(model)

        return ordered

    def _request_json_from_model(
        self,
        *,
        model: str,
        image_b64: str,
        location_suggestions: Sequence[SaiaOCRLocationSuggestion],
    ) -> dict[str, Any]:
        messages = build_saia_ocr_messages(
            image_b64,
            repair_json=False,
            location_suggestions=location_suggestions,
        )
        response = self._chat_completion_with_optional_json_object(model=model, messages=messages)
        parsed = _parse_ocr_payload(str(response.get("text") or ""))
        if parsed is not None:
            return parsed

        repair_messages = build_saia_ocr_messages(
            image_b64,
            repair_json=True,
            location_suggestions=location_suggestions,
        )
        repair_response = self._chat_completion_with_optional_json_object(model=model, messages=repair_messages)
        repaired = _parse_ocr_payload(str(repair_response.get("text") or ""))
        if repaired is None:
            return {
                "lines": [],
                "text": "",
                "script_hint": "unknown",
                "detected_language": "unknown",
                "confidence": 0.0,
                "warnings": ["INVALID_OCR_JSON"],
            }
        repaired["warnings"] = list(dict.fromkeys([*repaired["warnings"], "JSON_REPAIR_RETRY"]))
        return repaired

    def _chat_completion_with_optional_json_object(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            return self.client.chat_completion(
                model=model,
                temperature=self.temperature,
                top_p=1.0,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                messages=messages,
            )
        except Exception as exc:
            text = str(exc).lower()
            if "response_format" not in text and "json_object" not in text:
                raise
            return self.client.chat_completion(
                model=model,
                temperature=self.temperature,
                top_p=1.0,
                max_tokens=self.max_tokens,
                messages=messages,
            )

    def _maybe_proofread(
        self,
        *,
        model: str,
        script_hint: str,
        detected_language: str,
        lines: list[str],
        text: str,
        enabled: bool,
    ) -> tuple[list[str], str, list[str]]:
        if not enabled:
            return lines, text, []
        if len(lines) < 3:
            return lines, text, []
        if _garbage_ratio(text) >= 0.35:
            return lines, text, ["PROOFREAD_SKIPPED_LOW_QUALITY"]

        proofreader = OcrProofreaderAgent(client=self.client, model_override=model)
        try:
            corrected = proofreader.proofread(text, script_hint, detected_language=detected_language)
        except Exception:
            return lines, text, ["PROOFREAD_FAILED"]

        corrected = corrected.strip()
        if not corrected:
            return lines, text, ["PROOFREAD_EMPTY_RESULT"]
        corrected_lines = [line.strip() for line in corrected.splitlines() if line.strip()]
        corrected_text = "\n".join(corrected_lines).strip()
        if not corrected_text:
            return lines, text, ["PROOFREAD_EMPTY_RESULT"]
        return corrected_lines, corrected_text, []

    def extract(self, payload: SaiaOCRRequest) -> SaiaOCRResponse:
        image_b64 = _extract_base64_payload(payload.image_b64 or "")
        image_b64, preprocess_warnings = _prepare_image_for_ocr(image_b64)
        available_models = self.client.list_models()
        candidate_models = self._select_candidate_models(available_models)
        if not candidate_models:
            raise SaiaOCRAgentError("No InternVL model is available on SAIA /models.")

        fallbacks: list[SaiaOCRFallback] = []
        best_partial: SaiaOCRResponse | None = None

        for index, model in enumerate(candidate_models):
            current_image_b64 = image_b64
            size_retry_count = 0
            resized_after_limit = False
            parsed: dict[str, Any] | None = None
            try:
                while True:
                    try:
                        parsed = self._request_json_from_model(
                            model=model,
                            image_b64=current_image_b64,
                            location_suggestions=payload.location_suggestions,
                        )
                        break
                    except Exception as exc:
                        if _is_image_too_large_error(exc) and size_retry_count < SAIA_OCR_IMAGE_SIZE_RETRY_LIMIT:
                            shrunk = _downscale_image_b64_for_retry(current_image_b64)
                            if shrunk:
                                current_image_b64 = shrunk
                                size_retry_count += 1
                                resized_after_limit = True
                                continue
                        raise
            except Exception as exc:
                err = "MODEL_NOT_FOUND" if is_model_not_found_error(exc) else str(exc)
                fallbacks.append(SaiaOCRFallback(model=model, error=err))
                continue
            if parsed is None:
                fallbacks.append(SaiaOCRFallback(model=model, error="EMPTY_MODEL_RESPONSE"))
                continue

            lines = [str(line) for line in parsed["lines"] if str(line).strip()]
            text = str(parsed["text"] or "").strip()
            script_hint = str(parsed["script_hint"] or "unknown").lower()
            detected_language = _normalize_detected_language(str(parsed.get("detected_language") or "unknown"))
            if script_hint not in ALLOWED_SCRIPT_HINTS:
                script_hint = _detect_script_hint(text)
            if script_hint not in ALLOWED_SCRIPT_HINTS:
                script_hint = "unknown"
            if detected_language == "unknown":
                detected_language = _fallback_detected_language(script_hint, text)
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
            warnings = [*preprocess_warnings, *[str(item) for item in parsed.get("warnings", []) if str(item).strip()]]
            if resized_after_limit:
                warnings.append("AUTO_RESIZED_FOR_LIMIT_RETRY")

            script_drift = _script_drift_ratio(text, script_hint)
            if script_drift >= 0.08:
                warnings.append("SCRIPT_DRIFT")

            hallucinated = _looks_hallucinated_passage(text, lines)
            if hallucinated:
                warnings.append("HALLUCINATION_SUSPECTED")

            low_quality = _is_low_text_quality(text, lines, script_hint)
            if low_quality:
                warnings.append("LOW_TEXT_QUALITY")

            if (hallucinated or low_quality) and index + 1 < len(candidate_models):
                fallbacks.append(SaiaOCRFallback(model=model, error="LOW_TEXT_QUALITY"))
                partial_status = "PARTIAL" if text else "FAIL"
                partial_candidate = SaiaOCRResponse(
                    status=partial_status,
                    model_used=model,
                    fallbacks=list(fallbacks),
                    fallbacks_used=[item.model for item in fallbacks],
                    warnings=list(dict.fromkeys(warnings)),
                    lines=lines,
                    text=text,
                    script_hint=script_hint,
                    detected_language=detected_language,
                    confidence=confidence,
                    raw_json=parsed,
                )
                if best_partial is None or len(partial_candidate.text) > len(best_partial.text):
                    best_partial = partial_candidate
                continue

            proof_lines, proof_text, proof_warnings = self._maybe_proofread(
                model=model,
                script_hint=script_hint,
                detected_language=detected_language,
                lines=lines,
                text=text,
                enabled=bool(payload.apply_proofread),
            )
            warnings.extend(proof_warnings)
            final_lines = proof_lines if proof_text else lines
            final_text = proof_text if proof_text else text

            status = "FULL"
            if not final_text:
                status = "FAIL"
            elif warnings or fallbacks:
                status = "PARTIAL"

            return SaiaOCRResponse(
                status=status,
                model_used=model,
                fallbacks=fallbacks,
                fallbacks_used=[item.model for item in fallbacks],
                warnings=list(dict.fromkeys(warnings)),
                lines=final_lines,
                text=final_text,
                script_hint=script_hint,
                detected_language=detected_language,
                confidence=confidence,
                raw_json=parsed,
            )

        if best_partial is not None:
            return best_partial

        warning_details = [*preprocess_warnings]
        for fallback in fallbacks:
            warning_details.append(f"MODEL_ERROR:{fallback.model}:{_compact_error_text(fallback.error)}")

        return SaiaOCRResponse(
            status="FAIL",
            model_used="",
            fallbacks=fallbacks,
            fallbacks_used=[item.model for item in fallbacks],
            warnings=list(dict.fromkeys(["OCR_FAILED_ALL_MODELS", *warning_details])),
            lines=[],
            text="",
            script_hint="unknown",
            detected_language="unknown",
            confidence=0.0,
            raw_json=None,
        )
