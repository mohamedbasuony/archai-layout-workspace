from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageFilter

from app.agents.ocr_proofreader_agent import OcrProofreaderAgent, check_proofread_delta
from app.config import settings
from app.services.lexicon_trust import lexical_trust_adjustment
from app.services.multiview import generate_variants, pick_retry_variant
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
    "Transcribe ONLY what is visibly written in the provided image in natural reading order.\n"
    "NEVER invent, continue, normalize, or complete missing text from memory or external context.\n"
    "Use uncertainty markers exactly: ? for unclear characters and […] for unclear spans.\n"
    "Do not guess words.\n"
    "If a word is not clearly legible, you MUST NOT output a plausible substitute.\n"
    "For partially legible words: keep visible letters and replace unclear characters with ? (example: tresr?uerend).\n"
    "For wholly unclear words/spans: output […], not a made-up token.\n"
    "Never replace unclear text with a different familiar word.\n"
    "Treat decorated initials as letters when they clearly form the first letter(s) of a line/word.\n"
    "\n"
    "CRITICAL: GOTHIC BOOKHAND / TEXTURA READING GUIDE\n"
    "Medieval manuscripts are typically written in Gothic textura or bookhand. You MUST apply these rules:\n"
    "\n"
    "MINIM CONFUSION — the most common OCR error:\n"
    "  Gothic script writes i, u, n, m as sequences of identical vertical strokes (minims).\n"
    "  - Two minims = u or n (context decides).\n"
    "  - Three minims = m, in, ni, ui, iu.\n"
    "  - Do NOT guess: if a minim sequence is ambiguous, transcribe the most likely reading based on surrounding letters.\n"
    "  - Common minim-heavy words in Old French/Latin: uilain (vilain), honeur (honneur), enuie, dominus, anima.\n"
    "\n"
    "LONG-S (ſ) vs f:\n"
    "  - Long-s (ſ) looks almost identical to f but has NO crossbar or only a half-crossbar on the left.\n"
    "  - f has a FULL crossbar through the stem.\n"
    "  - Common misread: 'furent' read as 'surent', or 'si' read as 'fi'. Check the crossbar carefully.\n"
    "  - In diplomatic transcription, you may output either 's' or 'ſ' for long-s.\n"
    "\n"
    "TIRONIAN ET (⁊ / z-shaped):\n"
    "  - In many medieval manuscripts, 'et' (and) is written as a z-shaped or ⁊-shaped symbol.\n"
    "  - Transcribe it as 'z' or '⁊' or 'et' — but be consistent. The symbol is NOT the letter z.\n"
    "  - If you see a standalone z-like character between words, it almost certainly means 'et'.\n"
    "\n"
    "ABBREVIATION MARKS:\n"
    "  - A horizontal bar (macron/tilde) over a vowel usually indicates a missing nasal: ā = an/am, ō = on/om, ū = un/um.\n"
    "  - A bar over a consonant often indicates a missing following letter: q̄ = que, p̄ = per/par/pre.\n"
    "  - Superscript letters indicate omitted sequences.\n"
    "  - DO NOT expand abbreviations — keep the abbreviated form as written.\n"
    "  - If an abbreviation mark is visible, note it; do not silently drop it.\n"
    "\n"
    "LETTER CONFUSIONS TO WATCH:\n"
    "  - c/t: in Gothic, c and t look very similar. Context helps: 'ct' clusters are common in Latin.\n"
    "  - r: Gothic r has two forms — normal r and round-r (shaped like 2) used after o, p, b, d.\n"
    "  - d: Gothic d can be upright (like modern d) or uncial (rounded back, looks like ð without the stroke).\n"
    "  - v/b: Gothic v and b can be confused. v is usually used word-initially for 'u' sounds.\n"
    "  - a/o: in some hands these are very similar.\n"
    "\n"
    "DECORATED / RUBRICATED INITIALS:\n"
    "  - Large coloured letters (red, blue, gold) at line starts are decorated initials.\n"
    "  - They are real letters — transcribe them as the letter they represent.\n"
    "  - A large R, C, L, Q, etc. at the start of a stanza is the first letter of the first word.\n"
    "\n"
    "LAYOUT:\n"
    "  - Many manuscript pages have TWO COLUMNS. Read each column top-to-bottom, left column first.\n"
    "  - Marginal annotations may appear outside the main text block.\n"
    "  - Verse texts have one line of poetry per manuscript line.\n"
    "\n"
    "NUMERAL AND SPACING RULES\n"
    "- Do NOT output Arabic numerals (0-9) unless clearly written as numerals in the manuscript.\n"
    "- Do NOT insert spaces inside a single word.\n"
    "- If two letters appear close but you are unsure whether they belong to the same word, keep them together.\n"
    "\n"
    "Language selection rubric (detected_language)\n"
    "- detected_language must reflect the language of the visible text.\n"
    "- Choose latin for clearly Latin morphology (many -us/-um/-ae endings).\n"
    "- For French-family text, prefer old_french for pre-1300 texts, middle_french for c.1300-1500, french for modern.\n"
    "- anglo_norman only with explicit Anglo-Norman evidence; if unsure, use old_french or middle_french.\n"
    "- Confidence guidance:\n"
    "  - Long readable text with strong language cues: 0.80-0.95.\n"
    "  - Noisy or short text: 0.55-0.75.\n"
    "  - Truly ambiguous or mixed: choose mixed and keep confidence <= 0.60.\n"
    "\n"
    "Return exactly one JSON object with keys: lines, text, script_hint, detected_language, confidence, warnings.\n"
    "text MUST equal \"\\n\".join(lines).\n"
    "detected_language MUST be one of: latin, old_english, middle_english, french, old_french, middle_french, anglo_norman, occitan, old_high_german, middle_high_german, german, dutch, italian, spanish, portuguese, catalan, church_slavonic, greek, hebrew, arabic, mixed, unknown.\n"
    "If you feel tempted to output a known passage that is not clearly visible, STOP and output lines=[] and text=\"\" instead.\n"
    "If you are not sure a word is correct, you MUST output ? or […]; do not output a plausible-looking word without visible support."
)
SAIA_OCR_USER_PROMPT = (
    "Return JSON only.\n"
    "Schema: keys must be exactly: lines, text, script_hint, detected_language, confidence, warnings.\n"
    "- One manuscript line per entry in lines. Each entry MUST NOT contain '\\n'.\n"
    "- text must equal \"\\n\".join(lines).\n"
    "- Preserve original spelling, abbreviations, punctuation, capitalization, and line breaks.\n"
    "- Do not translate, normalize, modernize, or expand abbreviations.\n"
    "- Use uncertainty markers: ? for unclear characters, […] for unclear spans.\n"
    "- No markdown, no explanations, no extra keys.\n"
    "\n"
    "READING REMINDERS FOR GOTHIC SCRIPT:\n"
    "- This manuscript is likely in Gothic textura/bookhand. Read each letter carefully.\n"
    "- Watch for MINIM confusion: sequences of identical vertical strokes encode i, u, n, m.\n"
    "- The z-shaped symbol between words = Tironian et (meaning 'and'). Transcribe as z or et.\n"
    "- Long-s (ſ) has NO crossbar; f HAS a crossbar. Do not confuse them.\n"
    "- Bars/tildes over letters = abbreviation marks (nasal or missing letters). Keep them.\n"
    "- Two-column layouts: read left column completely first, then right column.\n"
    "- Decorated/colored initials are real letters — include them in the transcription.\n"
    "\n"
    "Language: use old_french for pre-1300 French, middle_french for c.1300-1500.\n"
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


# ---------------------------------------------------------------------------
# Image enhancement for manuscript OCR  (language-agnostic, single-pass)
# ---------------------------------------------------------------------------
# These transforms improve the signal the vision model receives without
# adding extra SAIA API calls.  They target three common manuscript issues:
#   1. Low / uneven contrast (faded ink, stained parchment)  → CLAHE on L*
#   2. Soft focus / blurry strokes                           → unsharp-mask
#   3. Sensor / compression noise                            → bilateral filter
# All operations preserve the original RGB colour space so the model still
# sees colour cues (rubricated initials, ink colour, marginalia).
# ---------------------------------------------------------------------------

_ENHANCE_CLAHE_CLIP = 2.0       # clip limit for CLAHE (moderate — don't blow out minims)
_ENHANCE_CLAHE_GRID = 8         # tile grid for adaptive histogram
_ENHANCE_BILATERAL_D = 5        # bilateral filter diameter (lighter touch)
_ENHANCE_BILATERAL_SC = 40.0    # sigma-colour  (preserve edges, lighter)
_ENHANCE_BILATERAL_SS = 40.0    # sigma-space   (spatial smoothing, lighter)
_ENHANCE_SHARPEN_RADIUS = 1     # unsharp-mask radius in px (gentle — don't warp minims)
_ENHANCE_SHARPEN_PERCENT = 80   # unsharp-mask strength (%) (reduced from 120)
_ENHANCE_SHARPEN_THRESH = 4     # unsharp-mask threshold (raised to avoid sharpening noise)
_ENHANCE_ENABLED = True         # master switch (can be overridden via env)


def _enhance_manuscript_image(image: Image.Image) -> Image.Image:
    """Enhance a manuscript image for better vision-model OCR.

    Steps:
      1. Convert RGB → L*a*b*, apply CLAHE to L* channel, convert back.
      2. Bilateral-filter the result (denoise while keeping stroke edges).
      3. Unsharp-mask sharpen to crisp up character strokes.

    Returns an enhanced PIL RGB image of the same dimensions.
    """
    if not _ENHANCE_ENABLED:
        return image

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        # Fallback: PIL-only sharpening if OpenCV is unavailable
        return image.filter(
            ImageFilter.UnsharpMask(
                radius=_ENHANCE_SHARPEN_RADIUS,
                percent=_ENHANCE_SHARPEN_PERCENT,
                threshold=_ENHANCE_SHARPEN_THRESH,
            )
        )

    arr = np.array(image.convert("RGB"), dtype=np.uint8)

    # --- 1. CLAHE on the L* channel of CIE-LAB ---
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=_ENHANCE_CLAHE_CLIP,
        tileGridSize=(_ENHANCE_CLAHE_GRID, _ENHANCE_CLAHE_GRID),
    )
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # --- 2. Bilateral filter (edge-preserving denoise) ---
    arr = cv2.bilateralFilter(
        arr,
        d=_ENHANCE_BILATERAL_D,
        sigmaColor=_ENHANCE_BILATERAL_SC,
        sigmaSpace=_ENHANCE_BILATERAL_SS,
    )

    enhanced = Image.fromarray(arr, mode="RGB")

    # --- 3. Unsharp-mask sharpening (PIL, works on RGB directly) ---
    enhanced = enhanced.filter(
        ImageFilter.UnsharpMask(
            radius=_ENHANCE_SHARPEN_RADIUS,
            percent=_ENHANCE_SHARPEN_PERCENT,
            threshold=_ENHANCE_SHARPEN_THRESH,
        )
    )

    return enhanced


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
    warnings: list[str] = []

    # --- Enhancement: improve contrast / sharpness for the vision model ---
    image = _enhance_manuscript_image(image)

    resized, changed = _resize_for_limits(image)
    if changed:
        warnings.append(
            "AUTO_RESIZED_FOR_LIMIT:"
            f"{original_w}x{original_h}->{resized.width}x{resized.height}"
        )
        image = resized

    return _encode_png_base64(image), warnings


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
    normalized: list[str] = []
    for item in lines:
        raw = str(item)
        if "\n" in raw or "\r" in raw:
            normalized.extend([fragment.strip() for fragment in raw.splitlines() if fragment.strip()])
            continue
        stripped = raw.strip()
        if stripped:
            normalized.append(stripped)
    if not normalized and text:
        normalized = [line.strip() for line in text.splitlines() if line.strip()]
    return normalized


def _repair_lines_and_text(lines: Sequence[Any], text: str) -> tuple[list[str], str, list[str]]:
    normalized_warnings: list[str] = []
    had_embedded_newline = any(("\n" in str(item) or "\r" in str(item)) for item in lines)
    normalized_lines = _normalize_lines(lines, text)

    if had_embedded_newline:
        normalized_warnings.append("repair:lines_split_embedded_newlines")
    if not any(str(item).strip() for item in lines) and str(text).strip() and normalized_lines:
        normalized_warnings.append("repair:rebuilt_lines_from_text")

    normalized_text = "\n".join(normalized_lines).strip()
    if str(text).strip() != normalized_text:
        normalized_warnings.append("TEXT_JOIN_FIXED")

    return normalized_lines, normalized_text, list(dict.fromkeys(normalized_warnings))


_OCR_VOWELS = set("aeiouyàáâãäåæèéêëìíîïòóôõöùúûüýÿœ")


def _junk_ratio(text: str) -> float:
    tokens = [token for token in re.split(r"\s+", str(text or "").strip()) if token]
    considered = 0
    junk = 0
    for token in tokens:
        if len(token) < 3:
            continue
        considered += 1
        letters = [ch for ch in token if ch.isalpha()]
        letters_count = len(letters)
        if letters_count < 3:
            junk += 1
            continue
        vowel_count = sum(1 for ch in letters if ch.lower() in _OCR_VOWELS)
        vowel_ratio = vowel_count / max(1, letters_count)
        non_letters = len(token) - letters_count
        non_letter_ratio = non_letters / max(1, len(token))
        if vowel_ratio < 0.20 or non_letter_ratio > 0.30:
            junk += 1
    if considered == 0:
        return 0.0
    return junk / considered


def _letter_count(text: str) -> int:
    return sum(1 for ch in str(text or "") if ch.isalpha())


def _apply_ocr_confidence_caps(text: str, confidence: float, warnings: list[str]) -> tuple[float, list[str]]:
    junk_ratio = _junk_ratio(text)
    letters_count = _letter_count(text)
    next_warnings = list(warnings)
    if junk_ratio >= 0.35 or (letters_count < 120 and junk_ratio >= 0.25):
        capped = min(confidence, 0.75)
        next_warnings.extend(["ocr:confidence_capped", f"ocr:junk_ratio={junk_ratio:.2f}"])
        return capped, list(dict.fromkeys(next_warnings))
    if junk_ratio >= 0.20:
        capped = min(confidence, 0.85)
        next_warnings.append(f"ocr:junk_ratio={junk_ratio:.2f}")
        return capped, list(dict.fromkeys(next_warnings))
    return confidence, list(dict.fromkeys(next_warnings))


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

    normalized_lines, normalized_text, normalized_warnings = _repair_lines_and_text(lines, text)
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
    filtered = []
    for suggestion in suggestions:
        category = str(suggestion.category or "").lower()
        if any(token in category for token in ("miniature", "illustration", "graphic")):
            continue
        filtered.append(suggestion)
    if not filtered:
        filtered = list(suggestions)

    lines = ["Location suggestions (x,y,w,h in full-page coordinates; hints only):", ""]
    xmins: list[float] = []
    ymins: list[float] = []
    xmaxs: list[float] = []
    ymaxs: list[float] = []

    for idx, suggestion in enumerate(filtered, start=1):
        bbox = [float(v) for v in (suggestion.bbox_xywh or [])]
        if len(bbox) != 4:
            continue
        region_id = str(suggestion.region_id or f"r{idx}")
        category = str(suggestion.category or "text")
        x, y, w, h = bbox
        xmins.append(x)
        ymins.append(y)
        xmaxs.append(x + w)
        ymaxs.append(y + h)
        lines.append(f"{region_id} [{category}]: ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")
    if xmins and ymins and xmaxs and ymaxs:
        tx = min(xmins)
        ty = min(ymins)
        tw = max(xmaxs) - tx
        th = max(ymaxs) - ty
        lines.insert(0, f"Suggested text block: ({tx:.1f}, {ty:.1f}, {tw:.1f}, {th:.1f})")
        lines.insert(1, "")
    if len(lines) <= 2:
        return ""
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tile-based OCR helpers: crop → OCR each tile → stitch in reading order
# ---------------------------------------------------------------------------

import logging as _logging

_tile_logger = _logging.getLogger("saia_ocr_agent.tiles")

_TILE_MIN_AREA_RATIO = 0.0003  # tiles < 0.03% of image area are noise
_TILE_PAD_RATIO = 0.06  # 6% padding around each tile (avoids clipping ascenders/descenders)
_TILE_MIN_SIDE = 64  # minimum crop dimension (px) before upscaling
_TILE_UPSCALE_TARGET = 512  # upscale tiny crops so shortest side >= this
_TILE_COLUMN_SPLIT_X_RATIO = 0.45  # x-center threshold to distinguish columns

SAIA_TILE_SYSTEM_ADDENDUM = (
    "\nThis is a cropped region of a manuscript page. "
    "Transcribe ONLY what is visible in this crop. "
    "Do NOT repeat text that is not present in the crop. "
    "Preserve line breaks as they appear in the crop. "
    "Do NOT output Arabic numerals (0-9) unless clearly written as numerals. "
    "Do NOT insert spaces inside a single word.\n"
    "REMEMBER: Gothic minims (vertical strokes) encode i/u/n/m — read them carefully. "
    "The z-shaped mark between words = 'et' (Tironian nota). "
    "Long-s has no crossbar; f has a crossbar. "
    "Bars over letters = abbreviation marks — keep them."
)


def _suggestions_to_rects(
    suggestions: Sequence[SaiaOCRLocationSuggestion],
    img_w: int,
    img_h: int,
) -> list[tuple[int, int, int, int]]:
    """Convert location suggestions (xywh) → pixel rects (x1,y1,x2,y2), filtering tiny boxes."""
    img_area = float(img_w * img_h)
    rects: list[tuple[int, int, int, int]] = []
    for s in suggestions:
        bbox = list(s.bbox_xywh or [])
        if len(bbox) != 4:
            continue
        x, y, w, h = [float(v) for v in bbox]
        if w <= 0 or h <= 0:
            continue
        # Filter tiny boxes
        if (w * h) / max(img_area, 1.0) < _TILE_MIN_AREA_RATIO:
            continue
        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = min(img_w, int(round(x + w)))
        y2 = min(img_h, int(round(y + h)))
        if (x2 - x1) < 8 or (y2 - y1) < 8:
            continue
        rects.append((x1, y1, x2, y2))
    return rects


def _merge_overlapping_rects(
    rects: list[tuple[int, int, int, int]],
    merge_distance: int = 10,
) -> list[tuple[int, int, int, int]]:
    """Merge rectangles that overlap or are within `merge_distance` pixels."""
    if not rects:
        return []
    merged = list(rects)
    changed = True
    while changed:
        changed = False
        new_merged: list[tuple[int, int, int, int]] = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            ax1, ay1, ax2, ay2 = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = merged[j]
                # Check if they overlap or are within merge_distance
                if (
                    ax1 - merge_distance <= bx2
                    and ax2 + merge_distance >= bx1
                    and ay1 - merge_distance <= by2
                    and ay2 + merge_distance >= by1
                ):
                    ax1 = min(ax1, bx1)
                    ay1 = min(ay1, by1)
                    ax2 = max(ax2, bx2)
                    ay2 = max(ay2, by2)
                    used[j] = True
                    changed = True
            new_merged.append((ax1, ay1, ax2, ay2))
            used[i] = True
        merged = new_merged
    return merged


# ---------------------------------------------------------------------------
# Morphological block merging via PIL mask dilation + BFS connected components
# ---------------------------------------------------------------------------

_MORPH_DOWNSCALE = 4          # work on 1/4-resolution mask for speed
_MORPH_DILATE_RADIUS = 18     # dilation radius in downscaled px (~72 original px)
_MORPH_MIN_BLOCK_AREA = 400   # ignore tiny connected components (in downscaled px²)


def _rects_to_block_rects(
    rects: list[tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
) -> list[tuple[int, int, int, int]]:
    """Merge nearby suggestion rects into block-level rects using morphological dilation.

    Algorithm:
      1. Paint all suggestion rects onto a binary mask (downscaled for speed).
      2. Dilate the mask with a circular kernel to bridge nearby regions.
      3. Find connected components via BFS.
      4. For each component return the bounding box (upscaled back to original coords).

    This replaces the naive rect-overlap merge and correctly merges text lines
    that belong to the same column/block even when there are small gaps.
    """
    if not rects:
        return []

    ds = _MORPH_DOWNSCALE
    mw = max(1, img_w // ds)
    mh = max(1, img_h // ds)

    # --- 1. Create binary mask ---
    mask = Image.new("L", (mw, mh), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in rects:
        draw.rectangle(
            [x1 // ds, y1 // ds, max(x1 // ds + 1, x2 // ds), max(y1 // ds + 1, y2 // ds)],
            fill=255,
        )

    # --- 2. Dilate using PIL's MaxFilter (square approximation of dilation) ---
    from PIL import ImageFilter
    # Apply MaxFilter repeatedly; kernel size 3 means 1-px dilation per pass.
    # Number of passes ≈ radius to reach desired dilation radius.
    passes = max(1, _MORPH_DILATE_RADIUS)
    for _ in range(passes):
        mask = mask.filter(ImageFilter.MaxFilter(size=3))

    # --- 3. BFS connected components on the dilated mask ---
    pixels = list(mask.getdata())
    labels = [0] * (mw * mh)
    label_id = 0

    def _bfs(start_idx: int, lid: int) -> tuple[int, int, int, int]:
        """BFS flood-fill returning bbox (x1, y1, x2, y2) in mask coords."""
        queue = [start_idx]
        labels[start_idx] = lid
        min_x = start_idx % mw
        max_x = min_x
        min_y = start_idx // mw
        max_y = min_y
        head = 0
        while head < len(queue):
            idx = queue[head]
            head += 1
            cx = idx % mw
            cy = idx // mw
            if cx < min_x:
                min_x = cx
            if cx > max_x:
                max_x = cx
            if cy < min_y:
                min_y = cy
            if cy > max_y:
                max_y = cy
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < mw and 0 <= ny < mh:
                    ni = ny * mw + nx
                    if labels[ni] == 0 and pixels[ni] > 0:
                        labels[ni] = lid
                        queue.append(ni)
        return min_x, min_y, max_x, max_y

    component_bboxes: list[tuple[int, int, int, int]] = []
    for i in range(mw * mh):
        if pixels[i] > 0 and labels[i] == 0:
            label_id += 1
            bx1, by1, bx2, by2 = _bfs(i, label_id)
            area = (bx2 - bx1 + 1) * (by2 - by1 + 1)
            if area >= _MORPH_MIN_BLOCK_AREA:
                component_bboxes.append((bx1, by1, bx2, by2))

    # --- 4. Upscale bboxes back to original coordinates ---
    block_rects: list[tuple[int, int, int, int]] = []
    for bx1, by1, bx2, by2 in component_bboxes:
        ox1 = max(0, bx1 * ds)
        oy1 = max(0, by1 * ds)
        ox2 = min(img_w, (bx2 + 1) * ds)
        oy2 = min(img_h, (by2 + 1) * ds)
        if (ox2 - ox1) >= 16 and (oy2 - oy1) >= 16:
            block_rects.append((ox1, oy1, ox2, oy2))

    if not block_rects:
        # Fallback: if morphological merge yields nothing, use simple rect merge
        return _merge_overlapping_rects_simple(rects, merge_distance=max(10, min(img_w, img_h) // 40))

    return block_rects


def _merge_overlapping_rects_simple(
    rects: list[tuple[int, int, int, int]],
    merge_distance: int = 10,
) -> list[tuple[int, int, int, int]]:
    """Simple fallback: merge rects that overlap or are within merge_distance px."""
    if not rects:
        return []
    merged = list(rects)
    changed = True
    while changed:
        changed = False
        new_merged: list[tuple[int, int, int, int]] = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            ax1, ay1, ax2, ay2 = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = merged[j]
                if (
                    ax1 - merge_distance <= bx2
                    and ax2 + merge_distance >= bx1
                    and ay1 - merge_distance <= by2
                    and ay2 + merge_distance >= by1
                ):
                    ax1 = min(ax1, bx1)
                    ay1 = min(ay1, by1)
                    ax2 = max(ax2, bx2)
                    ay2 = max(ay2, by2)
                    used[j] = True
                    changed = True
            new_merged.append((ax1, ay1, ax2, ay2))
            used[i] = True
        merged = new_merged
    return merged


def _pad_rect(
    rect: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    pad_ratio: float = _TILE_PAD_RATIO,
) -> tuple[int, int, int, int]:
    """Add *dynamic* padding around a rectangle, clamped to image bounds.

    Padding adapts to region size:
      - Small regions (<300px short side) get extra padding to avoid
        clipping ascenders, descenders, and decorated initials.
      - The vertical pad is always >= horizontal pad because manuscript
        text lines have ascenders/descenders that extend above/below.
    """
    x1, y1, x2, y2 = rect
    rw = x2 - x1
    rh = y2 - y1
    short_side = min(rw, rh)

    # Base pad from ratio
    base_px = max(4, int(round(rw * pad_ratio)))
    base_py = max(4, int(round(rh * pad_ratio)))

    # Dynamic boost for small regions (ascender/descender protection)
    if short_side < 300:
        boost = max(8, int(round(short_side * 0.12)))
        base_px += boost
        base_py += boost

    # Vertical pad should be >= horizontal pad (text has vertical extenders)
    py = max(base_py, base_px)
    px = base_px

    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(img_w, x2 + px),
        min(img_h, y2 + py),
    )


def _order_tiles_reading(
    rects: list[tuple[int, int, int, int]],
    img_w: int,
) -> list[tuple[int, int, int, int]]:
    """Sort tiles in reading order with adaptive column detection.

    Instead of a fixed 45% x-threshold, uses gap-based clustering:
      1. Sort x-centers, find the largest horizontal gap.
      2. If the gap is > 8% of page width AND both sides have tiles,
         treat as two columns.
      3. For each column group: sort top→bottom.
      4. Marginal regions (very narrow, at page edges) are appended last
         to avoid polluting the main text flow.
    """
    if len(rects) <= 1:
        return list(rects)

    # --- Separate marginal regions from main text ---
    main_rects: list[tuple[int, int, int, int]] = []
    marginal_rects: list[tuple[int, int, int, int]] = []
    for r in rects:
        rw = r[2] - r[0]
        # Marginal: very narrow (<12% of page width) AND at page edges
        is_marginal = (
            rw < float(img_w) * 0.12
            and (r[0] < float(img_w) * 0.08 or r[2] > float(img_w) * 0.92)
        )
        if is_marginal:
            marginal_rects.append(r)
        else:
            main_rects.append(r)

    if not main_rects:
        # All marginal → just sort top-to-bottom
        return sorted(rects, key=lambda r: (r[1], r[0]))

    # --- Adaptive column detection via gap analysis ---
    x_centers = sorted([(r[0] + r[2]) / 2.0 for r in main_rects])

    # Find largest gap between consecutive x-centers
    best_gap = 0.0
    best_gap_pos = 0.0
    for i in range(len(x_centers) - 1):
        gap = x_centers[i + 1] - x_centers[i]
        if gap > best_gap:
            best_gap = gap
            best_gap_pos = (x_centers[i] + x_centers[i + 1]) / 2.0

    # Two columns if gap > 8% of page width
    is_two_col = best_gap > float(img_w) * 0.08

    if is_two_col:
        left = [(r, (r[0] + r[2]) / 2.0) for r in main_rects if (r[0] + r[2]) / 2.0 < best_gap_pos]
        right = [(r, (r[0] + r[2]) / 2.0) for r in main_rects if (r[0] + r[2]) / 2.0 >= best_gap_pos]
        if left and right:
            left_sorted = sorted(left, key=lambda t: t[0][1])
            right_sorted = sorted(right, key=lambda t: t[0][1])
            ordered = [r for r, _ in left_sorted] + [r for r, _ in right_sorted]
        else:
            ordered = sorted(main_rects, key=lambda r: (r[1], r[0]))
    else:
        ordered = sorted(main_rects, key=lambda r: (r[1], r[0]))

    # Marginal regions last, sorted top→bottom
    if marginal_rects:
        ordered.extend(sorted(marginal_rects, key=lambda r: (r[1], r[0])))

    return ordered


def _crop_and_maybe_upscale(
    image: Image.Image,
    rect: tuple[int, int, int, int],
) -> Image.Image:
    """Crop tile from image, upscale if too small, and enhance for OCR."""
    x1, y1, x2, y2 = rect
    crop = image.crop((x1, y1, x2, y2))
    cw, ch = crop.size
    if cw < _TILE_UPSCALE_TARGET or ch < _TILE_UPSCALE_TARGET:
        short_side = min(cw, ch)
        if short_side > 0:
            scale = max(1.0, float(_TILE_UPSCALE_TARGET) / float(short_side))
            if scale > 1.0:
                new_w = max(1, int(round(cw * scale)))
                new_h = max(1, int(round(ch * scale)))
                crop = crop.resize((new_w, new_h), Image.Resampling.BICUBIC)
    # Enhance the tile crop for better OCR
    crop = _enhance_manuscript_image(crop)
    return crop


def _ocr_sanity_metrics(text: str) -> dict[str, float]:
    """Compute simple sanity metrics on OCR output."""
    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    if not tokens:
        return {"single_char_ratio": 0.0, "digit_ratio": 0.0, "junk_ratio": 0.0}

    single_char = sum(1 for t in tokens if len(t) == 1)
    digit_tokens = sum(1 for t in tokens if any(c.isdigit() for c in t))
    non_letter = sum(1 for t in tokens if not any(c.isalpha() for c in t))

    n = float(len(tokens))
    return {
        "single_char_ratio": single_char / n,
        "digit_ratio": digit_tokens / n,
        "junk_ratio": non_letter / n,
    }


_SANITY_SINGLE_CHAR_THRESHOLD = 0.30
_SANITY_DIGIT_THRESHOLD = 0.25
_SANITY_JUNK_THRESHOLD = 0.35


def _is_sane_ocr(text: str) -> bool:
    """Return True if OCR text passes basic sanity checks."""
    metrics = _ocr_sanity_metrics(text)
    if metrics["single_char_ratio"] > _SANITY_SINGLE_CHAR_THRESHOLD:
        return False
    if metrics["digit_ratio"] > _SANITY_DIGIT_THRESHOLD:
        return False
    if metrics["junk_ratio"] > _SANITY_JUNK_THRESHOLD:
        return False
    return True


# ---------------------------------------------------------------------------
# Diplomatic sanity: compute_ocr_sanity + enforce_diplomatic_uncertainty
# ---------------------------------------------------------------------------

_IMPROBABLE_BIGRAMS = {"qj", "vvv", "iii", "kk", "iiit", "xq", "jq", "zx", "qx"}
_ROMAN_NUMERAL_RE = re.compile(r"^[MDCLXVI]{2,}$", re.IGNORECASE)
_YEAR_LIKE_RE = re.compile(r"^(anno|an|m[cdxlvi]+)$", re.IGNORECASE)
_STANDALONE_FUNCTION_WORDS: dict[str, set[str]] = {
    "latin": {"a", "e", "i", "o", "u", "y", "et"},
    "old_french": {"a", "e", "i", "o", "u", "y", "et", "à"},
    "middle_french": {"a", "e", "i", "o", "u", "y", "et", "à"},
    "french": {"a", "e", "i", "o", "u", "y", "et", "à"},
    "old_english": {"a", "i"},
    "middle_english": {"a", "i", "o"},
}
_CONSONANT_MERGE_SET = set("cdlnprstvxCDLNPRSTVX")
_CONSONANT_LEADING_SET = set("cdlnprstvx")
_ZERO_WIDTH_RE = re.compile(r"[\ufeff\u200b\u200c\u200d\u2060\ufffe]")
_STRAY_K_ENDING = re.compile(r"^([a-zA-Z\u00C0-\u00FF]{3,}[bdfghjlmnpqrstvwxz])k$", re.IGNORECASE)
# Minimum consecutive single-char tokens to treat as a fragment run and collapse
_FRAGMENT_RUN_MIN = 3

# Languages where 'w' and 'k' are rare OCR artifacts (Old French, Latin, etc.)
_WK_SUSPICIOUS_LANGUAGES = {"old_french", "middle_french", "french", "latin"}
# Tokens that are entirely w/k junk → replace with '[…]'
_WK_JUNK_TOKENS = {"w", "k", "wr", "wk", "kw", "ww", "kk"}


def compute_sanity(text: str) -> dict[str, float]:
    """Compute archival sanity metrics for quality gate.

    Returns dict with:
      - single_char_ratio
      - digit_ratio
      - weird_ratio  (token len>=10 with vowel_ratio<0.20 OR improbable bigrams)
      - junk_ratio   (tokens where >40% of chars are non-letter / non-'?')
      - leading_fragment_ratio
      - uncertainty_marker_ratio
    """
    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    lines = [ln for ln in text.splitlines() if ln.strip()]
    n = float(len(tokens)) if tokens else 1.0
    n_lines = float(len(lines)) if lines else 1.0

    single_char = sum(1 for t in tokens if len(t) == 1)
    digit_tokens = sum(1 for t in tokens if any(ch.isdigit() for ch in t))

    weird = 0
    junk = 0
    for t in tokens:
        letters = [ch for ch in t if ch.isalpha()]
        lcount = len(letters)
        # Junk: tokens where >40% of chars are non-letter / non-'?' and len>=2
        if len(t) >= 2 and t not in ("[…]", "?"):
            non_letter = sum(1 for ch in t if not ch.isalpha() and ch != "?" and ch != "…" and ch not in "[]'''-–—.,;:")
            if non_letter / max(1, len(t)) > 0.40:
                junk += 1
        if lcount == 0:
            continue
        lower_t = t.lower()
        # Token len >= 10 with low vowels
        if lcount >= 10:
            vowel_ratio = sum(1 for ch in letters if ch.lower() in _OCR_VOWELS) / max(1, lcount)
            if vowel_ratio < 0.20:
                weird += 1
                continue
        # Consonant runs >= 5 (unlikely in Latin-script)
        if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", lower_t):
            weird += 1
            continue
        # Improbable bigrams
        if any(bg in lower_t for bg in _IMPROBABLE_BIGRAMS):
            weird += 1
            continue
        # 3+ repeated same char
        if re.search(r"(.)\1{2,}", lower_t) and not _ROMAN_NUMERAL_RE.match(t):
            weird += 1
            continue

    uncertainty_markers = sum(1 for t in tokens if t == "?" or t == "[…]" or "?" in t)

    leading_frag = 0
    for ln in lines:
        parts = ln.strip().split()
        if not parts:
            continue
        first = parts[0]
        if len(first) == 1 or (first.isdigit() and len(first) == 1):
            leading_frag += 1

    return {
        "single_char_ratio": single_char / n,
        "digit_ratio": digit_tokens / n,
        "weird_ratio": weird / n,
        "junk_ratio": junk / n,
        "leading_fragment_ratio": leading_frag / n_lines,
        "uncertainty_marker_ratio": uncertainty_markers / n,
    }


# Backward-compatible alias
compute_ocr_sanity = compute_sanity


def _quality_label_from_sanity(metrics: dict[str, float]) -> str:
    """Compute HIGH / MEDIUM / LOW quality label from sanity metrics."""
    scr = metrics.get("single_char_ratio", metrics.get("single_char_token_ratio", 0.0))
    dr = metrics.get("digit_ratio", metrics.get("digit_token_ratio", 0.0))
    wr = metrics.get("weird_ratio", metrics.get("weird_token_ratio", 0.0))
    jr = metrics.get("junk_ratio", 0.0)
    if scr < 0.05 and dr < 0.01 and wr < 0.08 and jr < 0.03:
        return "HIGH"
    if scr < 0.10 and dr < 0.01 and wr < 0.15 and jr < 0.06:
        return "MEDIUM"
    return "LOW"


def _token_is_suspicious(t: str) -> bool:
    """Return True if a single token should be masked by sanitize rules."""
    letters = [ch for ch in t if ch.isalpha()]
    lcount = len(letters)
    if lcount == 0:
        return False
    lower_t = t.lower()
    # Digit tokens (not roman numerals / year-like)
    if any(ch.isdigit() for ch in t):
        stripped_alpha = "".join(ch for ch in t if ch.isalpha())
        if (
            not _ROMAN_NUMERAL_RE.match(t)
            and not _YEAR_LIKE_RE.match(t)
            and not _ROMAN_NUMERAL_RE.match(stripped_alpha)
        ):
            return True
    # Long token with low vowels
    if lcount >= 12:
        vowel_ratio = sum(1 for ch in letters if ch.lower() in _OCR_VOWELS) / max(1, lcount)
        if vowel_ratio < 0.20:
            return True
    # Improbable bigrams or repeated chars
    if lcount >= 3:
        if any(bg in lower_t for bg in _IMPROBABLE_BIGRAMS):
            return True
        if re.search(r"(.)\1{2,}", lower_t) and not _ROMAN_NUMERAL_RE.match(t):
            return True
    # Consonant run >= 5 (exempt Roman numerals)
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", lower_t) and not _ROMAN_NUMERAL_RE.match(t):
        return True
    return False


def _collapse_single_char_runs(
    tokens: list[str],
    func_words: set[str],
) -> tuple[list[str], int]:
    """Collapse runs of >= _FRAGMENT_RUN_MIN consecutive single-char tokens.

    Runs of single-char tokens like "a a mno y a nis" indicate severe
    segmentation failure. We collapse each such run into '[…]'.

    Single-char tokens that are legitimate function words are still counted
    in the run because the pattern "a a … a" is fragment noise, not real
    function words appearing consecutively.

    Returns (new_tokens, replacement_count).
    """
    out: list[str] = []
    replacements = 0
    i = 0
    n = len(tokens)
    while i < n:
        # Check if we're starting a run of single-char tokens
        if len(tokens[i]) == 1 and tokens[i].isalpha():
            # Count the run
            run_end = i + 1
            while run_end < n and len(tokens[run_end]) == 1 and tokens[run_end].isalpha():
                run_end += 1
            run_len = run_end - i
            if run_len >= _FRAGMENT_RUN_MIN:
                # Collapse the entire run to '[…]'
                out.append("[…]")
                replacements += run_len
                i = run_end
                continue
        out.append(tokens[i])
        i += 1
    return out, replacements


def _collapse_fragment_sequences(
    tokens: list[str],
    func_words: set[str],
) -> tuple[list[str], int]:
    """Collapse sequences where >=60% of tokens in a window of 5+ are single-char.

    Handles patterns like "a a mno y a nis" where short non-single-char
    tokens are interspersed among single-char fragments.
    """
    n = len(tokens)
    if n < 5:
        return tokens, 0

    # Mark positions that are single-char alphabetic
    is_single = [len(t) == 1 and t.isalpha() for t in tokens]

    # Use a sliding window to find stretches dominated by single-char tokens
    window = 5
    mask = [False] * n
    for start in range(n - window + 1):
        single_count = sum(1 for j in range(start, start + window) if is_single[j])
        if single_count / window >= 0.60:
            for j in range(start, start + window):
                mask[j] = True

    # Apply mask: consecutive masked regions become a single '[…]'
    out: list[str] = []
    replacements = 0
    i = 0
    while i < n:
        if mask[i]:
            # Consume entire masked run
            while i < n and mask[i]:
                replacements += 1
                i += 1
            out.append("[…]")
        else:
            out.append(tokens[i])
            i += 1
    return out, replacements


def sanitize_lines(
    lines: list[str],
    detected_language: str,
    script_hint: str,
) -> tuple[list[str], int]:
    """Deterministic diplomatic sanitizer.

    Returns (sanitized_lines, replacement_count).
    Rules:
      - Remove BOM / zero-width chars, drop empty lines.
      - Merge leading consonant initials {c,d,l,n,p,r,s,t,v,w,x} with next token.
      - Collapse runs of >=3 consecutive single-char tokens to '[…]'.
      - Collapse fragment-dominated sequences (>=60% single-char in windows of 5+).
      - Digit tokens (not roman numerals) → '[…]'.
      - Long low-vowel tokens (>=12 letters, <20% vowels) → '[…]'.
      - Improbable bigrams / 3+ repeated char → '[…]'.
      - Consonant run >= 5 in a token → '[…]'.
      - Trailing stray 'k' (OCR noise): replace trailing k with '?' if base
        is plausible, else '[…]'.
    """
    func_words = _STANDALONE_FUNCTION_WORDS.get(
        detected_language, _STANDALONE_FUNCTION_WORDS.get("latin", set())
    )
    replacements = 0
    result_lines: list[str] = []

    for line in lines:
        # Remove BOM / zero-width characters
        line = _ZERO_WIDTH_RE.sub("", line)
        stripped = line.strip()
        if not stripped:
            continue  # drop empty lines

        tokens = stripped.split()
        out: list[str] = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            lower_t = t.lower()

            # --- w/k masking (language-aware) ---
            # Skip tokens with digits (let digit rule handle those)
            if detected_language in _WK_SUSPICIOUS_LANGUAGES and not any(ch.isdigit() for ch in t):
                if lower_t in _WK_JUNK_TOKENS:
                    out.append("[…]")
                    replacements += 1
                    i += 1
                    continue
                if len(t) >= 2 and ("w" in lower_t or "k" in lower_t):
                    masked = ""
                    did_mask = False
                    for ch in t:
                        if ch.lower() in ("w", "k"):
                            masked += "?"
                            did_mask = True
                        else:
                            masked += ch
                    if did_mask:
                        out.append(masked)
                        replacements += 1
                        i += 1
                        continue

            # --- Leading consonant merge ---
            if (
                len(t) == 1
                and lower_t in _CONSONANT_LEADING_SET
                and lower_t not in func_words
                and i + 1 < len(tokens)
                and any(ch.isalpha() for ch in tokens[i + 1])
            ):
                merged = t + tokens[i + 1]
                out.append(merged)
                replacements += 1
                i += 2
                continue

            # --- Digit token → '[…]' ---
            if any(ch.isdigit() for ch in t):
                stripped_alpha = "".join(ch for ch in t if ch.isalpha())
                if (
                    not _ROMAN_NUMERAL_RE.match(t)
                    and not _YEAR_LIKE_RE.match(t)
                    and not _ROMAN_NUMERAL_RE.match(stripped_alpha)
                ):
                    out.append("[…]")
                    replacements += 1
                    i += 1
                    continue

            # --- Long low-vowel token → '[…]' ---
            letters = [ch for ch in t if ch.isalpha()]
            lcount = len(letters)
            if lcount >= 12:
                vowel_count = sum(1 for ch in letters if ch.lower() in _OCR_VOWELS)
                if vowel_count / max(1, lcount) < 0.20:
                    out.append("[…]")
                    replacements += 1
                    i += 1
                    continue

            # --- Improbable bigrams / repeated chars → '[…]' ---
            if lcount >= 3:
                if any(bg in lower_t for bg in _IMPROBABLE_BIGRAMS):
                    out.append("[…]")
                    replacements += 1
                    i += 1
                    continue
                # 3+ repeated char
                if re.search(r"(.)\1{2,}", lower_t) and not _ROMAN_NUMERAL_RE.match(t):
                    out.append("[…]")
                    replacements += 1
                    i += 1
                    continue

            # --- Consonant run >= 5 → '[…]' (exempt Roman numerals) ---
            if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", lower_t) and not _ROMAN_NUMERAL_RE.match(t):
                out.append("[…]")
                replacements += 1
                i += 1
                continue

            # --- Trailing stray 'k' (OCR noise) ---
            m = _STRAY_K_ENDING.match(t)
            if m:
                base = m.group(1)
                base_vowels = sum(1 for ch in base if ch.lower() in _OCR_VOWELS)
                if base_vowels > 0:
                    out.append(base + "?")
                else:
                    out.append("[…]")
                replacements += 1
                i += 1
                continue

            out.append(t)
            i += 1

        # --- Pass 2: Collapse consecutive single-char fragment runs ---
        out, run_repl = _collapse_single_char_runs(out, func_words)
        replacements += run_repl

        # --- Pass 3: Collapse fragment-dominated sequences ---
        out, seq_repl = _collapse_fragment_sequences(out, func_words)
        replacements += seq_repl

        # If >30% of line tokens were fully replaced with '[…]', 
        # the line is not archival-safe. Diplomatic '?' marks are fine.
        full_mask_count = sum(1 for tok in out if tok == "[…]")
        if out and (full_mask_count / max(1, len(out))) > 0.30:
            result_lines.append("[…]")
            replacements += len(out)
            continue

        # Deduplicate consecutive '[…]' markers
        deduped: list[str] = []
        for tok in out:
            if tok == "[…]" and deduped and deduped[-1] == "[…]":
                continue
            deduped.append(tok)

        result_lines.append(" ".join(deduped))

    return result_lines, replacements


def enforce_diplomatic_uncertainty(
    lines: list[str],
    script_hint: str,
    detected_language: str = "latin",
) -> tuple[list[str], bool]:
    """Backward-compatible wrapper around sanitize_lines."""
    sanitized, count = sanitize_lines(lines, detected_language, script_hint)
    return sanitized, count > 0


def quality_gate_enforce(
    lines: list[str],
    max_line_suspicious_ratio: float = 0.30,
) -> tuple[list[str], int]:
    """Second-pass masking: replace entire suspicious lines with '[…]'.

    A line is suspicious when > max_line_suspicious_ratio of its tokens
    are flagged by _token_is_suspicious.

    Returns (gated_lines, masked_line_count).
    """
    gated: list[str] = []
    masked_count = 0
    for line in lines:
        tokens = line.split()
        if not tokens:
            gated.append(line)
            continue
        suspicious_count = sum(1 for t in tokens if _token_is_suspicious(t))
        ratio = suspicious_count / len(tokens)
        # Mask lines with >30% suspicious tokens, or lines that are just '[…]' already
        if ratio > max_line_suspicious_ratio:
            gated.append("[…]")
            masked_count += 1
        else:
            gated.append(line)
    return gated, masked_count


def force_uncertainty_markers(
    lines: list[str],
    sanity: dict[str, float],
    weird_threshold: float = 0.05,
    junk_threshold: float = 0.03,
) -> tuple[list[str], int]:
    """Force uncertainty markers when text is noisy but has none.

    If after sanitization uncertainty_marker_ratio == 0 AND
    (weird_ratio > weird_threshold OR junk_ratio > junk_threshold),
    force-mask the top-N weirdest/most-suspicious tokens until at least
    one uncertainty marker is present.

    Returns (updated_lines, forced_mask_count).
    """
    unc = sanity.get("uncertainty_marker_ratio", 0.0)
    wr = sanity.get("weird_ratio", 0.0)
    jr = sanity.get("junk_ratio", 0.0)

    # Only force if no uncertainty but metrics indicate noise
    if unc > 0.0:
        return lines, 0
    if wr <= weird_threshold and jr <= junk_threshold:
        return lines, 0

    # Collect all tokens with suspicion scores, ranked by suspiciousness
    scored: list[tuple[int, int, float]] = []  # (line_idx, tok_idx, score)
    for li, line in enumerate(lines):
        tokens = line.split()
        for ti, t in enumerate(tokens):
            if t in ("[…]", "?"):
                continue
            score = 0.0
            if _token_is_suspicious(t):
                score += 1.0
            letters = [ch for ch in t if ch.isalpha()]
            lcount = len(letters)
            if lcount >= 6:
                vr = sum(1 for ch in letters if ch.lower() in _OCR_VOWELS) / max(1, lcount)
                if vr < 0.25:
                    score += 0.5
            if lcount >= 3 and re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", t.lower()):
                score += 0.3
            if score > 0:
                scored.append((li, ti, score))

    if not scored:
        # Nothing suspicious at token level; add a trailing uncertainty marker
        if lines:
            lines[-1] = lines[-1].rstrip() + " [?]"
            return lines, 1
        return lines, 0

    # Sort by suspicion score descending, mask top-N (at least 1, at most 3)
    scored.sort(key=lambda x: -x[2])
    n_to_mask = min(max(1, int(len(scored) * 0.10)), 3)
    forced = 0
    # Group by line for efficiency
    masks_by_line: dict[int, set[int]] = {}
    for li, ti, _score in scored[:n_to_mask]:
        masks_by_line.setdefault(li, set()).add(ti)
        forced += 1

    result: list[str] = []
    for li, line in enumerate(lines):
        if li in masks_by_line:
            tokens = line.split()
            for ti in masks_by_line[li]:
                if ti < len(tokens):
                    tokens[ti] = tokens[ti] + "?"
            result.append(" ".join(tokens))
        else:
            result.append(line)

    return result, forced


def format_sanity_summary(sanity: dict[str, float]) -> str:
    return (
        "SANITY "
        f"single_char={sanity.get('single_char_ratio', 0.0):.3f} "
        f"weird={sanity.get('weird_ratio', 0.0):.3f} "
        f"junk={sanity.get('junk_ratio', 0.0):.3f} "
        f"uncert={sanity.get('uncertainty_marker_ratio', 0.0):.3f}"
    )


def sanity_adjust_confidence(confidence_raw: float, sanity: dict[str, float]) -> float:
    """Sanity-adjust confidence: cannot be high when metrics are bad.

    Uses a multiplicative penalty for single_char_ratio, weird_ratio,
    and junk_ratio. Adds an extra penalty when uncertainty_marker_ratio == 0
    but weird_ratio is non-trivial (the "confident garbage" scenario).
    """
    scr = sanity.get("single_char_ratio", 0.0)
    wr = sanity.get("weird_ratio", 0.0)
    jr = sanity.get("junk_ratio", 0.0)
    umr = sanity.get("uncertainty_marker_ratio", 0.0)

    adjusted = confidence_raw * (1.0 - scr) * (1.0 - wr) * (1.0 - jr)

    # Extra penalty: "confident garbage" = no uncertainty markers but noisy output
    if umr == 0.0 and (wr > 0.05 or jr > 0.03):
        adjusted *= 0.70

    # Extra penalty for very high single_char_ratio
    if scr > 0.15:
        adjusted *= 0.60

    return max(0.05, min(0.95, adjusted))


def build_saia_ocr_messages(
    image_b64: str,
    *,
    repair_json: bool = False,
    location_suggestions: Sequence[SaiaOCRLocationSuggestion] | None = None,
) -> list[dict[str, Any]]:
    text_prompt = SAIA_OCR_USER_PROMPT
    location_block = _format_location_suggestions(location_suggestions or [])
    if location_block:
        text_prompt = (
            f"{text_prompt}\n\n"
            "Transcribe only within the suggested text block; ignore the miniature.\n\n"
            f"{location_block}"
        )
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


def build_tile_ocr_messages(
    tile_image_b64: str,
    *,
    repair_json: bool = False,
    tile_index: int = 0,
    total_tiles: int = 1,
) -> list[dict[str, Any]]:
    """Build messages for a single tile crop OCR. Uses a system prompt addendum to
    prevent duplication and keep output focused on visible content only."""
    system_prompt = SAIA_OCR_SYSTEM_PROMPT + SAIA_TILE_SYSTEM_ADDENDUM
    text_prompt = (
        f"Tile {tile_index + 1} of {total_tiles}. "
        "Transcribe ONLY what is visible in this cropped region.\n\n"
        + SAIA_OCR_USER_PROMPT
    )
    if repair_json:
        text_prompt = f"{text_prompt}\n\n{SAIA_OCR_JSON_REPAIR_PROMPT}"
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tile_image_b64}"}},
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


def _pick_better_result(tile_result: "SaiaOCRResponse", fullpage_result: "SaiaOCRResponse") -> "SaiaOCRResponse":
    """Choose the better OCR result between tile-based and full-page based on sanity metrics."""
    tile_text = str(tile_result.text or "").strip()
    full_text = str(fullpage_result.text or "").strip()

    if not tile_text:
        return fullpage_result
    if not full_text:
        return tile_result

    tile_metrics = _ocr_sanity_metrics(tile_text)
    full_metrics = _ocr_sanity_metrics(full_text)

    # Compute a simple score: lower is better (less junk)
    def _score(m: dict[str, float], conf: float) -> float:
        penalty = m["single_char_ratio"] * 2.0 + m["digit_ratio"] * 1.5 + m["junk_ratio"] * 1.0
        return penalty - (conf * 0.5)

    tile_score = _score(tile_metrics, float(tile_result.confidence or 0.0))
    full_score = _score(full_metrics, float(fullpage_result.confidence or 0.0))

    _tile_logger.info(
        "Quality comparison: tile_score=%.3f full_score=%.3f (lower is better)",
        tile_score,
        full_score,
    )

    if tile_score <= full_score:
        return tile_result
    return fullpage_result


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

    @staticmethod
    def _apply_diplomatic_enforcement(
        lines: list[str],
        script_hint: str,
        detected_language: str,
    ) -> tuple[list[str], str, list[str]]:
        """Run sanitize_lines + force_uncertainty_markers, return (lines, text, extra_warnings)."""
        sanitized, count = sanitize_lines(lines, detected_language, script_hint)
        sanitized_text = "\n".join(sanitized).strip()
        extra_warnings: list[str] = []
        if count > 0:
            extra_warnings.append(f"SANITIZED_TOKENS:{count}")

        # Force uncertainty markers if text is noisy but has none
        post_san_sanity = compute_sanity(sanitized_text)
        sanitized, forced = force_uncertainty_markers(sanitized, post_san_sanity)
        if forced > 0:
            extra_warnings.append(f"FORCED_UNCERTAINTY:{forced}")
            sanitized_text = "\n".join(sanitized).strip()

        return sanitized, sanitized_text, extra_warnings

    @staticmethod
    def _enrich_raw_json(
        raw_json: Any,
        sanity_metrics: dict[str, float],
        quality_label: str,
    ) -> dict[str, Any]:
        """Pack sanity_metrics + quality_label into the raw_json blob."""
        blob: dict[str, Any] = {}
        if isinstance(raw_json, dict):
            blob.update(raw_json)
        blob["sanity_metrics"] = sanity_metrics
        blob["quality_label"] = quality_label
        return blob

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
        if corrected_text.strip() == text.strip():
            return corrected_lines, corrected_text, ["proofread:no_changes"]

        # ── Proofreader guard: reject hallucinated rewrites ──
        verdict = check_proofread_delta(text, corrected_text)
        if not verdict.accepted:
            return lines, text, [verdict.reason]

        return corrected_lines, corrected_text, []

    def _request_tile_json_from_model(
        self,
        *,
        model: str,
        tile_image_b64: str,
        tile_index: int,
        total_tiles: int,
    ) -> dict[str, Any]:
        """OCR a single tile crop using the tile-specific prompt."""
        messages = build_tile_ocr_messages(
            tile_image_b64,
            repair_json=False,
            tile_index=tile_index,
            total_tiles=total_tiles,
        )
        response = self._chat_completion_with_optional_json_object(model=model, messages=messages)
        parsed = _parse_ocr_payload(str(response.get("text") or ""))
        if parsed is not None:
            return parsed

        repair_messages = build_tile_ocr_messages(
            tile_image_b64,
            repair_json=True,
            tile_index=tile_index,
            total_tiles=total_tiles,
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

    # ---- Column-split fallback ------------------------------------------------

    _COL_SPLIT_MIN_WIDTH = 1200   # only try column split on wide-enough images
    _COL_SPLIT_OVERLAP = 0.04     # 4% overlap between left/right halves

    def _try_column_split_ocr(
        self,
        *,
        image_b64: str,
        model: str,
    ) -> SaiaOCRResponse | None:
        """Split a wide page into left/right halves, OCR each, stitch.

        Returns None when the image is not wide enough to warrant splitting.
        """
        image = _open_rgb_image_from_b64(image_b64)
        img_w, img_h = image.size

        if img_w < self._COL_SPLIT_MIN_WIDTH:
            return None

        # Only try column split when the page is roughly portrait or square
        aspect = img_w / max(1, img_h)
        if aspect > 1.6:
            # Very landscape → unlikely two-column manuscript page
            return None

        overlap_px = max(8, int(img_w * self._COL_SPLIT_OVERLAP))
        mid = img_w // 2

        col_rects = [
            (0, 0, mid + overlap_px, img_h),          # left column
            (mid - overlap_px, 0, img_w, img_h),       # right column
        ]

        all_lines: list[str] = []
        all_warnings: list[str] = ["COLUMN_SPLIT_OCR:2_cols"]
        confidences: list[tuple[float, float]] = []
        languages: list[str] = []

        for idx, rect in enumerate(col_rects):
            crop = image.crop(rect)
            tile_b64 = _encode_png_base64(crop)
            tile_b64, _ = _prepare_image_for_ocr(tile_b64)

            try:
                parsed = self._request_tile_json_from_model(
                    model=model,
                    tile_image_b64=tile_b64,
                    tile_index=idx,
                    total_tiles=2,
                )
            except Exception as exc:
                all_warnings.append(f"col_{idx}: OCR failed: {exc}")
                continue

            tile_lines = [str(ln) for ln in parsed.get("lines", []) if str(ln).strip()]
            tile_lang = _normalize_detected_language(str(parsed.get("detected_language") or "unknown"))
            tile_conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
            tile_warnings = [str(w) for w in parsed.get("warnings", []) if str(w).strip()]
            for w in tile_warnings:
                all_warnings.append(f"col_{idx}: {w}")

            if tile_lines:
                all_lines.extend(tile_lines)
            if tile_lang and tile_lang != "unknown":
                languages.append(tile_lang)
            area = float((rect[2] - rect[0]) * (rect[3] - rect[1]))
            confidences.append((tile_conf, area))
            _tile_logger.info("Column %d: %d lines, lang=%s, conf=%.2f", idx, len(tile_lines), tile_lang, tile_conf)

        if not all_lines:
            return None

        stitched_text = "\n".join(all_lines)
        total_area = sum(a for _, a in confidences) or 1.0
        final_confidence = sum(c * a for c, a in confidences) / total_area
        final_script_hint = _detect_script_hint(stitched_text)
        if final_script_hint not in ALLOWED_SCRIPT_HINTS:
            final_script_hint = "unknown"

        if languages:
            from collections import Counter
            lc = Counter(languages)
            top_lang, _ = lc.most_common(1)[0]
            final_language = top_lang
        else:
            final_language = _fallback_detected_language(final_script_hint, stitched_text)

        final_confidence, all_warnings = _apply_ocr_confidence_caps(stitched_text, final_confidence, all_warnings)
        all_warnings = list(dict.fromkeys(all_warnings))

        return SaiaOCRResponse(
            status="FULL" if stitched_text.strip() else "FAIL",
            model_used=model,
            fallbacks=[],
            fallbacks_used=[],
            warnings=all_warnings,
            lines=all_lines,
            text=stitched_text,
            script_hint=final_script_hint,
            detected_language=final_language,
            confidence=final_confidence,
            raw_json={"column_split": True, "columns": 2},
        )

    def extract_tiles(
        self,
        *,
        image_b64: str,
        location_suggestions: Sequence[SaiaOCRLocationSuggestion],
        model: str,
        page_script_hint: str = "latin",
    ) -> SaiaOCRResponse | None:
        """Tile-based OCR: crop image by location suggestions, OCR each tile,
        stitch results in reading order. Returns None if tiles cannot be produced
        (e.g., no valid rects)."""
        if not location_suggestions:
            return None

        image = _open_rgb_image_from_b64(image_b64)
        img_w, img_h = image.size

        # 1. Normalize suggestions → rects
        rects = _suggestions_to_rects(location_suggestions, img_w, img_h)
        if not rects:
            _tile_logger.info("No valid rects from %d suggestions; skipping tile OCR.", len(location_suggestions))
            return None

        # 2. Morphological block merging (dilation + connected components)
        rects = _rects_to_block_rects(rects, img_w, img_h)
        _tile_logger.info("After block merge: %d blocks from %d suggestions.", len(rects), len(location_suggestions))

        # 3. Pad each tile
        rects = [_pad_rect(r, img_w, img_h) for r in rects]

        # 4. Reading order
        rects = _order_tiles_reading(rects, img_w)

        # 5. Crop and OCR each tile
        all_lines: list[str] = []
        all_warnings: list[str] = []
        languages: list[str] = []
        confidences: list[tuple[float, float]] = []  # (confidence, area_weight)
        tile_results: list[dict[str, Any]] = []

        for idx, rect in enumerate(rects):
            crop = _crop_and_maybe_upscale(image, rect)
            tile_b64 = _encode_png_base64(crop)
            tile_b64, tile_prep_warnings = _prepare_image_for_ocr(tile_b64)

            try:
                parsed = self._request_tile_json_from_model(
                    model=model,
                    tile_image_b64=tile_b64,
                    tile_index=idx,
                    total_tiles=len(rects),
                )
            except Exception as exc:
                all_warnings.append(f"tile_{idx}: OCR failed: {exc}")
                _tile_logger.warning("Tile %d OCR failed: %s", idx, exc)
                continue

            tile_lines = [str(line) for line in parsed.get("lines", []) if str(line).strip()]
            tile_text = str(parsed.get("text") or "").strip()
            tile_script = str(parsed.get("script_hint") or "unknown").lower()
            tile_lang = _normalize_detected_language(str(parsed.get("detected_language") or "unknown"))
            tile_conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
            tile_warnings = [str(w) for w in parsed.get("warnings", []) if str(w).strip()]

            tile_sanity = compute_sanity(tile_text) if tile_text else {
                "weird_ratio": 1.0,
                "uncertainty_marker_ratio": 0.0,
            }

            # --- Multi-view retry: if initial result is weak, try alt variant ---
            _mv_weak = (
                tile_text
                and tile_conf < 0.65
                and (
                    tile_sanity.get("weird_ratio", 0.0) >= 0.08
                    or tile_sanity.get("single_char_ratio", 0.0) >= 0.12
                )
            )
            if _mv_weak:
                try:
                    variants = generate_variants(crop)
                    alt = pick_retry_variant(variants, "enhanced_rgb")
                    if alt is not None:
                        alt_b64 = _encode_png_base64(alt.image)
                        alt_parsed = self._request_tile_json_from_model(
                            model=model,
                            tile_image_b64=alt_b64,
                            tile_index=idx,
                            total_tiles=len(rects),
                        )
                        alt_text = str(alt_parsed.get("text") or "").strip()
                        alt_conf = max(0.0, min(1.0, float(alt_parsed.get("confidence", 0.0) or 0.0)))
                        alt_sanity = compute_sanity(alt_text) if alt_text else {"weird_ratio": 1.0}
                        # Pick the variant with lower weird_ratio + higher conf
                        alt_score = alt_conf * (1.0 - alt_sanity.get("weird_ratio", 0.0))
                        orig_score = tile_conf * (1.0 - tile_sanity.get("weird_ratio", 0.0))
                        if alt_score > orig_score and alt_text:
                            _tile_logger.info(
                                "Tile %d: multiview %s beat enhanced_rgb (%.3f > %.3f)",
                                idx, alt.label, alt_score, orig_score,
                            )
                            parsed = alt_parsed
                            tile_lines = [str(ln) for ln in parsed.get("lines", []) if str(ln).strip()]
                            tile_text = alt_text
                            tile_script = str(parsed.get("script_hint") or "unknown").lower()
                            tile_lang = _normalize_detected_language(str(parsed.get("detected_language") or "unknown"))
                            tile_conf = alt_conf
                            tile_warnings = [str(w) for w in parsed.get("warnings", []) if str(w).strip()]
                            tile_sanity = alt_sanity
                            all_warnings.append(f"tile_{idx}: MULTIVIEW_RETRY:{alt.label}")
                except Exception as exc:
                    _tile_logger.debug("Tile %d multiview retry failed: %s", idx, exc)

            # --- Lexical trust adjustment ---
            if tile_text and tile_lang != "unknown":
                tile_conf, lex_warns = lexical_trust_adjustment(tile_conf, tile_text, tile_lang)
                tile_warnings.extend(lex_warns)

            # --- Non-Latin tile contamination check ---
            has_non_latin_warning = any(
                "non-latin" in str(w).lower() or "non_latin" in str(w).lower()
                for w in tile_warnings
            )
            script_mismatch = (
                page_script_hint == "latin"
                and tile_script not in ("latin", "unknown")
            )
            if tile_text and (
                script_mismatch or has_non_latin_warning
            ):
                # Try rerun with +10% padding
                padded = _pad_rect(rect, img_w, img_h, pad_ratio=_TILE_PAD_RATIO + 0.10)
                retry_ok = False
                if padded != rect:
                    try:
                        crop2 = _crop_and_maybe_upscale(image, padded)
                        tile_b64_2 = _encode_png_base64(crop2)
                        tile_b64_2, _ = _prepare_image_for_ocr(tile_b64_2)
                        parsed2 = self._request_tile_json_from_model(
                            model=model, tile_image_b64=tile_b64_2,
                            tile_index=idx, total_tiles=len(rects),
                        )
                        r2_script = str(parsed2.get("script_hint") or "unknown").lower()
                        r2_text = str(parsed2.get("text") or "").strip()
                        r2_warnings = [str(w) for w in parsed2.get("warnings", []) if str(w).strip()]
                        r2_sanity = compute_sanity(r2_text) if r2_text else {
                            "weird_ratio": 1.0,
                            "uncertainty_marker_ratio": 0.0,
                        }
                        r2_has_non_latin = any(
                            "non-latin" in w.lower() or "non_latin" in w.lower()
                            for w in r2_warnings
                        )
                        retry_bad = (
                            r2_has_non_latin
                            or (page_script_hint == "latin" and r2_script not in ("latin", "unknown"))
                            or r2_sanity.get("weird_ratio", 1.0) >= 0.15
                            or (
                                r2_sanity.get("uncertainty_marker_ratio", 0.0) == 0.0
                                and r2_sanity.get("weird_ratio", 0.0) > 0.05
                            )
                        )
                        if not retry_bad:
                            parsed = parsed2
                            tile_lines = [str(ln) for ln in parsed.get("lines", []) if str(ln).strip()]
                            tile_text = r2_text
                            tile_script = r2_script
                            tile_lang = _normalize_detected_language(str(parsed.get("detected_language") or "unknown"))
                            tile_conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
                            tile_warnings = r2_warnings
                            tile_sanity = r2_sanity
                            all_warnings.append(f"tile_{idx}: RETRIED_WITH_PADDING_OK")
                            retry_ok = True
                    except Exception:
                        pass
                if not retry_ok:
                    drop_reason = "non_latin_or_weird_after_retry"
                    all_warnings.append(f"DROPPED_TILE_{idx}:{drop_reason}")
                    _tile_logger.info("Dropped tile %d: %s", idx, drop_reason)
                    continue

            if tile_text and (
                tile_sanity.get("weird_ratio", 0.0) >= 0.18
                or (
                    tile_sanity.get("uncertainty_marker_ratio", 0.0) == 0.0
                    and tile_sanity.get("weird_ratio", 0.0) > 0.05
                )
            ):
                all_warnings.append(f"DROPPED_TILE_{idx}:sanity")
                _tile_logger.info("Dropped tile %d on sanity gate: %s", idx, tile_sanity)
                continue

            # Prefix tile-level warnings
            for w in tile_prep_warnings:
                all_warnings.append(f"tile_{idx}: {w}")
            for w in tile_warnings:
                all_warnings.append(f"tile_{idx}: {w}")

            if tile_lines:
                all_lines.extend(tile_lines)
            elif tile_text:
                all_lines.extend([ln.strip() for ln in tile_text.splitlines() if ln.strip()])

            if tile_lang and tile_lang != "unknown":
                languages.append(tile_lang)

            # Weight confidence by tile area
            x1, y1, x2, y2 = rect
            area = float((x2 - x1) * (y2 - y1))
            confidences.append((tile_conf, area))

            tile_results.append(parsed)
            _tile_logger.info(
                "Tile %d/%d: %d lines, lang=%s, conf=%.2f",
                idx + 1, len(rects), len(tile_lines), tile_lang, tile_conf,
            )

        if not all_lines:
            _tile_logger.info("Tile OCR produced no lines; returning None to trigger fallback.")
            return None

        # 6. Stitch results
        stitched_text = "\n".join(all_lines)
        stitched_lines = all_lines

        # Final detected_language: most frequent non-unknown
        if languages:
            from collections import Counter
            lang_counts = Counter(languages)
            most_common_lang, most_common_count = lang_counts.most_common(1)[0]
            if len(lang_counts) == 1:
                final_language = most_common_lang
            elif most_common_count > len(languages) // 2:
                final_language = most_common_lang
            else:
                # All tiles are Latin-script: prefer majority vote over "mixed"
                # Only return "mixed" when there's strong multi-script evidence
                if final_script_hint in ("latin", "unknown"):
                    final_language = most_common_lang
                else:
                    final_language = "mixed"
        else:
            final_language = _fallback_detected_language(
                _detect_script_hint(stitched_text), stitched_text,
            )

        # Final confidence: area-weighted average
        total_area = sum(area for _, area in confidences) or 1.0
        final_confidence = sum(conf * area for conf, area in confidences) / total_area
        final_confidence = max(0.0, min(1.0, final_confidence))

        # Script hint from stitched text
        final_script_hint = _detect_script_hint(stitched_text)
        if final_script_hint not in ALLOWED_SCRIPT_HINTS:
            final_script_hint = "unknown"

        # Apply confidence caps
        final_confidence, all_warnings = _apply_ocr_confidence_caps(stitched_text, final_confidence, all_warnings)

        all_warnings.insert(0, f"TILE_OCR:{len(rects)}_tiles")
        all_warnings = list(dict.fromkeys(all_warnings))

        status = "FULL" if stitched_text.strip() else "FAIL"
        if all_warnings:
            status = "PARTIAL" if stitched_text.strip() else "FAIL"

        return SaiaOCRResponse(
            status=status,
            model_used=model,
            fallbacks=[],
            fallbacks_used=[],
            warnings=all_warnings,
            lines=stitched_lines,
            text=stitched_text,
            script_hint=final_script_hint,
            detected_language=final_language,
            confidence=final_confidence,
            raw_json={"tile_results": tile_results, "tile_count": len(rects)},
        )

    def extract(self, payload: SaiaOCRRequest) -> SaiaOCRResponse:
        image_b64 = _extract_base64_payload(payload.image_b64 or "")
        image_b64, preprocess_warnings = _prepare_image_for_ocr(image_b64)
        available_models = self.client.list_models()
        candidate_models = self._select_candidate_models(available_models)
        if not candidate_models:
            raise SaiaOCRAgentError("No InternVL model is available on SAIA /models.")

        # --- Tile-based OCR when location_suggestions are present ---
        if payload.location_suggestions and len(payload.location_suggestions) >= 1:
            primary_model = candidate_models[0]
            _tile_logger.info(
                "Attempting tile-based OCR with %d location suggestions, model=%s",
                len(payload.location_suggestions),
                primary_model,
            )
            try:
                tile_result = self.extract_tiles(
                    image_b64=image_b64,
                    location_suggestions=payload.location_suggestions,
                    model=primary_model,
                    page_script_hint=str(payload.script_hint_seed or "latin").lower(),
                )
            except Exception as exc:
                _tile_logger.warning("Tile OCR failed, falling back to full-page: %s", exc)
                tile_result = None

            if tile_result is not None and tile_result.text.strip():
                # Quality gate: check if tile-based result is sane
                if _is_sane_ocr(tile_result.text):
                    _tile_logger.info("Tile OCR passed sanity check; using tile result.")
                    # Diplomatic sanitization
                    enforced_lines, enforced_text, enforce_warns = self._apply_diplomatic_enforcement(
                        list(tile_result.lines),
                        str(tile_result.script_hint or "unknown"),
                        str(tile_result.detected_language or "latin"),
                    )
                    combined_warnings = [*preprocess_warnings, *tile_result.warnings, *enforce_warns]
                    sanity = compute_sanity(enforced_text)
                    # Quality gate
                    if sanity["single_char_ratio"] > 0.10 or sanity["weird_ratio"] > 0.15:
                        enforced_lines, mask_count = quality_gate_enforce(enforced_lines)
                        if mask_count > 0:
                            combined_warnings.append(f"QUALITY_GATE_MASKED:{mask_count}_lines")
                            enforced_text = "\n".join(enforced_lines).strip()
                            sanity = compute_sanity(enforced_text)
                    qlabel = _quality_label_from_sanity(sanity)
                    tile_conf = sanity_adjust_confidence(
                        float(tile_result.confidence or 0.0), sanity,
                    )
                    _tile_logger.info("Tile quality_label=%s sanity=%s conf=%.3f", qlabel, sanity, tile_conf)
                    tile_result = tile_result.model_copy(
                        update={
                            "lines": enforced_lines,
                            "text": enforced_text,
                            "confidence": tile_conf,
                            "warnings": list(dict.fromkeys(combined_warnings)),
                            "raw_json": self._enrich_raw_json(tile_result.raw_json, sanity, qlabel),
                        }
                    )
                    return tile_result
                else:
                    _tile_logger.info("Tile OCR failed sanity check; falling back to full-page OCR.")
                    # Keep tile result as a candidate for comparison with full-page
                    tile_fallback = tile_result
            else:
                tile_fallback = tile_result if tile_result else None

            # Fall through to full-page OCR; will compare results later
        else:
            # --- Column-split fallback when no location suggestions ---
            tile_fallback = None
            try:
                col_result = self._try_column_split_ocr(
                    image_b64=image_b64,
                    model=candidate_models[0],
                )
            except Exception as exc:
                _tile_logger.warning("Column-split OCR failed: %s", exc)
                col_result = None
            if col_result is not None and col_result.text.strip() and _is_sane_ocr(col_result.text):
                _tile_logger.info("Column-split OCR passed sanity; using as tile_fallback candidate.")
                tile_fallback = col_result
        # --- End tile-based / column-split OCR block ---

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
            confidence, warnings = _apply_ocr_confidence_caps(text, confidence, warnings)
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

            # --- Lexical trust adjustment on full-page result ---
            if text and detected_language != "unknown":
                confidence, lex_warns = lexical_trust_adjustment(confidence, text, detected_language)
                warnings.extend(lex_warns)

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

            # --- Diplomatic sanitization (proofreading deferred to route) ---
            enforced_lines, enforced_text, enforce_warns = self._apply_diplomatic_enforcement(
                lines, script_hint, detected_language,
            )
            if enforce_warns:
                warnings.extend(enforce_warns)
            final_lines = enforced_lines
            final_text = enforced_text
            confidence, warnings = _apply_ocr_confidence_caps(final_text, confidence, warnings)

            # Quality gate enforcement
            final_sanity = compute_sanity(final_text)
            if final_sanity["single_char_ratio"] > 0.10 or final_sanity["weird_ratio"] > 0.15:
                final_lines, mask_count = quality_gate_enforce(final_lines)
                if mask_count > 0:
                    warnings.append(f"QUALITY_GATE_MASKED:{mask_count}_lines")
                    final_text = "\n".join(final_lines).strip()
                    final_sanity = compute_sanity(final_text)
            final_qlabel = _quality_label_from_sanity(final_sanity)
            confidence = sanity_adjust_confidence(confidence, final_sanity)
            _tile_logger.info("Full-page quality_label=%s sanity=%s conf=%.3f", final_qlabel, final_sanity, confidence)

            status = "FULL"
            if not final_text:
                status = "FAIL"
            elif warnings or fallbacks:
                status = "PARTIAL"

            fullpage_result = SaiaOCRResponse(
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
                raw_json=self._enrich_raw_json(parsed, final_sanity, final_qlabel),
            )

            # Compare with tile fallback if available
            if tile_fallback is not None and tile_fallback.text.strip():
                chosen = _pick_better_result(tile_fallback, fullpage_result)
                _tile_logger.info(
                    "Tile vs full-page comparison: chose %s",
                    "tile" if chosen is tile_fallback else "full-page",
                )
                return chosen
            return fullpage_result

        if best_partial is not None:
            # Compare best partial with tile fallback too
            if tile_fallback is not None and tile_fallback.text.strip():
                chosen = _pick_better_result(tile_fallback, best_partial)
                return chosen
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
