from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import json
import logging
import re
import time

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import settings
from app.schemas.agents_ocr import OCRDocumentMetadata


logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """Strict diplomatic transcription task.

Transcribe the manuscript text exactly as written.

- Preserve reading order exactly.
- Preserve line breaks exactly.
- Output one manuscript line per output line.
- Do not normalize spelling.
- Do not modernize language.
- Do not translate.
- Do not explain.
- Do not summarize.
- Do not invent missing text.
- Do not repeat lines.
- Preserve punctuation, abbreviations, unusual glyphs, and capitalization exactly as written.
- If a character or word is unclear, use [?] only for the unclear part.
- Output plain text only.

Return only the transcription.
"""

_DEFAULT_VARIANT_MAX_SIDES = (1536, 1280, 1024)
_DEFAULT_QUALITY_STEPS = (90, 84, 78, 72, 66, 60, 54)
_DEFAULT_SHRINK_FACTOR = 0.88
_DEFAULT_MAX_SHRINK_STEPS = 6
_DEFAULT_METADATA_HINT_MAX_LEN = 240
_DEFAULT_METADATA_NOTES_MAX_LEN = 500


class GlmOllamaOcrError(RuntimeError):
    """Raised when GLM OCR via Ollama cannot produce a usable transcription."""


@dataclass(frozen=True)
class PreparedOcrImage:
    name: str
    image_bytes: bytes
    width: int
    height: int
    size_bytes: int
    resized: bool
    compressed: bool
    applied_autocontrast: bool
    jpeg_quality: int


@dataclass(frozen=True)
class GlmOllamaOcrResult:
    text: str
    lines: list[str]
    model_used: str
    warnings: list[str]
    original_size_bytes: int
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    processed_size_bytes: int
    preprocessing_applied: bool
    processed_variant_name: str
    attempts_used: int
    duration_seconds: float


def _normalize_prompt_hint(value: str | None, *, max_length: int = _DEFAULT_METADATA_HINT_MAX_LEN) -> str | None:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
    if not cleaned:
        return None
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max(1, max_length - 3)].rstrip() + "..."


def build_glm_ocr_prompt(
    *,
    base_prompt: str | None = None,
    language_hint: str | None = None,
    script_hint_seed: str | None = None,
    metadata: OCRDocumentMetadata | None = None,
) -> str:
    selected_base_prompt = str(base_prompt or DEFAULT_PROMPT).strip()
    metadata = metadata or OCRDocumentMetadata()

    context_lines: list[str] = []
    seen_languages: set[str] = set()
    language_values: list[str] = []
    for raw_value in (
        _normalize_prompt_hint(metadata.language),
        _normalize_prompt_hint(language_hint),
    ):
        if raw_value is None:
            continue
        key = raw_value.casefold()
        if key in seen_languages:
            continue
        seen_languages.add(key)
        language_values.append(raw_value)
    if language_values:
        context_lines.append(f"- Likely manuscript language: {'; '.join(language_values)}")

    normalized_year = _normalize_prompt_hint(metadata.year)
    if normalized_year:
        context_lines.append(f"- Approximate manuscript date or year: {normalized_year}")

    normalized_script_seed = _normalize_prompt_hint(script_hint_seed)
    if normalized_script_seed:
        context_lines.append(f"- Script hint: {normalized_script_seed}")

    normalized_script_family = _normalize_prompt_hint(metadata.script_family)
    if normalized_script_family:
        context_lines.append(f"- Script family metadata: {normalized_script_family}")

    normalized_place = _normalize_prompt_hint(metadata.place_or_origin)
    if normalized_place:
        context_lines.append(f"- Place or origin: {normalized_place}")

    normalized_document_type = _normalize_prompt_hint(metadata.document_type)
    if normalized_document_type:
        context_lines.append(f"- Document type: {normalized_document_type}")

    normalized_notes = _normalize_prompt_hint(
        metadata.notes,
        max_length=_DEFAULT_METADATA_NOTES_MAX_LEN,
    )
    if normalized_notes:
        context_lines.append(f"- Additional manuscript notes: {normalized_notes}")

    if not context_lines:
        return selected_base_prompt

    prompt_sections = [
        selected_base_prompt,
        "",
        "Manuscript context hints:",
        "Use these hints only to disambiguate ambiguous characters, abbreviations, or script forms.",
        "Do not override clearly visible text with metadata assumptions.",
        "Do not normalize, translate, summarize, or invent text based on the hints.",
        *context_lines,
    ]
    return "\n".join(prompt_sections).strip() + "\n"


def open_image_safely(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = ImageOps.exif_transpose(img)

            if img.mode in ("P", "PA") or "transparency" in img.info:
                img = img.convert("RGBA")
            elif img.mode == "LA":
                img = img.convert("RGBA")
            elif img.mode == "L":
                img = img.convert("RGB")
            elif img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            if img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.getchannel("A"))
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            return img.copy()
    except UnidentifiedImageError as exc:
        raise GlmOllamaOcrError("Unreadable image payload.") from exc
    except OSError as exc:
        raise GlmOllamaOcrError(f"Could not open image payload: {exc}") from exc


def resize_to_max_side(img: Image.Image, max_side: int) -> Image.Image:
    width, height = img.size
    longest = max(width, height)
    if longest <= max_side:
        return img
    scale = max_side / longest
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def encode_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _encode_under_size_limit(
    img: Image.Image,
    *,
    name: str,
    max_bytes: int,
    applied_autocontrast: bool,
) -> PreparedOcrImage:
    base_width, base_height = img.size
    working = img
    shrink_steps = 0

    while True:
        for quality in _DEFAULT_QUALITY_STEPS:
            encoded = encode_jpeg_bytes(working, quality=quality)
            if len(encoded) < max_bytes:
                return PreparedOcrImage(
                    name=name,
                    image_bytes=encoded,
                    width=working.width,
                    height=working.height,
                    size_bytes=len(encoded),
                    resized=(working.width, working.height) != (base_width, base_height),
                    compressed=quality != _DEFAULT_QUALITY_STEPS[0],
                    applied_autocontrast=applied_autocontrast,
                    jpeg_quality=quality,
                )

        if shrink_steps >= _DEFAULT_MAX_SHRINK_STEPS:
            break

        target_width = max(1, int(round(working.width * _DEFAULT_SHRINK_FACTOR)))
        target_height = max(1, int(round(working.height * _DEFAULT_SHRINK_FACTOR)))
        if (target_width, target_height) == working.size:
            break
        working = working.resize((target_width, target_height), Image.Resampling.LANCZOS)
        shrink_steps += 1

    raise GlmOllamaOcrError(f"Could not compress OCR payload below {max_bytes} bytes.")


def preprocess_variants(image_bytes: bytes, *, max_payload_bytes: int) -> tuple[tuple[int, int], list[PreparedOcrImage]]:
    base = open_image_safely(image_bytes)
    original_size = (base.width, base.height)
    variants: list[PreparedOcrImage] = []

    for max_side in _DEFAULT_VARIANT_MAX_SIDES:
        resized = resize_to_max_side(base, max_side)
        variants.append(
            _encode_under_size_limit(
                resized,
                name=f"rgb_jpeg_{max_side}",
                max_bytes=max_payload_bytes,
                applied_autocontrast=False,
            )
        )

    auto = ImageOps.autocontrast(base)
    for max_side in _DEFAULT_VARIANT_MAX_SIDES:
        resized = resize_to_max_side(auto, max_side)
        variants.append(
            _encode_under_size_limit(
                resized,
                name=f"rgb_autocontrast_jpeg_{max_side}",
                max_bytes=max_payload_bytes,
                applied_autocontrast=True,
            )
        )

    return original_size, variants


def encode_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def extract_error_message(resp: requests.Response) -> str:
    try:
        payload = resp.json()
    except ValueError:
        text = resp.text.strip()
        return text if text else "(no response body)"

    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"]
    return json.dumps(payload, ensure_ascii=False)


def raise_for_ollama_status(resp: requests.Response, model: str) -> None:
    if resp.ok:
        return

    detail = extract_error_message(resp)
    lower_detail = detail.lower()

    if resp.status_code == 404 and "model" in lower_detail and "not found" in lower_detail:
        raise GlmOllamaOcrError(
            f"Ollama model '{model}' is not available. Pull it first with: ollama pull {model}"
        )

    raise GlmOllamaOcrError(f"Ollama HTTP {resp.status_code} at {resp.url}: {detail}")


def extract_text_from_generate(data: dict[str, object]) -> str:
    return str(data.get("response", "")).strip()


def extract_text_from_chat(data: dict[str, object]) -> str:
    message = data.get("message", {})
    if not isinstance(message, dict):
        return ""
    return str(message.get("content", "")).strip()


def ollama_extract(
    image_bytes: bytes,
    *,
    model: str,
    prompt: str,
    host: str,
    timeout: int,
    temperature: float,
) -> str:
    base_url = host.rstrip("/")
    image_b64 = encode_b64(image_bytes)

    chat_url = base_url + "/api/chat"
    chat_payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    try:
        response = requests.post(chat_url, json=chat_payload, timeout=timeout)
        if response.status_code != 404:
            raise_for_ollama_status(response, model=model)
            text = extract_text_from_chat(response.json())
            if text:
                return text
    except requests.RequestException as exc:
        raise GlmOllamaOcrError(f"Ollama request failed at {chat_url}: {exc}") from exc

    generate_url = base_url + "/api/generate"
    generate_payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    try:
        response = requests.post(generate_url, json=generate_payload, timeout=timeout)
        raise_for_ollama_status(response, model=model)
        text = extract_text_from_generate(response.json())
        if not text:
            raise GlmOllamaOcrError("Model returned empty text.")
        return text
    except requests.RequestException as exc:
        raise GlmOllamaOcrError(f"Ollama request failed at {generate_url}: {exc}") from exc


def is_backend_shape_error(exc: Exception) -> bool:
    value = str(exc).lower()
    triggers = [
        "ggml_assert",
        "an error was encountered while running the model",
        "http 500",
        "/api/generate",
        "/api/chat",
    ]
    return any(trigger in value for trigger in triggers)


def clean_model_output(text: str) -> str:
    if not text:
        return text

    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
    cleaned = re.sub(r"\n```$", "", cleaned)
    cleaned = re.sub(
        r"^\s*(Transcription|Extracted text|Output|Result)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    patterns = [
        r"Step\s*4\s*:\s*Return the extracted text\s*(.*)",
        r"Step\s*2\s*:\s*Extract the text\s*(.*)",
        r"The extracted text is\s*:\s*(.*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
            break

    cleaned = re.split(r"\n\s*Note\s*:", cleaned, flags=re.IGNORECASE)[0].strip()
    lines: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if re.match(r"^\*+\s*Step\s+\d+", stripped, flags=re.IGNORECASE):
            continue
        if re.match(r"^Step\s+\d+", stripped, flags=re.IGNORECASE):
            continue
        if re.match(
            r"^(The text appears|The extracted text is|Return the extracted text)",
            stripped,
            flags=re.IGNORECASE,
        ):
            continue
        lines.append(line.rstrip())

    return "\n".join(lines).strip()


def run_glm_ollama_ocr(
    image_bytes: bytes,
    *,
    image_ref: str,
    prompt: str | None = None,
    model: str | None = None,
    host: str | None = None,
    timeout: int | None = None,
    temperature: float | None = None,
    retries: int | None = None,
    max_payload_bytes: int | None = None,
) -> GlmOllamaOcrResult:
    started_at = time.perf_counter()
    selected_model = str(model or settings.glmocr_ollama_model or "glm-ocr:latest").strip() or "glm-ocr:latest"
    selected_host = str(host or settings.glmocr_ollama_host or "http://localhost:11434").strip() or "http://localhost:11434"
    selected_prompt = str(prompt or DEFAULT_PROMPT)
    selected_timeout = max(1, int(timeout if timeout is not None else settings.glmocr_ollama_timeout_seconds))
    selected_temperature = float(
        temperature if temperature is not None else settings.glmocr_ollama_temperature
    )
    selected_retries = max(1, int(retries if retries is not None else settings.glmocr_ollama_retries_per_variant))
    payload_limit = max(1, int(max_payload_bytes if max_payload_bytes is not None else settings.glmocr_max_payload_bytes))

    original_size, variants = preprocess_variants(image_bytes, max_payload_bytes=payload_limit)
    original_width, original_height = original_size
    logger.info(
        "OCR image selected ref=%s original_bytes=%s original_dimensions=%sx%s variants=%s model=%s",
        image_ref,
        len(image_bytes),
        original_width,
        original_height,
        len(variants),
        selected_model,
    )

    recent_errors: list[str] = []
    first_empty_variant: PreparedOcrImage | None = None
    first_empty_attempts_used: int | None = None

    for variant in variants:
        logger.info(
            "OCR variant ref=%s variant=%s processed_dimensions=%sx%s processed_bytes=%s resized=%s compressed=%s autocontrast=%s quality=%s",
            image_ref,
            variant.name,
            variant.width,
            variant.height,
            variant.size_bytes,
            variant.resized,
            variant.compressed,
            variant.applied_autocontrast,
            variant.jpeg_quality,
        )
        for attempt in range(1, selected_retries + 1):
            try:
                raw_text = ollama_extract(
                    image_bytes=variant.image_bytes,
                    model=selected_model,
                    prompt=selected_prompt,
                    host=selected_host,
                    timeout=selected_timeout,
                    temperature=selected_temperature,
                )
                cleaned_text = clean_model_output(raw_text)
                if not cleaned_text:
                    if first_empty_variant is None:
                        first_empty_variant = variant
                        first_empty_attempts_used = attempt
                    recent_errors.append(f"{variant.name} attempt {attempt}: empty text")
                    break

                duration = time.perf_counter() - started_at
                warnings: list[str] = []
                if variant.resized:
                    warnings.append("OCR_PAYLOAD_RESIZED")
                if variant.compressed:
                    warnings.append("OCR_PAYLOAD_COMPRESSED")
                if variant.applied_autocontrast:
                    warnings.append("OCR_PAYLOAD_AUTOCONTRAST")
                if variant.name != variants[0].name:
                    warnings.append(f"OCR_VARIANT_FALLBACK:{variant.name}")
                if attempt > 1:
                    warnings.append(f"OCR_RETRY_ATTEMPTS_USED:{attempt}")

                logger.info(
                    "OCR completed ref=%s variant=%s model=%s duration_seconds=%.3f text_chars=%s",
                    image_ref,
                    variant.name,
                    selected_model,
                    duration,
                    len(cleaned_text),
                )
                return GlmOllamaOcrResult(
                    text=cleaned_text,
                    lines=[line.rstrip() for line in cleaned_text.splitlines()],
                    model_used=selected_model,
                    warnings=warnings,
                    original_size_bytes=len(image_bytes),
                    original_width=original_width,
                    original_height=original_height,
                    processed_width=variant.width,
                    processed_height=variant.height,
                    processed_size_bytes=variant.size_bytes,
                    preprocessing_applied=(
                        variant.applied_autocontrast
                        or variant.resized
                        or variant.compressed
                        or (variant.width, variant.height) != (original_width, original_height)
                    ),
                    processed_variant_name=variant.name,
                    attempts_used=attempt,
                    duration_seconds=duration,
                )
            except Exception as exc:
                message = f"{variant.name} attempt {attempt}: {exc}"
                recent_errors.append(message)
                logger.warning("OCR variant failed ref=%s %s", image_ref, message)
                if "Pull it first with: ollama pull" in str(exc):
                    raise GlmOllamaOcrError(str(exc)) from exc
                if is_backend_shape_error(exc):
                    break

        if first_empty_variant is not None and first_empty_variant == variant:
            continue

    duration = time.perf_counter() - started_at
    if first_empty_variant is not None:
        warnings = ["EMPTY_TEXT"]
        if first_empty_variant.resized:
            warnings.append("OCR_PAYLOAD_RESIZED")
        if first_empty_variant.compressed:
            warnings.append("OCR_PAYLOAD_COMPRESSED")
        if first_empty_variant.applied_autocontrast:
            warnings.append("OCR_PAYLOAD_AUTOCONTRAST")
        if first_empty_variant.name != variants[0].name:
            warnings.append(f"OCR_VARIANT_FALLBACK:{first_empty_variant.name}")
        if first_empty_attempts_used and first_empty_attempts_used > 1:
            warnings.append(f"OCR_RETRY_ATTEMPTS_USED:{first_empty_attempts_used}")
        logger.warning(
            "OCR completed empty ref=%s variant=%s model=%s duration_seconds=%.3f",
            image_ref,
            first_empty_variant.name,
            selected_model,
            duration,
        )
        return GlmOllamaOcrResult(
            text="",
            lines=[],
            model_used=selected_model,
            warnings=warnings,
            original_size_bytes=len(image_bytes),
            original_width=original_width,
            original_height=original_height,
            processed_width=first_empty_variant.width,
            processed_height=first_empty_variant.height,
            processed_size_bytes=first_empty_variant.size_bytes,
            preprocessing_applied=(
                first_empty_variant.applied_autocontrast
                or first_empty_variant.resized
                or first_empty_variant.compressed
                or (first_empty_variant.width, first_empty_variant.height) != (original_width, original_height)
            ),
            processed_variant_name=first_empty_variant.name,
            attempts_used=max(1, int(first_empty_attempts_used or 1)),
            duration_seconds=duration,
        )

    last_error = recent_errors[-1] if recent_errors else "unknown error"
    joined_errors = "\n".join(recent_errors[-8:])
    raise GlmOllamaOcrError(
        f"{image_ref}: all OCR image variants failed.\nLast: {last_error}\n\nRecent errors:\n{joined_errors}"
    )
