from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import io
import logging
import re
import time
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageOps

from app.agents.base import BaseAgent
from app.agents.crop_agent import decode_image_bytes, encode_png_base64
from app.config import settings
from app.core.constants import FINAL_CLASSES
from app.services.glm_ollama_ocr import GlmOllamaOcrError, build_glm_ocr_prompt, run_glm_ollama_ocr
from app.services.model_router import get_task_model_assignments
from app.services.saia_client import SaiaClient, is_model_not_found_error

log = logging.getLogger(__name__)
_TRANSIENT_LABEL_RETRY_DELAYS = (0.75, 1.5)
_TOTAL_LABEL_ANALYSIS_DEADLINE_SECONDS = 30.0

_LABEL_GUIDANCE = {
    "Border": "Treat this as a framing or boundary device. Focus on ruling, framing, pigments, and whether it structures or decorates the page.",
    "Table": "Treat this as a ruled or tabular structure. Focus on layout, function, and whether it organizes information.",
    "Diagram": "Treat this as a schematic, geometric, or explanatory figure. Focus on form, layout purpose, and historical manuscript context.",
    "Main script black": "Treat this as the main textual hand. Focus on script style, ductus, page function, and visible layout cues; only read text if explicitly asked.",
    "Main script coloured": "Treat this as main text with rubrication or colored execution. Focus on scribal and decorative function.",
    "Variant script black": "Treat this as a secondary text hand or differentiated textual layer. Focus on how it differs from the main hand.",
    "Variant script coloured": "Treat this as a colored secondary textual layer. Focus on rubrication, emphasis, and function.",
    "Historiated": "Treat this as a historiated initial or narrative decorated letter. Focus on iconography, style, pigments, and manuscript function.",
    "Inhabited": "Treat this as an inhabited initial. Focus on figures, creatures, foliage, and stylistic context.",
    "Zoo - Anthropomorphic": "Treat this as a zoomorphic or anthropomorphic decorative initial or motif. Focus on the creature-human hybrid features and decorative role.",
    "Embellished": "Treat this as an embellished initial or decorated letterform. Focus on ornament, penwork, flourish, and likely stylistic tradition.",
    "Plain initial- coloured": "Treat this as a colored plain initial. Focus on color, hierarchy, and textual structuring role.",
    "Plain initial - Highlighted": "Treat this as a highlighted plain initial. Focus on emphasis and layout hierarchy more than iconography.",
    "Plain initial - Black": "Treat this as an uncolored plain initial. Focus on script hierarchy and layout function.",
    "Page Number": "Treat this as a foliation or page numbering mark. Focus on placement, numbering convention, and manuscript organization.",
    "Quire Mark": "Treat this as a quire signature or collation mark. Focus on codicological function rather than decoration.",
    "Running header": "Treat this as a running title or header. Focus on navigation, placement, and textual function.",
    "Catchword": "Treat this as a catchword at the foot of a page or quire. Focus on codicological purpose and placement.",
    "Gloss": "Treat this as gloss or marginal/interlinear annotation. Focus on relation to main text, placement, and hand.",
    "Illustrations": "Treat this as an illustration or figurative image. Focus on iconography, style, pigments, and likely manuscript function.",
    "Column": "Treat this as a structural text column or writing block. Focus on page architecture and layout function.",
    "GraphicZone": "Treat this as a non-text graphic area. Focus on whether it is decorative, diagrammatic, or illustrative.",
    "MusicLine": "Treat this as a musical line or staff-related feature. Focus on notation layout and liturgical/music manuscript context.",
    "MusicZone": "Treat this as a music block. Focus on notation area, staff layout, and function.",
    "Music": "Treat this as music notation or music-bearing content. Focus on visible notation style and manuscript use.",
}

_TEXTUAL_LABELS = {
    "Main script black",
    "Main script coloured",
    "Variant script black",
    "Variant script coloured",
    "Page Number",
    "Quire Mark",
    "Running header",
    "Catchword",
    "Gloss",
    "MusicLine",
    "Music",
}
_VISUAL_LABELS = {
    "Historiated",
    "Inhabited",
    "Zoo - Anthropomorphic",
    "Embellished",
    "Plain initial- coloured",
    "Plain initial - Highlighted",
    "Plain initial - Black",
    "Illustrations",
    "GraphicZone",
    "Diagram",
}
_STRUCTURAL_LABELS = set(FINAL_CLASSES) - _TEXTUAL_LABELS - _VISUAL_LABELS
_TEXT_REQUEST_RE = re.compile(r"\b(what does it say|read|transcribe|extract text|page number|number)\b", re.IGNORECASE)
_INITIAL_LABELS = {
    "Historiated",
    "Inhabited",
    "Zoo - Anthropomorphic",
    "Embellished",
    "Plain initial- coloured",
    "Plain initial - Highlighted",
    "Plain initial - Black",
}
_INITIAL_LETTER_REQUEST_RE = re.compile(
    r"\b(what letter|which letter|what\s+is\b[^?.!\n]{0,60}\bletter\b|identify (the )?letter|identify (the )?initial|which initial|initial letter)\b",
    re.IGNORECASE,
)
_PLURAL_INITIAL_REQUEST_RE = re.compile(
    r"\b(?:what\s+(?:is|are)|which|identify|list|name)\b[^?.!\n]{0,80}\b(?:letters|initials)\b",
    re.IGNORECASE,
)
_INITIAL_LETTER_VALUE_RE = re.compile(r"(?im)^\s*(?:letter\s*:\s*)?(UNKNOWN|[A-Z])\s*$")
_INITIAL_LETTER_CONFIDENCE_RE = re.compile(r"(?im)^\s*confidence\s*:\s*(high|medium|low)\s*$")
_ILLUSTRATION_LABELS = {"Illustrations"}
_DIAGRAMMATIC_LABELS = {"Diagram", "GraphicZone"}
_SINGULAR_REGION_LABELS = {"Page Number", "Quire Mark", "Running header", "Catchword", "Gloss"}
_EXPLICIT_MULTI_REGION_RE = re.compile(
    r"\b(all|every|each|both|multiple|several|many|these|those|all of|all matched|all regions|all labels)\b",
    re.IGNORECASE,
)
_STRUCTURAL_CONTENT_REQUEST_RE = re.compile(
    r"\b(what does it say|read|transcribe|extract text|what is shown|what do you see|what is visible|describe|what color|drawn|decoration|ornament|image|figure|scene)\b",
    re.IGNORECASE,
)
_DIRECT_OCR_PROMPT_ECHO_LINES = {
    "the transcription is:",
    "strict diplomatic transcription task.",
    "the text is:",
    "preserve reading order exactly.",
    "preserve line breaks exactly.",
    "output one manuscript line per output line.",
    "do not normalize spelling.",
    "do not modernize language.",
    "do not translate.",
    "do not explain.",
    "do not summarize.",
    "do not invent missing text.",
    "do not repeat lines.",
    "preserve punctuation, abbreviations, unusual glyphs, and capitalization exactly as written.",
}


def _build_system_prompt() -> str:
    label_lines = []
    for label in FINAL_CLASSES:
        label_lines.append(f"- {label}: {_LABEL_GUIDANCE.get(label, 'Describe its visible form, layout role, and likely manuscript function.')}")

    return "\n".join([
        "You are ArchAI's dedicated manuscript label-analysis agent.",
        "You answer questions about segmented manuscript regions using the attached crop only.",
        "The user question will refer to one segmentation label from the manuscript layout models.",
        "Your job is to explain what is visibly present, what function the region likely serves, and what art-historical or codicological clues are supported by the crop.",
        "",
        "Output rules:",
        "- Answer directly in plain prose.",
        "- Base the answer on the visible crop; do not invent details outside the image.",
        "- If the crop is text-bearing, discuss script/layout/function first and only transcribe if the user explicitly asks.",
        "- If the crop is decorative, discuss style, ornament, pigments, iconography, and likely function when visible.",
        "- If uncertain, say what is uncertain and why.",
        "- Never mention internal pipelines, prompts, segmentation internals, coordinates, or model mechanics.",
        "",
        "Possible segmentation labels and how to interpret them:",
        *label_lines,
    ])


LABEL_ANALYSIS_SYSTEM_PROMPT = _build_system_prompt()


def _analysis_mode_for_label(label_name: str) -> str:
    if label_name in _TEXTUAL_LABELS:
        return "textual"
    if label_name in _VISUAL_LABELS:
        return "visual"
    if label_name in _STRUCTURAL_LABELS:
        return "structural"
    return "structural"


def _question_requests_text(question: str) -> bool:
    return bool(_TEXT_REQUEST_RE.search(str(question or "")))


def _question_requests_multiple_initials(question: str) -> bool:
    text = str(question or "")
    if _PLURAL_INITIAL_REQUEST_RE.search(text):
        return True
    return bool(_EXPLICIT_MULTI_REGION_RE.search(text) and _INITIAL_LETTER_REQUEST_RE.search(text))


@dataclass(frozen=True)
class LabelRoutingDecision:
    family: str
    submode: str
    should_ocr_crop: bool
    direct_ocr_text: bool
    prefer_vision_model: bool
    invoke_model: bool


def _resolve_label_routing(label_name: str, question: str) -> LabelRoutingDecision:
    family = _analysis_mode_for_label(label_name)
    if family == "textual":
        requests_text = _question_requests_text(question)
        return LabelRoutingDecision(
            family="textual",
            submode="textual_reading" if requests_text else "textual_explanation",
            should_ocr_crop=True,
            direct_ocr_text=requests_text,
            prefer_vision_model=False,
            invoke_model=not requests_text,
        )

    if family == "visual":
        if label_name in _INITIAL_LABELS and _question_requests_multiple_initials(question):
            return LabelRoutingDecision(
                family="visual",
                submode="multi_initial_letter_identification",
                should_ocr_crop=False,
                direct_ocr_text=False,
                prefer_vision_model=True,
                invoke_model=True,
            )
        if label_name in _INITIAL_LABELS and _INITIAL_LETTER_REQUEST_RE.search(question or ""):
            return LabelRoutingDecision(
                family="visual",
                submode="initial_letter_identification",
                should_ocr_crop=False,
                direct_ocr_text=False,
                prefer_vision_model=True,
                invoke_model=True,
            )
        if label_name in _INITIAL_LABELS:
            return LabelRoutingDecision(
                family="visual",
                submode="decorated_initial_analysis",
                should_ocr_crop=False,
                direct_ocr_text=False,
                prefer_vision_model=True,
                invoke_model=True,
            )
        if label_name in _ILLUSTRATION_LABELS:
            return LabelRoutingDecision(
                family="visual",
                submode="illustration_analysis",
                should_ocr_crop=False,
                direct_ocr_text=False,
                prefer_vision_model=True,
                invoke_model=True,
            )
        if label_name in _DIAGRAMMATIC_LABELS:
            return LabelRoutingDecision(
                family="visual",
                submode="diagrammatic_analysis",
                should_ocr_crop=False,
                direct_ocr_text=False,
                prefer_vision_model=True,
                invoke_model=True,
            )
        return LabelRoutingDecision(
            family="visual",
            submode="visual_analysis",
            should_ocr_crop=False,
            direct_ocr_text=False,
            prefer_vision_model=True,
            invoke_model=True,
        )

    requests_text = _question_requests_text(question)
    requests_structural_content = bool(_STRUCTURAL_CONTENT_REQUEST_RE.search(question or ""))
    return LabelRoutingDecision(
        family="structural",
        submode="structural_text_assist" if requests_text else "structural_analysis",
        should_ocr_crop=requests_text,
        direct_ocr_text=requests_text,
        prefer_vision_model=requests_structural_content and not requests_text,
        invoke_model=requests_structural_content and not requests_text,
    )


def _build_mode_prompt(
    *,
    label_name: str,
    analysis_mode: str,
    analysis_submode: str,
    question: str,
    region_count: int,
    filename: str,
    ocr_text: str,
) -> str:
    mode_lines = [
        f"Requested segmentation label: {label_name}",
        f"Analysis mode: {analysis_mode}",
        f"Analysis submode: {analysis_submode}",
        f"Matched regions in crop: {region_count}",
        *(["Page filename: " + filename] if filename else []),
        f"Label-specific guidance: {_LABEL_GUIDANCE.get(label_name, 'Describe its visible form, layout role, and likely manuscript function.')}",
        "",
    ]
    if analysis_mode == "textual":
        if analysis_submode == "textual_reading":
            mode_lines.extend(
                [
                    "This is a textual label reading task.",
                    "Treat OCR text as the primary evidence.",
                    "If OCR text is available, answer with the OCR text directly.",
                    "Do not describe the page, crop, or decoration unless the user explicitly asks for that.",
                    "",
                ]
            )
        else:
            mode_lines.extend(
                [
                    "This is a textual label explanation task.",
                    "Treat OCR text as the primary evidence and visible placement as secondary evidence.",
                    "Explain what the label is doing on the page before offering broader interpretation.",
                    "",
                ]
            )
        mode_lines.extend(
            [
                "OCR text for the crop:",
                ocr_text or "(no confident OCR text available)",
                "",
            ]
        )
    elif analysis_mode == "visual":
        if analysis_submode == "initial_letter_identification":
            mode_lines.extend(
                [
                    "This is an initial-letter identification task.",
                    "Identify one letter only if the crop supports it visually.",
                    "Do not describe ornament, iconography, or decoration before deciding whether the letterform itself is identifiable.",
                    "If the letter cannot be identified reliably, return UNKNOWN.",
                    "Return either:",
                    "Letter: <single uppercase Latin letter or UNKNOWN>",
                    "You may add a second line only if confidence is clearly supportable:",
                    "Confidence: <high|medium|low>",
                    "",
                ]
            )
        elif analysis_submode == "multi_initial_letter_identification":
            mode_lines.extend(
                [
                    "This is a multi-region initial-letter identification task.",
                    "You are analysing one initial region at a time in reading order.",
                    "For the current region only, identify one uppercase Latin letter if the letterform is visibly supportable.",
                    "If the current region is too uncertain, return UNKNOWN.",
                    "Do not describe decoration, color, or ornament in this mode.",
                    "Return either:",
                    "Letter: <single uppercase Latin letter or UNKNOWN>",
                    "You may add a second line only if confidence is clearly supportable:",
                    "Confidence: <high|medium|low>",
                    "",
                ]
            )
        elif analysis_submode == "decorated_initial_analysis":
            mode_lines.extend(
                [
                    "This is a decorated-initial analysis task.",
                    "Start from the letterform and the initial's textual hierarchy or decorative function.",
                    "Only then discuss ornament, penwork, pigments, figures, or flourish if they are visibly supported.",
                    "Do not treat this crop like a standalone illustration unless it clearly contains a scene or figure.",
                    "",
                ]
            )
        elif analysis_submode == "illustration_analysis":
            mode_lines.extend(
                [
                    "This is an illustration analysis task.",
                    "Describe depicted figures, objects, gestures, setting, and visible composition before broader interpretation.",
                    "Do not answer as though this were a decorated initial unless the crop is clearly letter-based.",
                    "",
                ]
            )
        elif analysis_submode == "diagrammatic_analysis":
            mode_lines.extend(
                [
                    "This is a diagrammatic or graphic analysis task.",
                    "Describe structure, geometry, lines, labels, and explanatory function before interpretation.",
                    "",
                ]
            )
        else:
            mode_lines.extend(
                [
                    "Treat visual form and iconography as the main evidence.",
                    "Answer from the crop only, with uncertainty stated plainly.",
                    "",
                ]
            )
    else:
        mode_lines.extend(
            [
                "This is a structural/layout analysis task.",
                "Treat codicological role, page architecture, and spatial structure as the main evidence.",
                "Answer from layout function first, not from decorative speculation.",
                "Only use OCR text if the user explicitly asks about textual content in the region.",
                *(["OCR text for the crop:", ocr_text or "(no confident OCR text available)", ""] if ocr_text else []),
            ]
        )

    mode_lines.extend(["User question:", question])
    return "\n".join(mode_lines)


@dataclass
class LabelAnalysisResult:
    status: str
    text: str
    label_name: str
    analysis_mode: str
    model_used: str
    warnings: list[str]
    region_count: int
    crop_image_b64: str
    crop_bounds_xyxy: list[int]
    ocr_text: str = ""
    stage_metadata: dict[str, Any] | None = None
    inspection: dict[str, Any] | None = None


class LabelAnalysisAgentError(RuntimeError):
    """Raised when label analysis cannot be completed."""


@dataclass(frozen=True)
class SelectedRegions:
    regions: list[Any]
    strategy: str
    input_count: int
    used_count: int


@dataclass(frozen=True)
class PreparedLabelPayload:
    image_b64: str
    image_bytes: bytes
    width: int
    height: int
    image_format: str


class LabelAnalysisAgent(BaseAgent):
    name = "label-analysis-agent"

    def __init__(self, client: SaiaClient | None = None) -> None:
        self.client = client or SaiaClient()

    def _visual_model(self) -> str:
        assignments = get_task_model_assignments()
        return str(
            settings.label_visual_model
            or settings.saia_label_analysis_model
            or assignments.label_visual_model
        ).strip() or assignments.label_visual_model

    def _text_model(self) -> str:
        assignments = get_task_model_assignments()
        return assignments.chat_rag_model

    @staticmethod
    def _is_vision_model(model_id: str) -> bool:
        key = str(model_id or "").lower()
        return any(token in key for token in ("vl", "vision", "internvl"))

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        text = str(exc or "").lower()
        return any(
            token in text
            for token in (
                "502",
                "503",
                "504",
                "proxy error",
                "bad gateway",
                "gateway timeout",
                "error reading from remote server",
                "upstream server",
                "temporarily unavailable",
                "timed out",
                "timeout",
            )
        )

    @staticmethod
    def _friendly_error(exc: Exception) -> str:
        if LabelAnalysisAgent._is_transient_error(exc):
            return "Upstream visual-analysis provider timed out or returned a transient proxy error. Please retry."
        return str(exc)

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        text = str(exc or "").lower()
        return "timed out" in text or "timeout" in text

    @staticmethod
    def _max_tokens_for_submode(analysis_submode: str) -> int:
        if analysis_submode == "initial_letter_identification":
            return 32
        if analysis_submode == "multi_initial_letter_identification":
            return 24
        if analysis_submode == "textual_explanation":
            return 420
        if analysis_submode == "textual_reading":
            return 160
        if analysis_submode == "structural_analysis":
            return 260
        return 520

    @staticmethod
    def _timeout_seconds_for_submode(analysis_submode: str) -> float:
        if analysis_submode == "initial_letter_identification":
            return 12.0
        if analysis_submode == "multi_initial_letter_identification":
            return 8.0
        if analysis_submode == "textual_reading":
            return 20.0
        if analysis_submode == "textual_explanation":
            return 18.0
        if analysis_submode in {"decorated_initial_analysis", "illustration_analysis", "diagrammatic_analysis", "visual_analysis"}:
            return 22.0
        if analysis_submode == "structural_text_assist":
            return 15.0
        return 12.0

    @staticmethod
    def _question_requests_multiple_regions(question: str) -> bool:
        return bool(_EXPLICIT_MULTI_REGION_RE.search(str(question or "")))

    @staticmethod
    def _region_area(region: Any) -> int:
        bbox = LabelAnalysisAgent._extract_bbox(region)
        if bbox is None:
            return 0
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _bbox_sort_tuple(region: Any) -> tuple[int, int, int, int]:
        bbox = LabelAnalysisAgent._extract_bbox(region)
        if bbox is None:
            return (10**9, 10**9, 10**9, 10**9)
        x1, y1, x2, y2 = bbox
        return (y1, x1, x2, y2)

    @staticmethod
    def _select_regions_for_query(
        *,
        label_name: str,
        question: str,
        routing: LabelRoutingDecision,
        regions: Sequence[Any],
    ) -> SelectedRegions:
        valid_regions = [region for region in regions if LabelAnalysisAgent._extract_bbox(region) is not None]
        input_count = len(list(regions))
        if not valid_regions:
            return SelectedRegions(regions=[], strategy="no_valid_regions", input_count=input_count, used_count=0)
        if len(valid_regions) == 1:
            return SelectedRegions(regions=valid_regions, strategy="single_region_only", input_count=input_count, used_count=1)

        if routing.submode == "multi_initial_letter_identification":
            ordered = sorted(valid_regions, key=LabelAnalysisAgent._bbox_sort_tuple)
            return SelectedRegions(
                regions=ordered,
                strategy="all_initial_regions_reading_order",
                input_count=input_count,
                used_count=len(ordered),
            )

        if routing.submode == "initial_letter_identification":
            if len(valid_regions) > 1:
                ordered = sorted(valid_regions, key=LabelAnalysisAgent._bbox_sort_tuple)
                return SelectedRegions(
                    regions=ordered,
                    strategy="ambiguous_multiple_initial_regions",
                    input_count=input_count,
                    used_count=len(ordered),
                )
            best = max(valid_regions, key=lambda region: (LabelAnalysisAgent._region_area(region), -LabelAnalysisAgent._bbox_sort_tuple(region)[0]))
            return SelectedRegions(regions=[best], strategy="largest_single_initial_region", input_count=input_count, used_count=1)

        if LabelAnalysisAgent._question_requests_multiple_regions(question):
            return SelectedRegions(
                regions=valid_regions,
                strategy="all_regions_explicit",
                input_count=input_count,
                used_count=len(valid_regions),
            )

        if label_name in {"Page Number", "Quire Mark", "Running header"}:
            best = min(valid_regions, key=lambda region: (LabelAnalysisAgent._bbox_sort_tuple(region)[0], LabelAnalysisAgent._region_area(region), LabelAnalysisAgent._bbox_sort_tuple(region)[1]))
            return SelectedRegions(regions=[best], strategy="top_most_smallest_region", input_count=input_count, used_count=1)

        if label_name == "Catchword":
            best = max(valid_regions, key=lambda region: (LabelAnalysisAgent._bbox_sort_tuple(region)[3], -LabelAnalysisAgent._region_area(region)))
            return SelectedRegions(regions=[best], strategy="bottom_most_single_region", input_count=input_count, used_count=1)

        if label_name in _INITIAL_LABELS:
            best = max(valid_regions, key=lambda region: (LabelAnalysisAgent._region_area(region), -LabelAnalysisAgent._bbox_sort_tuple(region)[0]))
            return SelectedRegions(regions=[best], strategy="largest_decorative_region", input_count=input_count, used_count=1)

        if label_name in _ILLUSTRATION_LABELS:
            best = max(valid_regions, key=lambda region: (LabelAnalysisAgent._region_area(region), -LabelAnalysisAgent._bbox_sort_tuple(region)[0]))
            return SelectedRegions(regions=[best], strategy="largest_illustration_region", input_count=input_count, used_count=1)

        if label_name == "Gloss":
            best = min(valid_regions, key=lambda region: (LabelAnalysisAgent._bbox_sort_tuple(region)[0], LabelAnalysisAgent._bbox_sort_tuple(region)[1], LabelAnalysisAgent._region_area(region)))
            return SelectedRegions(regions=[best], strategy="first_single_gloss", input_count=input_count, used_count=1)

        if label_name in _TEXTUAL_LABELS:
            best = max(valid_regions, key=lambda region: (LabelAnalysisAgent._region_area(region), -LabelAnalysisAgent._bbox_sort_tuple(region)[0]))
            return SelectedRegions(regions=[best], strategy="largest_single_text_region", input_count=input_count, used_count=1)

        best = max(valid_regions, key=lambda region: (LabelAnalysisAgent._region_area(region), -LabelAnalysisAgent._bbox_sort_tuple(region)[0]))
        return SelectedRegions(regions=[best], strategy="largest_single_region", input_count=input_count, used_count=1)

    @staticmethod
    def _encode_png_bytes(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _prepare_label_crop_for_model(
        crop_image: Image.Image,
        *,
        label_name: str,
        analysis_submode: str,
    ) -> PreparedLabelPayload:
        if analysis_submode in {"initial_letter_identification", "multi_initial_letter_identification"} or label_name in {"Page Number", "Quire Mark"}:
            max_side = 768
            max_bytes = 400_000
        elif analysis_submode in {"illustration_analysis", "decorated_initial_analysis"} or label_name in _INITIAL_LABELS or label_name in _ILLUSTRATION_LABELS:
            max_side = 1024
            max_bytes = 900_000
        else:
            max_side = 1280
            max_bytes = 1_200_000

        image = crop_image.convert("RGB")
        width, height = image.size
        largest_side = max(width, height)
        if largest_side > max_side:
            scale = max_side / float(largest_side)
            image = image.resize(
                (
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                ),
                Image.LANCZOS,
            )

        qualities = [88, 80, 72, 64, 56, 48]
        best_bytes = b""
        best_size = image.size
        working = image
        while True:
            for quality in qualities:
                buffer = io.BytesIO()
                working.save(buffer, format="JPEG", quality=quality, optimize=True)
                data = buffer.getvalue()
                best_bytes = data
                best_size = working.size
                if len(data) <= max_bytes:
                    return PreparedLabelPayload(
                        image_b64=base64.b64encode(data).decode("utf-8"),
                        image_bytes=data,
                        width=best_size[0],
                        height=best_size[1],
                        image_format="jpeg",
                    )
            if max(working.size) <= 256:
                break
            working = working.resize(
                (
                    max(1, int(round(working.size[0] * 0.88))),
                    max(1, int(round(working.size[1] * 0.88))),
                ),
                Image.LANCZOS,
            )

        if best_bytes and len(best_bytes) <= max_bytes:
            return PreparedLabelPayload(
                image_b64=base64.b64encode(best_bytes).decode("utf-8"),
                image_bytes=best_bytes,
                width=best_size[0],
                height=best_size[1],
                image_format="jpeg",
            )
        raise LabelAnalysisAgentError("Prepared label crop exceeded the size limit for model analysis.")

    @staticmethod
    def _remaining_timeout_seconds(deadline: float, preferred_timeout_seconds: float) -> float:
        remaining = deadline - time.perf_counter()
        allowed = min(float(preferred_timeout_seconds), remaining)
        if allowed <= 0.5:
            raise LabelAnalysisAgentError("Label analysis exceeded its total deadline.")
        return round(allowed, 2)

    @staticmethod
    def _run_glm_ocr_for_crop(
        crop_image: Image.Image,
        *,
        label_name: str,
        timeout_seconds: float,
    ) -> tuple[Any, int, str]:
        crop_bytes = LabelAnalysisAgent._encode_png_bytes(crop_image)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                run_glm_ollama_ocr,
                crop_bytes,
                image_ref=f"label:{label_name}",
                prompt=build_glm_ocr_prompt(),
            )
            try:
                result = future.result(timeout=timeout_seconds)
            except FutureTimeoutError as exc:
                future.cancel()
                raise LabelAnalysisAgentError("Label OCR timed out.") from exc
        return result, len(crop_bytes), "png"

    @staticmethod
    def _sanitize_direct_ocr_text(label_name: str, raw_text: str) -> str:
        lines = []
        for raw_line in str(raw_text or "").replace("\r", "\n").split("\n"):
            line = str(raw_line or "").strip()
            if not line:
                continue
            lower = line.lower()
            if lower in _DIRECT_OCR_PROMPT_ECHO_LINES:
                continue
            if lower.startswith("if a character or word is unclear"):
                continue
            lines.append(line)
        cleaned = "\n".join(lines).strip()
        if label_name in {"Page Number", "Quire Mark", "Running header", "Catchword"}:
            short_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            compact = " ".join(short_lines).strip()
            if compact and len(short_lines) <= 2 and len(compact) <= 24:
                return compact
            candidates = re.findall(r"\b(?:[0-9]{1,4}|[IVXLCDM]{1,8}|[A-Za-z]{1,4}[0-9]{0,3})\b", compact)
            if candidates:
                return candidates[0]
        return cleaned

    @staticmethod
    def _structural_metadata_answer(*, label_name: str, question: str, region_count: int, ocr_text: str) -> str:
        if _question_requests_text(question):
            if ocr_text:
                return ocr_text
            return f'No confident OCR text could be extracted from the selected "{label_name}" region.'
        label_hint = _LABEL_GUIDANCE.get(label_name, f'Treat this as the structural manuscript label "{label_name}".')
        count_suffix = f" This answer refers to {region_count} matched region(s)." if region_count > 1 else ""
        return (
            f'This region is best understood as a structural "{label_name}" element. '
            f"{label_hint} Its primary value here is codicological and layout-related rather than pictorial."
            f"{count_suffix}"
        )

    def _choose_retry_model(self, requested_model: str, *, prefer_vision: bool) -> str | None:
        try:
            available = self.client.list_models()
        except Exception:
            available = []
        if not available:
            return None
        assignments = get_task_model_assignments()
        preferred = []
        if prefer_vision:
            preferred.extend([assignments.label_visual_model, assignments.label_visual_fallback_model])
        else:
            preferred.extend([assignments.chat_rag_model, assignments.label_visual_model])
        for candidate in preferred:
            if candidate and candidate != requested_model and candidate in available:
                return candidate
        if prefer_vision:
            for model_id in available:
                if model_id != requested_model and self._is_vision_model(model_id):
                    return model_id
        for model_id in available:
            if model_id != requested_model:
                return model_id
        return None

    @staticmethod
    def _allow_fallback_after_runtime_error(*, analysis_submode: str, prefer_vision: bool, fallback_used: bool) -> bool:
        return (
            prefer_vision
            and not fallback_used
            and analysis_submode in {"initial_letter_identification", "multi_initial_letter_identification"}
        )

    @staticmethod
    def _stage_metadata(
        *,
        model_used: str,
        mode_used: str,
        duration_ms: float,
        region_count_input: int,
        region_count_used: int,
        selected_region_strategy: str,
        crop_width: int,
        crop_height: int,
        crop_bytes: int,
        image_format_sent: str,
        timeout_seconds: float,
        retry_count: int,
    ) -> dict[str, Any]:
        return {
            "stage_name": "label_analysis",
            "model_used": model_used,
            "mode_used": mode_used,
            "duration_ms": round(float(duration_ms), 1),
            "region_count_input": int(region_count_input),
            "region_count_used": int(region_count_used),
            "selected_region_strategy": selected_region_strategy,
            "crop_width": int(crop_width),
            "crop_height": int(crop_height),
            "crop_bytes": int(crop_bytes),
            "image_format_sent": image_format_sent,
            "timeout_seconds": round(float(timeout_seconds), 2),
            "retry_count": int(retry_count),
        }

    @staticmethod
    def _normalize_initial_letter_output(text: str) -> tuple[str, str | None]:
        raw = str(text or "").strip()
        letter_match = _INITIAL_LETTER_VALUE_RE.search(raw)
        confidence_match = _INITIAL_LETTER_CONFIDENCE_RE.search(raw)
        letter = letter_match.group(1).upper() if letter_match else "UNKNOWN"
        confidence = confidence_match.group(1).lower() if confidence_match else None
        return letter, confidence

    def _run_saia_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        selected_model: str,
        prefer_vision: bool,
        analysis_submode: str,
        overall_deadline: float,
    ) -> tuple[dict[str, Any], str, list[str], int, float]:
        max_tokens = self._max_tokens_for_submode(analysis_submode)
        base_timeout_seconds = self._timeout_seconds_for_submode(analysis_submode)
        deterministic = analysis_submode in {"initial_letter_identification", "multi_initial_letter_identification"}
        transient_attempt = 0
        retry_count = 0
        fallback_used = False
        warnings: list[str] = []

        while True:
            try:
                timeout_used_seconds = self._remaining_timeout_seconds(overall_deadline, base_timeout_seconds)
                response = self.client.chat_completion(
                    model=selected_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=max_tokens,
                    deterministic=deterministic,
                    timeout_seconds=timeout_used_seconds,
                )
                return response, selected_model, warnings, retry_count, timeout_used_seconds
            except Exception as exc:
                if is_model_not_found_error(exc):
                    if fallback_used:
                        raise LabelAnalysisAgentError(
                            f"SAIA label analysis failed: {self._friendly_error(exc)}"
                        ) from exc
                    retry_model = self._choose_retry_model(selected_model, prefer_vision=prefer_vision)
                    if not retry_model:
                        raise LabelAnalysisAgentError(
                            f"SAIA label analysis failed: {self._friendly_error(exc)}"
                        ) from exc
                    selected_model = retry_model
                    fallback_used = True
                    retry_count += 1
                    warnings.append(f"LABEL_MODEL_FALLBACK:{retry_model}")
                    continue

                if self._is_timeout_error(exc):
                    if self._allow_fallback_after_runtime_error(
                        analysis_submode=analysis_submode,
                        prefer_vision=prefer_vision,
                        fallback_used=fallback_used,
                    ):
                        retry_model = self._choose_retry_model(selected_model, prefer_vision=prefer_vision)
                        if retry_model:
                            selected_model = retry_model
                            fallback_used = True
                            retry_count += 1
                            warnings.append(f"LABEL_TIMEOUT_FALLBACK:{retry_model}")
                            continue
                    raise LabelAnalysisAgentError(
                        f"SAIA label analysis failed: {self._friendly_error(exc)}"
                    ) from exc

                if self._is_transient_error(exc) and transient_attempt < 1:
                    delay = _TRANSIENT_LABEL_RETRY_DELAYS[transient_attempt]
                    transient_attempt += 1
                    retry_count += 1
                    if time.perf_counter() + delay >= overall_deadline:
                        raise LabelAnalysisAgentError(
                            f"SAIA label analysis failed: {self._friendly_error(exc)}"
                        ) from exc
                    time.sleep(delay)
                    continue
                if self._is_transient_error(exc) and self._allow_fallback_after_runtime_error(
                    analysis_submode=analysis_submode,
                    prefer_vision=prefer_vision,
                    fallback_used=fallback_used,
                ):
                    retry_model = self._choose_retry_model(selected_model, prefer_vision=prefer_vision)
                    if retry_model:
                        selected_model = retry_model
                        fallback_used = True
                        retry_count += 1
                        warnings.append(f"LABEL_TRANSIENT_FALLBACK:{retry_model}")
                        continue

                raise LabelAnalysisAgentError(
                    f"SAIA label analysis failed: {self._friendly_error(exc)}"
                ) from exc

    def _run_multi_initial_letter_identification(
        self,
        *,
        image_b64: str,
        label_name: str,
        question: str,
        filename: str,
        regions: Sequence[Any],
        overall_deadline: float,
    ) -> tuple[str, str, list[str], int, float, int, int, int, str]:
        ordered_regions = sorted(list(regions), key=self._bbox_sort_tuple)
        if not ordered_regions:
            raise LabelAnalysisAgentError("No valid coordinates were available for the requested label.")

        lines: list[str] = []
        warnings: list[str] = []
        total_retry_count = 0
        model_used = self._visual_model()
        timeout_used_seconds = 0.0
        max_crop_width = 0
        max_crop_height = 0
        max_crop_bytes = 0
        image_format_sent = "jpeg"

        for index, region in enumerate(ordered_regions, start=1):
            region_crop, _region_crop_b64, _region_bounds, _ = self._crop_regions(image_b64, [region])
            prepared_payload = self._prepare_label_crop_for_model(
                region_crop,
                label_name=label_name,
                analysis_submode="multi_initial_letter_identification",
            )
            max_crop_width = max(max_crop_width, prepared_payload.width)
            max_crop_height = max(max_crop_height, prepared_payload.height)
            max_crop_bytes = max(max_crop_bytes, len(prepared_payload.image_bytes))
            image_format_sent = prepared_payload.image_format

            prompt = _build_mode_prompt(
                label_name=label_name,
                analysis_mode="visual",
                analysis_submode="multi_initial_letter_identification",
                question=question,
                region_count=len(ordered_regions),
                filename=filename,
                ocr_text="",
            )
            prompt = "\n".join(
                [
                    prompt,
                    "",
                    f"Current region in reading order: {index} of {len(ordered_regions)}.",
                    "Return only the letter for this region.",
                ]
            )
            messages = [
                {"role": "system", "content": LABEL_ANALYSIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{prepared_payload.image_b64}"},
                        },
                    ],
                },
            ]
            response, model_used, call_warnings, retry_count, timeout_used_seconds = self._run_saia_completion(
                messages=messages,
                selected_model=model_used,
                prefer_vision=True,
                analysis_submode="multi_initial_letter_identification",
                overall_deadline=overall_deadline,
            )
            warnings.extend(call_warnings)
            total_retry_count += retry_count
            letter, _confidence = self._normalize_initial_letter_output(str(response.get("text", "") or "UNKNOWN"))
            lines.append(f"Region {index}: {letter}")

        deduped_warnings = list(dict.fromkeys(warnings))
        return (
            "\n".join(lines),
            model_used,
            deduped_warnings,
            total_retry_count,
            timeout_used_seconds,
            max_crop_width,
            max_crop_height,
            max_crop_bytes,
            image_format_sent,
        )

    @staticmethod
    def _inspection_payload(
        *,
        label_name: str,
        analysis_mode: str,
        analysis_submode: str,
        model_used: str,
        region_count: int,
        filename: str,
        ocr_text: str,
        output_text: str,
        warnings: list[str],
        stage_metadata: dict[str, Any],
        initial_letter_confidence: str | None = None,
    ) -> dict[str, Any]:
        evidence_used = {
            "crop_image": True,
            "ocr_text": bool(ocr_text),
            "ocr_text_chars": len(str(ocr_text or "")),
            "label_guidance": True,
        }
        payload: dict[str, Any] = {
            "input_source_summary": {
                "label_name": label_name,
                "region_count": region_count,
                "filename": filename or None,
            },
            "model_used": model_used,
            "evidence_used": evidence_used,
            "final_output": output_text,
            "confidence_or_assessment": initial_letter_confidence,
            "warnings": warnings[:4],
            "stage_metadata": stage_metadata,
        }
        return payload

    @staticmethod
    def _region_value(region: Any, key: str, default: Any) -> Any:
        if isinstance(region, dict):
            return region.get(key, default)
        return getattr(region, key, default)

    @staticmethod
    def _extract_bbox(region: Any) -> tuple[int, int, int, int] | None:
        bbox = list(LabelAnalysisAgent._region_value(region, "bbox_xyxy", []) or [])
        if len(bbox) >= 4:
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
            if x2 > x1 and y2 > y1:
                return x1, y1, x2, y2
        polygons = list(LabelAnalysisAgent._region_value(region, "polygons", []) or [])
        xs: list[int] = []
        ys: list[int] = []
        for polygon in polygons:
            if not isinstance(polygon, Sequence):
                continue
            coords = list(polygon)
            for index in range(0, len(coords) - 1, 2):
                try:
                    xs.append(int(round(float(coords[index]))))
                    ys.append(int(round(float(coords[index + 1]))))
                except Exception:
                    continue
        if xs and ys:
            return min(xs), min(ys), max(xs), max(ys)
        return None

    @staticmethod
    def _draw_region_mask(draw: ImageDraw.ImageDraw, region: Any) -> None:
        polygons = list(LabelAnalysisAgent._region_value(region, "polygons", []) or [])
        drew_polygon = False
        for polygon in polygons:
            if not isinstance(polygon, Sequence):
                continue
            coords = list(polygon)
            if len(coords) < 6:
                continue
            points: list[tuple[int, int]] = []
            for index in range(0, len(coords) - 1, 2):
                try:
                    points.append((int(round(float(coords[index]))), int(round(float(coords[index + 1])))))
                except Exception:
                    points = []
                    break
            if len(points) >= 3:
                draw.polygon(points, fill=255)
                drew_polygon = True
        if drew_polygon:
            return
        bbox = LabelAnalysisAgent._extract_bbox(region)
        if bbox is None:
            return
        draw.rectangle(bbox, fill=255)

    @staticmethod
    def _crop_regions(image_b64: str, regions: Sequence[Any]) -> tuple[Image.Image, str, list[int], int]:
        image_bytes = decode_image_bytes(image_b64)
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                source = ImageOps.exif_transpose(image).convert("RGB")
        except Exception as exc:
            raise LabelAnalysisAgentError("Could not decode the source page image.") from exc

        if not regions:
            raise LabelAnalysisAgentError("No label regions were provided.")

        width, height = source.size
        bounds: list[tuple[int, int, int, int]] = []
        mask = Image.new("L", source.size, 0)
        draw = ImageDraw.Draw(mask)

        for region in regions:
            bbox = LabelAnalysisAgent._extract_bbox(region)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            bounds.append((
                max(0, min(x1, width - 1)),
                max(0, min(y1, height - 1)),
                max(1, min(x2, width)),
                max(1, min(y2, height)),
            ))
            LabelAnalysisAgent._draw_region_mask(draw, region)

        if not bounds:
            raise LabelAnalysisAgentError("No valid coordinates were available for the requested label.")

        union_x1 = min(item[0] for item in bounds)
        union_y1 = min(item[1] for item in bounds)
        union_x2 = max(item[2] for item in bounds)
        union_y2 = max(item[3] for item in bounds)
        pad_x = max(12, int(round((union_x2 - union_x1) * 0.06)))
        pad_y = max(12, int(round((union_y2 - union_y1) * 0.06)))
        crop_box = [
            max(0, union_x1 - pad_x),
            max(0, union_y1 - pad_y),
            min(width, union_x2 + pad_x),
            min(height, union_y2 + pad_y),
        ]

        composite = Image.new("RGB", source.size, (255, 255, 255))
        composite.paste(source, mask=mask)
        crop = composite.crop(tuple(crop_box)).copy()
        return crop, encode_png_base64(crop), crop_box, len(bounds)

    def run(self, payload: Any) -> LabelAnalysisResult:
        started_at = time.perf_counter()
        overall_deadline = started_at + _TOTAL_LABEL_ANALYSIS_DEADLINE_SECONDS
        question = str(getattr(payload, "question", "") or "").strip()
        label_name = str(getattr(payload, "label_name", "") or "").strip()
        image_b64 = str(getattr(payload, "image_b64", "") or "").strip()
        regions = list(getattr(payload, "regions", []) or [])
        filename = str(getattr(payload, "filename", "") or "").strip()

        if not question:
            raise LabelAnalysisAgentError("Question is required.")
        if not label_name:
            raise LabelAnalysisAgentError("label_name is required.")
        if not image_b64:
            raise LabelAnalysisAgentError("image_b64 is required.")

        routing = _resolve_label_routing(label_name, question)
        selection = self._select_regions_for_query(
            label_name=label_name,
            question=question,
            routing=routing,
            regions=regions,
        )
        if not selection.regions:
            raise LabelAnalysisAgentError("No valid coordinates were available for the requested label.")
        crop_image, crop_image_b64, crop_bounds_xyxy, region_count = self._crop_regions(image_b64, selection.regions)
        analysis_mode = routing.family
        analysis_submode = routing.submode
        warnings: list[str] = []
        if analysis_submode == "initial_letter_identification" and selection.used_count > 1:
            analysis_submode = "multi_initial_letter_identification"
            warnings.append("Multiple initial regions matched; returning one letter per region in reading order.")
        ocr_text = ""
        crop_width, crop_height = crop_image.size
        crop_bytes_sent = 0
        image_format_sent = "none"
        timeout_used_seconds = 0.0
        retry_count = 0

        if routing.should_ocr_crop:
            try:
                timeout_used_seconds = self._remaining_timeout_seconds(
                    overall_deadline,
                    self._timeout_seconds_for_submode(analysis_submode),
                )
                ocr_result, crop_bytes_sent, image_format_sent = self._run_glm_ocr_for_crop(
                    crop_image,
                    label_name=label_name,
                    timeout_seconds=timeout_used_seconds,
                )
                ocr_text = self._sanitize_direct_ocr_text(label_name, str(ocr_result.text or "").strip())
                warnings.extend(list(ocr_result.warnings or []))
            except (ValueError, GlmOllamaOcrError, LabelAnalysisAgentError) as exc:
                warnings.append(f"LABEL_OCR_FAILED:{exc}")

        if routing.direct_ocr_text:
            direct_text = ocr_text
            if not direct_text:
                if label_name == "Page Number":
                    direct_text = "The page number is unreadable on the selected crop."
                elif label_name == "Quire Mark":
                    direct_text = "The quire mark is unreadable on the selected crop."
                else:
                    direct_text = f'No confident OCR text could be extracted from the selected "{label_name}" region.'
            duration_ms = (time.perf_counter() - started_at) * 1000
            stage_metadata = self._stage_metadata(
                model_used=get_task_model_assignments().ocr_model,
                mode_used=analysis_submode,
                duration_ms=duration_ms,
                region_count_input=selection.input_count,
                region_count_used=selection.used_count,
                selected_region_strategy=selection.strategy,
                crop_width=crop_width,
                crop_height=crop_height,
                crop_bytes=crop_bytes_sent,
                image_format_sent=image_format_sent,
                timeout_seconds=timeout_used_seconds,
                retry_count=retry_count,
            )
            log.info(
                "label analysis complete label=%s mode=%s submode=%s model=%s region_input=%d region_used=%d strategy=%s ocr_chars=%d duration_ms=%.1f",
                label_name,
                analysis_mode,
                analysis_submode,
                get_task_model_assignments().ocr_model,
                selection.input_count,
                selection.used_count,
                selection.strategy,
                len(ocr_text),
                duration_ms,
            )
            return LabelAnalysisResult(
                status="ok",
                text=direct_text,
                label_name=label_name,
                analysis_mode=analysis_mode,
                model_used=get_task_model_assignments().ocr_model,
                warnings=warnings,
                region_count=region_count,
                crop_image_b64=crop_image_b64,
                crop_bounds_xyxy=[int(value) for value in crop_bounds_xyxy],
                ocr_text=ocr_text,
                stage_metadata=stage_metadata,
                inspection=self._inspection_payload(
                    label_name=label_name,
                    analysis_mode=analysis_mode,
                    analysis_submode=analysis_submode,
                    model_used=get_task_model_assignments().ocr_model,
                    region_count=selection.used_count,
                    filename=filename,
                    ocr_text=ocr_text,
                    output_text=direct_text,
                    warnings=warnings,
                    stage_metadata=stage_metadata,
                ),
            )

        if analysis_mode == "structural" and not routing.invoke_model:
            text = self._structural_metadata_answer(
                label_name=label_name,
                question=question,
                region_count=selection.used_count,
                ocr_text=ocr_text,
            )
            duration_ms = (time.perf_counter() - started_at) * 1000
            stage_metadata = self._stage_metadata(
                model_used="metadata-only",
                mode_used=analysis_submode,
                duration_ms=duration_ms,
                region_count_input=selection.input_count,
                region_count_used=selection.used_count,
                selected_region_strategy=selection.strategy,
                crop_width=crop_width,
                crop_height=crop_height,
                crop_bytes=crop_bytes_sent,
                image_format_sent=image_format_sent,
                timeout_seconds=timeout_used_seconds,
                retry_count=0,
            )
            return LabelAnalysisResult(
                status="ok",
                text=text,
                label_name=label_name,
                analysis_mode=analysis_mode,
                model_used="metadata-only",
                warnings=warnings,
                region_count=region_count,
                crop_image_b64=crop_image_b64,
                crop_bounds_xyxy=[int(value) for value in crop_bounds_xyxy],
                ocr_text=ocr_text,
                stage_metadata=stage_metadata,
                inspection=self._inspection_payload(
                    label_name=label_name,
                    analysis_mode=analysis_mode,
                    analysis_submode=analysis_submode,
                    model_used="metadata-only",
                    region_count=selection.used_count,
                    filename=filename,
                    ocr_text=ocr_text,
                    output_text=text,
                    warnings=warnings,
                    stage_metadata=stage_metadata,
                ),
            )

        if analysis_submode == "multi_initial_letter_identification":
            text, selected_model, multi_warnings, retry_count, timeout_used_seconds, crop_width, crop_height, crop_bytes_sent, image_format_sent = self._run_multi_initial_letter_identification(
                image_b64=image_b64,
                label_name=label_name,
                question=question,
                filename=filename,
                regions=selection.regions,
                overall_deadline=overall_deadline,
            )
            warnings.extend(multi_warnings)
            duration_ms = (time.perf_counter() - started_at) * 1000
            stage_metadata = self._stage_metadata(
                model_used=selected_model,
                mode_used=analysis_submode,
                duration_ms=duration_ms,
                region_count_input=selection.input_count,
                region_count_used=selection.used_count,
                selected_region_strategy=selection.strategy,
                crop_width=crop_width,
                crop_height=crop_height,
                crop_bytes=crop_bytes_sent,
                image_format_sent=image_format_sent,
                timeout_seconds=timeout_used_seconds,
                retry_count=retry_count,
            )
            return LabelAnalysisResult(
                status="ok",
                text=text,
                label_name=label_name,
                analysis_mode=analysis_mode,
                model_used=selected_model,
                warnings=list(dict.fromkeys(warnings)),
                region_count=region_count,
                crop_image_b64=crop_image_b64,
                crop_bounds_xyxy=[int(value) for value in crop_bounds_xyxy],
                ocr_text="",
                stage_metadata=stage_metadata,
                inspection=self._inspection_payload(
                    label_name=label_name,
                    analysis_mode=analysis_mode,
                    analysis_submode=analysis_submode,
                    model_used=selected_model,
                    region_count=selection.used_count,
                    filename=filename,
                    ocr_text="",
                    output_text=text,
                    warnings=list(dict.fromkeys(warnings)),
                    stage_metadata=stage_metadata,
                ),
            )

        text_prompt = _build_mode_prompt(
            label_name=label_name,
            analysis_mode=analysis_mode,
            analysis_submode=analysis_submode,
            question=question,
            region_count=selection.used_count,
            filename=filename,
            ocr_text=ocr_text,
        )

        if analysis_mode == "textual":
            messages = [
                {"role": "system", "content": LABEL_ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": text_prompt},
            ]
            selected_model = self._text_model()
            prefer_vision = routing.prefer_vision_model
        else:
            messages = [
                {"role": "system", "content": LABEL_ANALYSIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,",
                            },
                        },
                    ],
                },
            ]
            selected_model = self._visual_model()
            prefer_vision = routing.prefer_vision_model

        prepared_payload: PreparedLabelPayload | None = None
        if analysis_mode != "textual":
            prepared_payload = self._prepare_label_crop_for_model(
                crop_image,
                label_name=label_name,
                analysis_submode=analysis_submode,
            )
            messages[1]["content"][1]["image_url"]["url"] = f"data:image/jpeg;base64,{prepared_payload.image_b64}"
            crop_width = prepared_payload.width
            crop_height = prepared_payload.height
            crop_bytes_sent = len(prepared_payload.image_bytes)
            image_format_sent = prepared_payload.image_format

        response, selected_model, call_warnings, retry_count, timeout_used_seconds = self._run_saia_completion(
            messages=messages,
            selected_model=selected_model,
            prefer_vision=prefer_vision,
            analysis_submode=analysis_submode,
            overall_deadline=overall_deadline,
        )
        warnings.extend(call_warnings)

        if analysis_submode in {"initial_letter_identification", "multi_initial_letter_identification"}:
            text = str(response.get("text", "") or "").strip()
            if not text:
                text = "UNKNOWN"
        else:
            text = str(response.get("text", "") or "").strip()
        if not text:
            text = "No answer could be produced for the requested label."
        initial_letter_confidence: str | None = None
        if analysis_submode == "initial_letter_identification":
            text, initial_letter_confidence = self._normalize_initial_letter_output(text)
        duration_ms = (time.perf_counter() - started_at) * 1000
        stage_metadata = self._stage_metadata(
            model_used=selected_model,
            mode_used=analysis_submode,
            duration_ms=duration_ms,
            region_count_input=selection.input_count,
            region_count_used=selection.used_count,
            selected_region_strategy=selection.strategy,
            crop_width=crop_width,
            crop_height=crop_height,
            crop_bytes=crop_bytes_sent,
            image_format_sent=image_format_sent,
            timeout_seconds=timeout_used_seconds,
            retry_count=retry_count,
        )

        log.info(
            "label analysis complete label=%s mode=%s submode=%s model=%s region_input=%d region_used=%d strategy=%s ocr_chars=%d retries=%d duration_ms=%.1f",
            label_name,
            analysis_mode,
            analysis_submode,
            selected_model,
            selection.input_count,
            selection.used_count,
            selection.strategy,
            len(ocr_text),
            retry_count,
            duration_ms,
        )

        return LabelAnalysisResult(
            status="ok",
            text=text,
            label_name=label_name,
            analysis_mode=analysis_mode,
            model_used=selected_model,
            warnings=warnings,
            region_count=region_count,
            crop_image_b64=crop_image_b64,
            crop_bounds_xyxy=[int(value) for value in crop_bounds_xyxy],
            ocr_text=ocr_text,
            stage_metadata=stage_metadata,
            inspection=self._inspection_payload(
                label_name=label_name,
                analysis_mode=analysis_mode,
                analysis_submode=analysis_submode,
                model_used=selected_model,
                region_count=selection.used_count,
                filename=filename,
                ocr_text=ocr_text,
                output_text=text,
                warnings=warnings,
                stage_metadata=stage_metadata,
                initial_letter_confidence=initial_letter_confidence,
            ),
        )
