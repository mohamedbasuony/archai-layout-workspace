from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

DEFAULT_SAIA_OCR_MODEL_PREFERENCES = [
    "internvl3.5-30b-a3b",
    "internvl2.5-8b-mpo",
    "internvl2-large",
]
OCR_SCRIPT_HINT = Literal["insular_old_english", "latin_medieval", "unknown"]
OCR_BACKEND_ID = Literal[
    "auto",
    "saia",
    "kraken",
    "kraken_mccatmus",
    "kraken_catmus",
    "kraken_cremma_medieval",
    "kraken_cremma_lat",
    "calamari",
    "glmocr",
]
MANUSCRIPT_DETECTED_LANGUAGE = Literal[
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
]


class OCRRegionInput(BaseModel):
    region_id: str | None = None
    bbox_xyxy: list[float] | None = None
    polygon: list[list[float]] | None = None
    label: str | None = None
    reading_order: int | None = None

    @model_validator(mode="after")
    def _validate_geometry(self) -> "OCRRegionInput":
        if self.bbox_xyxy is None and self.polygon is None:
            raise ValueError("Region requires bbox_xyxy or polygon.")
        if self.bbox_xyxy is not None and len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain exactly 4 numbers.")
        if self.polygon is not None:
            if len(self.polygon) < 3:
                raise ValueError("polygon must contain at least 3 points.")
            for point in self.polygon:
                if len(point) != 2:
                    raise ValueError("Each polygon point must contain exactly 2 numbers.")
        return self


class OCRExtractOptions(BaseModel):
    model_preference: list[str] | None = None
    max_fallbacks: int = Field(default=2, ge=0, le=4)
    quality_floor: float = Field(default=0.60, ge=0.0, le=1.0)
    language_hint: str = "unknown"
    diplomatic: bool = True
    apply_proofread: bool = True
    backend: OCR_BACKEND_ID = "auto"
    compare_backends: list[OCR_BACKEND_ID] = Field(default_factory=list)


class OCRDocumentMetadata(BaseModel):
    language: str | None = None
    year: str | None = None
    place_or_origin: str | None = None
    script_family: str | None = None
    document_type: str | None = None
    notes: str | None = None


class OCRExtractRequest(BaseModel):
    image_id: str | None = None
    page_id: str | None = None
    image_b64: str | None = None
    cropped_image_b64: str | None = None
    region: OCRRegionInput | None = None
    regions: list[OCRRegionInput] | None = None
    prefer_model: str | None = None
    mode: Literal["full", "simple"] = "full"
    options: OCRExtractOptions = Field(default_factory=OCRExtractOptions)
    metadata: OCRDocumentMetadata | None = None
    benchmark_text: str | None = None
    benchmark_source: str | None = None

    @model_validator(mode="after")
    def _validate_regions(self) -> "OCRExtractRequest":
        if not self.image_b64 and not self.cropped_image_b64:
            raise ValueError("Provide image_b64 or cropped_image_b64.")
        return self


class OCRFallback(BaseModel):
    model: str
    reason: str


class OCRRegionResult(BaseModel):
    region_id: str
    text: str
    quality: float = Field(ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    flags: list[str] = Field(default_factory=list)
    bbox_xyxy: list[float] | None = None
    polygon: list[list[float]] | None = None
    label: str | None = None
    reading_order: int | None = None
    backend_name: str | None = None
    model_name: str | None = None
    raw_metadata: dict[str, Any] | None = None


class OCRProvenance(BaseModel):
    crop_sha256: str
    prompt_version: str
    agent_version: str
    timestamp: str


class OCRRawOCRPayload(BaseModel):
    lines: list[str] = Field(default_factory=list)
    text: str = ""
    script_hint: OCR_SCRIPT_HINT = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class OCRComparisonResult(BaseModel):
    page_id: str | None = None
    region_id: str
    backend_name: str
    model_name: str
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    selected: bool = False
    raw_metadata: dict[str, Any] | None = None


class OCRExtractResponse(BaseModel):
    status: Literal["FULL", "PARTIAL", "FAILED"]
    model: str
    ocr_backend: OCR_BACKEND_ID | None = None
    fallbacksUsed: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    text: str
    script_hint: OCR_SCRIPT_HINT = "unknown"
    final_text: str = ""

    # Extended metadata for UI and evidence linkage.
    page_id: str | None = None
    image_id: str | None = None
    fallbacks: list[OCRFallback] = Field(default_factory=list)
    regions: list[OCRRegionResult]
    provenance: OCRProvenance
    raw_ocr: OCRRawOCRPayload | None = None
    comparison_results: list[OCRComparisonResult] = Field(default_factory=list)
    evidence_id: str | None = None
    is_evidence: bool | None = None
    is_verified: bool | None = None


class OCRExtractSimpleResponse(BaseModel):
    text: str
    script_hint: OCR_SCRIPT_HINT
    evidence_id: str | None = None
    is_evidence: bool | None = None
    is_verified: bool | None = None


OCRExtractAnyResponse = OCRExtractResponse | OCRExtractSimpleResponse


DEFAULT_SAIA_OCR_MODEL_PREFS = [
    "internvl3.5-30b-a3b",
    "internvl2.5-8b-mpo",
    "internvl2-large",
]


class SaiaOCRLocationSuggestion(BaseModel):
    region_id: str | None = None
    category: str | None = None
    bbox_xywh: list[float] = Field(default_factory=list)


class SaiaOCRRequest(BaseModel):
    image_id: str | None = None
    page_id: str | None = None
    image_b64: str | None = None
    script_hint_seed: str | None = None
    apply_proofread: bool = True
    location_suggestions: list[SaiaOCRLocationSuggestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_image(self) -> "SaiaOCRRequest":
        if not self.image_b64:
            raise ValueError("Provide image_b64.")
        return self


class SaiaOCRFallback(BaseModel):
    model: str
    error: str


class SaiaOCRResponse(BaseModel):
    status: Literal["FULL", "PARTIAL", "FAIL"]
    model_used: str
    fallbacks: list[SaiaOCRFallback] = Field(default_factory=list)
    fallbacks_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    lines: list[str] = Field(default_factory=list)
    text: str = ""
    script_hint: Literal["latin", "greek", "cyrillic", "mixed", "unknown"] = "unknown"
    detected_language: MANUSCRIPT_DETECTED_LANGUAGE = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_json: dict[str, Any] | None = None


class SaiaFullPageExtractRequest(BaseModel):
    document_id: str | None = None
    image_id: str | None = None
    page_id: str | None = None
    image_b64: str | None = None
    script_hint_seed: str | None = None
    apply_proofread: bool = True
    location_suggestions: list[SaiaOCRLocationSuggestion] = Field(default_factory=list)
    regions: list[OCRRegionInput] = Field(default_factory=list)
    ocr_backend: OCR_BACKEND_ID = "auto"
    compare_backends: list[OCR_BACKEND_ID] = Field(default_factory=list)
    language_hint: str | None = None
    metadata: OCRDocumentMetadata | None = None
    benchmark_text: str | None = None
    benchmark_source: str | None = None

    @model_validator(mode="after")
    def _validate_image(self) -> "SaiaFullPageExtractRequest":
        if not self.image_b64:
            raise ValueError("Provide image_b64.")
        return self


class SaiaFullPageExtractResponse(BaseModel):
    status: Literal["FULL", "PARTIAL", "EMPTY"]
    model_used: str
    fallbacks_used: list[str] = Field(default_factory=list)
    detected_language: MANUSCRIPT_DETECTED_LANGUAGE = "unknown"
    language_confidence: float | None = None
    script_hint: Literal["latin", "greek", "cyrillic", "mixed", "unknown"] = "unknown"
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)
    lines: list[str] = Field(default_factory=list)
    text: str = ""
    fallbacks: list[SaiaOCRFallback] = Field(default_factory=list)
    comparison_runs: list["OCRComparisonSummary"] = Field(default_factory=list)


class OCRComparisonSummary(BaseModel):
    backend_name: str
    model_name: str
    selected: bool = False
    text: str = ""
    lines: list[str] = Field(default_factory=list)
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)
    language_hint: str | None = None
    script_family: str | None = None
    notes: list[str] = Field(default_factory=list)


class EvidenceSpanCreateRequest(BaseModel):
    page_id: str
    region_id: str
    text: str
    bbox_xyxy: list[float] | None = None
    polygon: list[list[float]] | None = None
    model_used: str
    prompt_version: str
    crop_sha256: str


class EvidenceSpanRecord(BaseModel):
    span_id: str
    page_id: str
    region_id: str
    text: str
    bbox_xyxy: list[float] | None = None
    polygon: list[list[float]] | None = None
    model_used: str
    prompt_version: str
    crop_sha256: str
    created_at: str
