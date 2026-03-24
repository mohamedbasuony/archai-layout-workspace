import { ApiError, apiFetch } from "./client";

export interface OCRLocationSuggestionPayload {
  region_id?: string;
  category?: string;
  bbox_xywh: [number, number, number, number];
}

export interface OCRRegionPayload {
  region_id?: string;
  bbox_xyxy?: [number, number, number, number];
  polygon?: [number, number][];
  label?: string;
  reading_order?: number;
}

export interface OCRDocumentMetadataPayload {
  language?: string;
  year?: string;
  place_or_origin?: string;
  script_family?: string;
  document_type?: string;
  notes?: string;
}

export interface OCRComparisonRunPayload {
  backend_name: string;
  model_name: string;
  selected: boolean;
  text: string;
  lines: string[];
  confidence?: number | null;
  warnings?: string[];
  language_hint?: string | null;
  script_family?: string | null;
  notes?: string[];
}

export interface OCRExtractRequestPayload {
  document_id?: string;
  image_id?: string;
  page_id?: string;
  image_b64?: string;
  script_hint_seed?: string;
  language_hint?: string;
  apply_proofread?: boolean;
  ocr_backend?: "auto" | "saia" | "kraken" | "kraken_mccatmus" | "kraken_catmus" | "kraken_cremma_medieval" | "kraken_cremma_lat" | "calamari" | "glmocr";
  compare_backends?: ("auto" | "saia" | "kraken" | "kraken_mccatmus" | "kraken_catmus" | "kraken_cremma_medieval" | "kraken_cremma_lat" | "calamari" | "glmocr")[];
  location_suggestions?: OCRLocationSuggestionPayload[];
  regions?: OCRRegionPayload[];
  metadata?: OCRDocumentMetadataPayload;
  benchmark_text?: string;
  benchmark_source?: string;
}

export interface OCRFallback {
  model: string;
  error: string;
}

export interface OCRExtractResponse {
  status: "FULL" | "PARTIAL" | "EMPTY";
  model_used: string;
  fallbacks_used: string[];
  detected_language: string;
  language_confidence: number | null;
  script_hint: "latin" | "greek" | "cyrillic" | "mixed" | "unknown";
  confidence: number | null;
  warnings: string[];
  lines: string[];
  text: string;
  fallbacks?: OCRFallback[];
  comparison_runs?: OCRComparisonRunPayload[];
}

export interface OCRTraceStartResponse {
  run_id: string;
  ocr_result: {
    lines: string[];
    text: string;
    script_hint: "latin" | "greek" | "cyrillic" | "mixed" | "unknown";
    detected_language: string;
    confidence: number;
    warnings: string[];
  };
  proofread_text: string;
  detected_language: string;
  final_confidence: number | null;
}

export interface OCRTraceTable {
  table: string;
  columns: string[];
  rows: unknown[][];
}

export interface OCRTraceTablesResponse {
  run_id: string;
  tables: OCRTraceTable[];
}

export interface OCRAuthorityReportResponse {
  run_id: string;
  report: string;
}

interface LegacyOCRExtractResponse {
  status: "FULL" | "PARTIAL" | "FAIL";
  model_used: string;
  fallbacks: OCRFallback[];
  fallbacks_used?: string[];
  warnings: string[];
  lines: string[];
  text: string;
  script_hint: "latin" | "greek" | "cyrillic" | "mixed" | "unknown";
  detected_language?: string;
  confidence: number;
}

function languageFromScriptHint(scriptHint: LegacyOCRExtractResponse["script_hint"]): string {
  if (scriptHint === "greek") {
    return "greek";
  }
  if (scriptHint === "cyrillic") {
    return "church_slavonic";
  }
  if (scriptHint === "mixed") {
    return "mixed";
  }
  return "latin";
}

function normalizeLegacyResponse(payload: LegacyOCRExtractResponse): OCRExtractResponse {
  const status = payload.status === "FAIL" ? "EMPTY" : payload.status;
  const rawLanguage = (payload.detected_language || "").trim().toLowerCase();
  const detectedLanguage = rawLanguage && rawLanguage !== "unknown"
    ? rawLanguage
    : languageFromScriptHint(payload.script_hint);
  return {
    status,
    model_used: payload.model_used,
    fallbacks_used: payload.fallbacks_used ?? payload.fallbacks.map((item) => item.model),
    detected_language: detectedLanguage,
    language_confidence: null,
    script_hint: payload.script_hint,
    confidence: Number.isFinite(payload.confidence) ? payload.confidence : null,
    warnings: payload.warnings ?? [],
    lines: payload.lines ?? [],
    text: payload.text ?? "",
    fallbacks: payload.fallbacks ?? [],
    comparison_runs: [],
  };
}

export async function extractWithSaiaOcr(
  payload: OCRExtractRequestPayload,
): Promise<OCRExtractResponse> {
  try {
    return await apiFetch<OCRExtractResponse>("/ocr/extract_full_page", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (error: unknown) {
    if (error instanceof ApiError && error.status === 404) {
      const legacy = await apiFetch<LegacyOCRExtractResponse>("/ocr/saia", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_id: payload.image_id,
          page_id: payload.page_id,
          image_b64: payload.image_b64,
          script_hint_seed: payload.script_hint_seed,
          language_hint: payload.language_hint,
          apply_proofread: payload.apply_proofread,
          location_suggestions: payload.location_suggestions ?? [],
          regions: payload.regions ?? [],
        }),
      });
      return normalizeLegacyResponse(legacy);
    }
    throw error;
  }
}

export function normalizeTraceStartResponse(payload: OCRTraceStartResponse): OCRExtractResponse {
  const text = String(payload.proofread_text || payload.ocr_result.text || "");
  const lines = text
    ? text.split("\n").map((line) => line.trim()).filter(Boolean)
    : (payload.ocr_result.lines || []).map((line) => String(line || "").trim()).filter(Boolean);
  const confidence = payload.final_confidence ?? payload.ocr_result.confidence ?? null;
  const warnings = payload.ocr_result.warnings || [];
  const status: OCRExtractResponse["status"] = text.trim()
    ? (warnings.length ? "PARTIAL" : "FULL")
    : "EMPTY";
  return {
    status,
    model_used: "trace-pipeline",
    fallbacks_used: [],
    detected_language: payload.detected_language || payload.ocr_result.detected_language || "unknown",
    language_confidence: confidence,
    script_hint: payload.ocr_result.script_hint,
    confidence,
    warnings,
    lines,
    text,
    fallbacks: [],
  };
}

export async function extractWithSaiaOcrTrace(
  payload: OCRExtractRequestPayload,
): Promise<OCRTraceStartResponse> {
  return apiFetch<OCRTraceStartResponse>("/ocr/page_with_trace", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function fetchTraceTables(runId: string): Promise<OCRTraceTablesResponse> {
  const clean = String(runId || "").trim();
  if (!clean) {
    throw new Error("run_id is required for table fetch.");
  }
  return apiFetch<OCRTraceTablesResponse>(`/ocr/trace/${encodeURIComponent(clean)}/tables`, {
    method: "GET",
  });
}

export async function fetchAuthorityReport(runId: string): Promise<OCRAuthorityReportResponse> {
  const clean = String(runId || "").trim();
  if (!clean) {
    throw new Error("run_id is required for authority report fetch.");
  }
  return apiFetch<OCRAuthorityReportResponse>(`/authority/report/${encodeURIComponent(clean)}`, {
    method: "GET",
  });
}

export function mergeOCRResultText(payload: OCRExtractResponse): string {
  if (payload.text && payload.text.trim()) {
    return payload.text.trim();
  }
  return (payload.lines || []).map((line) => String(line || "").trim()).filter(Boolean).join("\n").trim();
}
