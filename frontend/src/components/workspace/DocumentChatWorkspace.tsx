"use client";

import { ChangeEvent, useEffect, useMemo, useState } from "react";
import Image from "next/image";

import {
  createChatCompletion,
  getChatModels,
  type ChatMessagePayload,
} from "@/lib/api/chat";
import {
  extractWithSaiaOcrTrace,
  fetchAuthorityReport,
  fetchTraceTables,
  mergeOCRResultText,
  normalizeTraceStartResponse,
  type OCRExtractResponse,
} from "@/lib/api/ocrAgent";
import { predictSinglePage } from "@/lib/api/predict";
import {
  type WorkspaceChatMessage,
  type WorkspaceDocument,
  type WorkspacePage,
  type WorkspacePersistedState,
} from "@/lib/workspace/types";

interface DocumentChatWorkspaceProps {
  initialDocumentId?: string;
}

const STORAGE_KEY = "archai_workspace_state_v1";
const LOCKED_MODEL_ID = "internvl3.5-30b-a3b";
const TEXT_LABEL_INCLUDE_TOKENS = [
  "script",
  "gloss",
  "header",
  "catchword",
  "page number",
  "quire",
  "line",
  "text",
  "paragraph",
];
const TEXT_LABEL_EXCLUDE_TOKENS = [
  "border",
  "column",
  "table",
  "diagram",
  "illustration",
  "graphic",
  "music",
  "zone",
];

interface CocoCategory {
  id: number;
  name: string;
}

interface CocoAnnotation {
  id: number;
  category_id: number;
  bbox: [number, number, number, number];
  segmentation?: number[] | number[][];
}

interface CocoPayload {
  categories?: CocoCategory[];
  annotations?: CocoAnnotation[];
}

interface OCRLocationSuggestion {
  region_id: string;
  category: string;
  bbox_xywh: [number, number, number, number];
}

function makeId(prefix: string): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.round(Math.random() * 1_000_000)}`;
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(new Error(`Failed to read file: ${file.name}`));
    reader.readAsDataURL(file);
  });
}

async function pageToFile(page: WorkspacePage): Promise<File> {
  const value = String(page.dataUrl || "");
  if (value.startsWith("data:")) {
    const match = value.match(/^data:([^;,]+)?(;base64)?,([\s\S]*)$/);
    if (!match) {
      throw new Error(`Failed to parse page bytes for ${page.name}.`);
    }

    const mime = match[1] || page.mimeType || "image/png";
    const payload = match[3] || "";
    const isBase64 = Boolean(match[2]);
    let blob: Blob;

    if (isBase64) {
      const binary = atob(payload);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
      }
      blob = new Blob([bytes], { type: mime });
    } else {
      blob = new Blob([decodeURIComponent(payload)], { type: mime });
    }

    return new File([blob], page.name, { type: page.mimeType || blob.type || "image/png" });
  }

  const response = await fetch(value);
  if (!response.ok) {
    throw new Error(`Failed to load page bytes for ${page.name}.`);
  }
  const blob = await response.blob();
  return new File([blob], page.name, { type: page.mimeType || blob.type || "image/png" });
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function sortedImages(files: FileList): File[] {
  return Array.from(files)
    .filter((file) => file.type.startsWith("image/"))
    .sort((a, b) => a.name.localeCompare(b.name));
}

function isRelevantTextLabel(label: string): boolean {
  const key = label.toLowerCase();
  if (TEXT_LABEL_EXCLUDE_TOKENS.some((token) => key.includes(token))) {
    return false;
  }
  return TEXT_LABEL_INCLUDE_TOKENS.some((token) => key.includes(token));
}

function extractLocationSuggestions(coco: CocoPayload | null | undefined): OCRLocationSuggestion[] {
  if (!coco) {
    return [];
  }
  const categories = Array.isArray(coco.categories) ? coco.categories : [];
  const annotations = Array.isArray(coco.annotations) ? coco.annotations : [];
  const categoryById = new Map<number, string>();
  for (const category of categories) {
    if (typeof category?.id === "number" && typeof category?.name === "string") {
      categoryById.set(category.id, category.name);
    }
  }

  const suggestions: OCRLocationSuggestion[] = [];
  for (const annotation of annotations) {
    if (!annotation || !Array.isArray(annotation.bbox) || annotation.bbox.length < 4) {
      continue;
    }
    const category = categoryById.get(Number(annotation.category_id)) || "";
    if (!category || !isRelevantTextLabel(category)) {
      continue;
    }
    const [x, y, w, h] = annotation.bbox;
    if (![x, y, w, h].every((value) => Number.isFinite(value))) {
      continue;
    }
    if (w < 8 || h < 8) {
      continue;
    }
    suggestions.push({
      region_id: String(annotation.id),
      category,
      bbox_xywh: [x, y, w, h],
    });
  }

  suggestions.sort((a, b) => {
    const ay = a.bbox_xywh[1];
    const by = b.bbox_xywh[1];
    if (Math.abs(ay - by) > 8) {
      return ay - by;
    }
    return a.bbox_xywh[0] - b.bbox_xywh[0];
  });

  return suggestions.slice(0, 60);
}

function toBase64(dataUrl: string): string {
  const index = dataUrl.indexOf(",");
  if (index === -1) {
    return dataUrl;
  }
  return dataUrl.slice(index + 1);
}

function formatTraceTablesForChat(payload: unknown): string {
  return JSON.stringify(payload, null, 2);
}

function tableRowsAsObjects(
  payload: { tables?: Array<{ table: string; columns: string[]; rows: unknown[][] }> },
  tableName: string,
): Array<Record<string, unknown>> {
  const table = (payload.tables || []).find((item) => item.table === tableName);
  if (!table) {
    return [];
  }
  return (table.rows || []).map((row) => {
    const out: Record<string, unknown> = {};
    for (let index = 0; index < table.columns.length; index += 1) {
      out[String(table.columns[index])] = row[index];
    }
    return out;
  });
}

function formatEntityLinkingReportForChat(payload: { run_id: string; tables?: Array<{ table: string; columns: string[]; rows: unknown[][] }> }): string {
  const pipelineRows = tableRowsAsObjects(payload, "pipeline_runs");
  const mentionRows = tableRowsAsObjects(payload, "entity_mentions");
  const candidateRows = tableRowsAsObjects(payload, "entity_candidates");
  const run = pipelineRows[0] || {};
  const linkedMentionIds = new Set(
    candidateRows
      .map((row) => String(row.mention_id || "").trim())
      .filter(Boolean),
  );

  const bestCandidateByMentionId = new Map<string, Record<string, unknown>>();
  for (const row of candidateRows) {
    const mentionId = String(row.mention_id || "").trim();
    if (!mentionId) {
      continue;
    }
    const score = Number(row.score ?? 0);
    const previous = bestCandidateByMentionId.get(mentionId);
    if (!previous || score > Number(previous.score ?? 0)) {
      bestCandidateByMentionId.set(mentionId, row);
    }
  }

  const topLinked = mentionRows
    .map((mention) => {
      const mentionId = String(mention.mention_id || "").trim();
      return {
        mention,
        candidate: bestCandidateByMentionId.get(mentionId) ?? null,
      };
    })
    .filter((item) => item.candidate)
    .slice(0, 10)
    .map(({ mention, candidate }) => {
      const surface = String(mention.surface || "").trim() || "(empty)";
      const entType = String(mention.ent_type || "unknown").trim();
      const target = String(candidate?.candidate || "").trim() || "(no candidate)";
      const score = Number(candidate?.score ?? 0);
      return `- ${surface} [${entType}] -> ${target} (score=${score.toFixed(2)})`;
    });

  const mentionPreview = mentionRows.slice(0, 10).map((mention) => {
    const surface = String(mention.surface || "").trim() || "(empty)";
    const entType = String(mention.ent_type || "unknown").trim();
    const confidence = Number(mention.confidence ?? 0);
    return `- ${surface} [${entType}] (confidence=${confidence.toFixed(2)})`;
  });

  return [
    "=== ENTITY LINKING REPORT ===",
    `run_id: ${payload.run_id}`,
    `asset_ref: ${String(run.asset_ref || "unknown")}`,
    `mentions_total: ${mentionRows.length}`,
    `candidates_total: ${candidateRows.length}`,
    `linked_total: ${linkedMentionIds.size}`,
    `ocr_quality: ${String(run.quality_label || run.quality_label_v2 || "unknown")}`,
    "",
    `=== TOP LINKED ENTITIES (N=${topLinked.length}) ===`,
    ...(topLinked.length ? topLinked : ["(none)"]),
    "",
    `=== MENTION PREVIEW (N=${mentionPreview.length}) ===`,
    ...(mentionPreview.length ? mentionPreview : ["(none)"]),
  ].join("\n");
}

function getExtractionStatus(result: OCRExtractResponse): "FULL" | "PARTIAL" | "EMPTY" {
  return result.status;
}

function buildSegmentationSummary(coco: CocoPayload | null | undefined): string {
  if (!coco) {
    return "Segmentation completed, but no COCO payload was returned.";
  }

  const categories = Array.isArray(coco.categories) ? coco.categories : [];
  const annotations = Array.isArray(coco.annotations) ? coco.annotations : [];
  const categoryById = new Map<number, string>();
  for (const category of categories) {
    if (typeof category?.id === "number" && typeof category?.name === "string") {
      categoryById.set(category.id, category.name);
    }
  }

  const counts = new Map<string, number>();
  for (const annotation of annotations) {
    const label = categoryById.get(Number(annotation?.category_id)) || `category_${String(annotation?.category_id ?? "unknown")}`;
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }

  const lines = Array.from(counts.entries())
    .sort((a, b) => {
      if (b[1] !== a[1]) {
        return b[1] - a[1];
      }
      return a[0].localeCompare(b[0]);
    })
    .map(([label, count]) => `- ${label}: ${count}`);

  const textLikeCount = extractLocationSuggestions(coco).length;
  return [
    "Segmentation complete.",
    "",
    `Total regions: ${annotations.length}`,
    `Text-like regions: ${textLikeCount}`,
    "",
    "Discovered labels:",
    ...(lines.length ? lines : ["- none"]),
  ].join("\n");
}

function normalizeCommandText(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .replace(/4/g, "a")
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ");
}

function normalizeLabelText(text: string): string {
  return normalizeCommandText(text)
    .replace(/\bembelished\b/g, "embellished")
    .replace(/\blabels\b/g, "label")
    .replace(/\bregions\b/g, "region");
}

type WorkspaceIntent = "segment" | "extract" | "translate" | "entities" | "crop" | null;

function isSegmentationIntent(text: string): boolean {
  const value = normalizeCommandText(text);
  return (
    value === "segment" ||
    value === "segment page" ||
    value === "segment this page" ||
    value === "run segmentation" ||
    value === "show segmentation" ||
    value.includes("segmentation") ||
    value.includes("labels") ||
    value.includes("regions") ||
    value.includes("bounding boxes")
  );
}

function isExtractionIntent(text: string): boolean {
  const value = normalizeCommandText(text);
  return (
    value === "extract text" ||
    value === "extract the text" ||
    value === "extract text in chat" ||
    value === "ocr this page" ||
    value === "run ocr" ||
    value.includes("extract text") ||
    value.includes("transcribe") ||
    value.includes("ocr") ||
    value.includes("read the text") ||
    value.includes("what does it say")
  );
}

function isTranslationIntent(text: string): boolean {
  const value = normalizeCommandText(text);
  return (
    value.startsWith("translate") ||
    value.includes("translate") ||
    value === "in english please" ||
    value === "english please" ||
    (value.includes("english") && (value.includes("text") || value.includes("page") || value.includes("into") || value.includes("to ")))
  );
}

function isCropIntent(text: string): boolean {
  const value = normalizeLabelText(text);
  return value.includes("crop") || value.includes("cut out") || value.includes("isolate");
}

function isEntityIntent(text: string): boolean {
  const value = normalizeCommandText(text);
  const hasEntityTopic = /(entity|entities|person|persons|people|place|places|location|locations|name|names|mention|mentioned|mentions|who|where)/.test(value);
  const hasQuestionShape = /(are there|there any|any\b|which|what|who|where|mentioned|mentions|mention|named|names)/.test(value);
  return hasEntityTopic && hasQuestionShape;
}

function detectWorkspaceIntent(text: string): WorkspaceIntent {
  if (isCropIntent(text)) {
    return "crop";
  }
  if (isSegmentationIntent(text)) {
    return "segment";
  }
  if (isEntityIntent(text)) {
    return "entities";
  }
  if (isExtractionIntent(text)) {
    return "extract";
  }
  if (isTranslationIntent(text)) {
    return "translate";
  }
  return null;
}

function buildEnglishTranslationPrompt(sourceText: string): string {
  return [
    "Translate the following OCR-extracted manuscript text into English.",
    "Return plain English prose only.",
    "Do not return JSON.",
    "Do not explain what you are doing.",
    "Preserve uncertainty when spans are unreadable or marked with ? or […].",
    "",
    "Source text:",
    sourceText,
  ].join("\n");
}

function buildContextualUserPrompt(
  userRequest: string,
  options: {
    sourceText?: string;
    authorityReport?: string;
    mode: "translation" | "entities";
  },
): string {
  const blocks = [
    "Answer the user's request directly.",
    "OCR has already been performed.",
    "Use the extracted text below as the source evidence.",
    "Do not perform OCR again.",
    "Do not repeat the source text unchanged unless the user explicitly asks for a transcription.",
    "Do not return JSON unless the user explicitly asks for JSON.",
    "Do not mention internal pipeline steps, runs, tables, or debugging output.",
    "",
    "User request:",
    userRequest,
  ];

  if (options.mode === "translation") {
    blocks.push(
      "",
      "Task: translate the OCR text into English.",
      "Return only the English translation.",
      "If a span is uncertain, preserve that uncertainty in English instead of copying the source wording verbatim.",
    );
  }

  if (options.mode === "entities") {
    blocks.push(
      "",
      "Task: answer the user's entity question using the OCR text and authority-linking information below.",
      "If no reliable persons or places are present, say that plainly.",
    );
  }

  if (options.sourceText) {
    blocks.push("", "OCR text:", options.sourceText);
  }

  if (options.authorityReport) {
    blocks.push("", "Authority-linking report:", options.authorityReport);
  }

  return blocks.join("\n");
}

function cocoCategoryMap(coco: CocoPayload | null | undefined): Map<number, string> {
  const out = new Map<number, string>();
  for (const category of coco?.categories || []) {
    if (typeof category?.id === "number" && typeof category?.name === "string") {
      out.set(category.id, category.name);
    }
  }
  return out;
}

function availableCropLabels(coco: CocoPayload | null | undefined): string[] {
  const byId = cocoCategoryMap(coco);
  const labels = new Set<string>();
  for (const annotation of coco?.annotations || []) {
    const name = byId.get(Number(annotation.category_id));
    if (name) {
      labels.add(name);
    }
  }
  return Array.from(labels).sort((a, b) => a.localeCompare(b));
}

function resolveCropLabelFromPrompt(text: string, coco: CocoPayload | null | undefined): string | null {
  const prompt = normalizeLabelText(text);
  const labels = availableCropLabels(coco);
  let bestLabel: string | null = null;
  let bestScore = 0;

  for (const label of labels) {
    const normalized = normalizeLabelText(label);
    let score = 0;
    if (prompt.includes(normalized)) {
      score += 100;
    }
    const tokens = normalized.split(" ").filter((token) => token.length >= 3);
    for (const token of tokens) {
      if (prompt.includes(token)) {
        score += 10;
      }
    }
    if (prompt.includes("embellished") && normalized.includes("embellished")) {
      score += 50;
    }
    if (prompt.includes("initial") && normalized.includes("initial")) {
      score += 30;
    }
    if (score > bestScore) {
      bestScore = score;
      bestLabel = label;
    }
  }

  return bestScore > 0 ? bestLabel : null;
}

function annotationPolygons(annotation: CocoAnnotation): number[][] {
  const raw = annotation.segmentation;
  if (!raw) {
    return [];
  }
  if (Array.isArray(raw) && raw.length && typeof raw[0] === "number") {
    return [raw as number[]];
  }
  if (Array.isArray(raw) && Array.isArray(raw[0])) {
    return (raw as number[][]).filter((poly) => Array.isArray(poly) && poly.length >= 6);
  }
  return [];
}

function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new window.Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to load page image for crop."));
    image.src = src;
  });
}

async function buildTransparentCropOverlay(
  pageDataUrl: string,
  coco: CocoPayload,
  label: string,
): Promise<{ imageUrl: string; matchCount: number }> {
  const categoryById = cocoCategoryMap(coco);
  const matches = (coco.annotations || []).filter((annotation) => categoryById.get(Number(annotation.category_id)) === label);
  if (!matches.length) {
    throw new Error(`No regions found for label: ${label}`);
  }

  const source = await loadImageElement(pageDataUrl);
  const canvas = document.createElement("canvas");
  canvas.width = source.naturalWidth || source.width;
  canvas.height = source.naturalHeight || source.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to create crop canvas.");
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (const annotation of matches) {
    const polygons = annotationPolygons(annotation);
    if (polygons.length) {
      for (const polygon of polygons) {
        ctx.save();
        ctx.beginPath();
        for (let index = 0; index < polygon.length; index += 2) {
          const x = polygon[index] ?? 0;
          const y = polygon[index + 1] ?? 0;
          if (index === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.closePath();
        ctx.clip();
        ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
        ctx.restore();
      }
      continue;
    }

    const [x, y, w, h] = annotation.bbox;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
    ctx.restore();
  }

  return {
    imageUrl: canvas.toDataURL("image/png"),
    matchCount: matches.length,
  };
}

export function DocumentChatWorkspace({ initialDocumentId }: DocumentChatWorkspaceProps) {
  const [clientReady, setClientReady] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const [documents, setDocuments] = useState<WorkspaceDocument[]>([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);
  const [pageIndexByDocument, setPageIndexByDocument] = useState<Record<string, number>>({});
  const [zoomByDocument, setZoomByDocument] = useState<Record<string, number>>({});
  const [messagesByDocument, setMessagesByDocument] = useState<Record<string, WorkspaceChatMessage[]>>({});

  const [visionModelIds, setVisionModelIds] = useState<Set<string>>(new Set());
  const [selectedModel, setSelectedModel] = useState<string>(LOCKED_MODEL_ID);
  const [includeCurrentPageImage, setIncludeCurrentPageImage] = useState(false);

  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [assistantLoadingLabel, setAssistantLoadingLabel] = useState("Thinking");
  const [error, setError] = useState<string | null>(null);
  const [segmentedPreviewByPageId, setSegmentedPreviewByPageId] = useState<Record<string, string>>({});
  const [segmentationCocoByPageId, setSegmentationCocoByPageId] = useState<Record<string, CocoPayload>>({});
  const [segmentationErrorByPageId, setSegmentationErrorByPageId] = useState<Record<string, string>>({});
  const [ocrTextByPageId, setOcrTextByPageId] = useState<Record<string, string>>({});
  const [ocrRunIdByPageId, setOcrRunIdByPageId] = useState<Record<string, string>>({});
  const [authorityReportByPageId, setAuthorityReportByPageId] = useState<Record<string, string>>({});
  const [segmentingPageId, setSegmentingPageId] = useState<string | null>(null);
  const [showSegmentationOverlay, setShowSegmentationOverlay] = useState(true);

  const currentDocument = useMemo(
    () => documents.find((doc) => doc.id === selectedDocumentId) ?? null,
    [documents, selectedDocumentId],
  );

  const currentPageIndex = currentDocument
    ? clamp(pageIndexByDocument[currentDocument.id] ?? 0, 0, Math.max(0, currentDocument.pages.length - 1))
    : 0;

  const currentPage = currentDocument ? currentDocument.pages[currentPageIndex] ?? null : null;
  const currentZoom = currentDocument ? zoomByDocument[currentDocument.id] ?? 1 : 1;
  const currentMessages = currentDocument ? (messagesByDocument[currentDocument.id] ?? []) : [];
  const currentSegmentedPreview = currentPage ? (segmentedPreviewByPageId[currentPage.id] ?? null) : null;
  const currentSegmentationCoco = currentPage ? (segmentationCocoByPageId[currentPage.id] ?? null) : null;
  const currentSegmentationError = currentPage ? (segmentationErrorByPageId[currentPage.id] ?? null) : null;
  const currentExtractedText = currentPage ? (ocrTextByPageId[currentPage.id] ?? "") : "";
  const currentOcrRunId = currentPage ? (ocrRunIdByPageId[currentPage.id] ?? "") : "";
  const currentAuthorityReport = currentPage ? (authorityReportByPageId[currentPage.id] ?? "") : "";
  const currentPageIsSegmenting = Boolean(currentPage && segmentingPageId === currentPage.id);

  useEffect(() => {
    setClientReady(true);
  }, []);

  useEffect(() => {
    let cancelled = false;
    getChatModels()
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setVisionModelIds(new Set(payload.vision_models));
        setSelectedModel(LOCKED_MODEL_ID);
        if (!payload.models.includes(LOCKED_MODEL_ID)) {
          setError(`Locked model ${LOCKED_MODEL_ID} is not available on the backend.`);
        }
      })
      .catch((err: unknown) => {
        if (cancelled) {
          return;
        }
        const message = err instanceof Error ? err.message : "Failed to load model list.";
        setError(message);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (!raw) {
        setHydrated(true);
        return;
      }

      const parsed = JSON.parse(raw) as WorkspacePersistedState;
      setIncludeCurrentPageImage(Boolean(parsed.includeCurrentPageImage));
      setSelectedModel(LOCKED_MODEL_ID);
    } catch {
      // Ignore malformed persisted state.
    } finally {
      setHydrated(true);
    }
  }, [initialDocumentId]);

  useEffect(() => {
    if (!hydrated) {
      return;
    }
    const payload: WorkspacePersistedState = {
      selectedModel: selectedModel || null,
      includeCurrentPageImage,
    };
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch {
      // Storage quota/availability should never block workspace usage.
    }
  }, [
    hydrated,
    selectedModel,
    includeCurrentPageImage,
  ]);

  useEffect(() => {
    if (!initialDocumentId || !documents.some((doc) => doc.id === initialDocumentId)) {
      return;
    }
    setSelectedDocumentId(initialDocumentId);
  }, [initialDocumentId, documents]);

  const selectDocument = (documentId: string) => {
    setSelectedDocumentId(documentId);
  };

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) {
      return;
    }

    try {
      setError(null);
      const images = sortedImages(files);
      if (!images.length) {
        setError("Please select one or more image files.");
        return;
      }

      const pages: WorkspacePage[] = await Promise.all(
        images.map(async (file, index) => ({
          id: makeId(`page-${index + 1}`),
          name: file.name,
          dataUrl: await readFileAsDataUrl(file),
          mimeType: file.type || "image/png",
        })),
      );

      const baseName = images[0].name.replace(/\.[^.]+$/, "") || "Document";
      const document: WorkspaceDocument = {
        id: makeId("doc"),
        name: pages.length > 1 ? `${baseName} (${pages.length} pages)` : baseName,
        pages,
        createdAt: Date.now(),
      };

      setDocuments((prev) => [document, ...prev]);
      setPageIndexByDocument((prev) => ({ ...prev, [document.id]: 0 }));
      setZoomByDocument((prev) => ({ ...prev, [document.id]: 1 }));
      setMessagesByDocument((prev) => ({ ...prev, [document.id]: [] }));
      setSelectedDocumentId(document.id);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load selected files.";
      setError(message);
    } finally {
      event.target.value = "";
    }
  };

  const updateCurrentPageIndex = (next: number) => {
    if (!currentDocument) {
      return;
    }
    setPageIndexByDocument((prev) => ({
      ...prev,
      [currentDocument.id]: clamp(next, 0, Math.max(0, currentDocument.pages.length - 1)),
    }));
  };

  const updateZoom = (next: number) => {
    if (!currentDocument) {
      return;
    }
    setZoomByDocument((prev) => ({
      ...prev,
      [currentDocument.id]: clamp(next, 0.5, 2.5),
    }));
  };

  const clearConversation = () => {
    if (!currentDocument) {
      return;
    }
    setMessagesByDocument((prev) => ({ ...prev, [currentDocument.id]: [] }));
  };

  const appendMessages = (documentId: string, nextMessages: WorkspaceChatMessage[]) => {
    setMessagesByDocument((prev) => ({
      ...prev,
      [documentId]: [...(prev[documentId] ?? []), ...nextMessages],
    }));
  };

  const sendPromptToChat = async (
    text: string,
    options?: {
      displayText?: string;
      attachImage?: boolean;
      forceAttachImage?: boolean;
      forcedImageDataUrl?: string;
      loadingLabel?: string;
      modelOverride?: string;
      historyForModel?: WorkspaceChatMessage[];
    },
  ): Promise<{ ok: boolean; error?: string }> => {
    if (!currentDocument || sending) {
      return { ok: false, error: "No active document or request already in progress." };
    }

    if (!text) {
      return { ok: false, error: "Prompt is empty." };
    }

    const explicitImageAttach = options?.attachImage;
    const forcedImageAttach = Boolean(options?.forceAttachImage);
    const shouldAttachImage =
      explicitImageAttach !== undefined
        ? explicitImageAttach
        : (forcedImageAttach || includeCurrentPageImage);
    const imageDataUrl = options?.forcedImageDataUrl ?? currentPage?.dataUrl ?? null;
    const requestedModel = (options?.modelOverride || selectedModel || LOCKED_MODEL_ID).trim();
    const modelForRequest = requestedModel;
    const priorMessagesForModel = [...(options?.historyForModel ?? currentMessages)];

    if (shouldAttachImage && !imageDataUrl) {
      setError("No current page image available to attach.");
      return { ok: false, error: "No current page image available to attach." };
    }
    if (shouldAttachImage) {
      if (!modelForRequest || !visionModelIds.has(modelForRequest)) {
        setError(`Locked model ${LOCKED_MODEL_ID} is not vision-capable or unavailable.`);
        return { ok: false, error: "Locked model is not vision-capable." };
      }
    }
    if (!modelForRequest) {
      setError("No chat model selected.");
      return { ok: false, error: "No chat model selected." };
    }

    const displayText = options?.displayText ?? text;
    const userMessage: WorkspaceChatMessage = {
      id: makeId("msg-user"),
      role: "user",
      content: displayText,
      createdAt: Date.now(),
    };
    const assistantMessage: WorkspaceChatMessage = {
      id: makeId("msg-assistant"),
      role: "assistant",
      content: "",
      createdAt: Date.now(),
    };

    const priorMessages = [...currentMessages];

    setMessagesByDocument((prev) => ({
      ...prev,
      [currentDocument.id]: [...priorMessages, userMessage, assistantMessage],
    }));
    setError(null);
    setAssistantLoadingLabel(options?.loadingLabel || "Thinking");
    setSending(true);

    const apiMessages: ChatMessagePayload[] = [
      ...priorMessagesForModel.map((message) => ({
        role: message.role,
        content: message.content,
      })),
      shouldAttachImage && imageDataUrl
        ? {
            role: "user",
            content: [
              { type: "text", text },
              { type: "image_url", image_url: { url: imageDataUrl } },
            ],
          }
        : {
            role: "user",
            content: text,
          },
    ];

    try {
      const result = await createChatCompletion(
        {
          messages: apiMessages,
          model: modelForRequest,
          temperature: 0.2,
          stream: true,
          context: {
            document_id: currentDocument.id,
            filename: currentPage?.name || currentDocument.name,
            current_page_index: currentPageIndex,
            page_count: currentDocument.pages.length,
            transcript: currentExtractedText || undefined,
            authority_report: currentAuthorityReport || undefined,
            ocr_run_id: currentOcrRunId || undefined,
          },
        },
        (delta) => {
          setMessagesByDocument((prev) => {
            const list = [...(prev[currentDocument.id] ?? [])];
            const index = list.findIndex((msg) => msg.id === assistantMessage.id);
            if (index === -1) {
              return prev;
            }
            list[index] = { ...list[index], content: `${list[index].content}${delta}` };
            return { ...prev, [currentDocument.id]: list };
          });
        },
      );

      if (result.text) {
        setMessagesByDocument((prev) => {
          const list = [...(prev[currentDocument.id] ?? [])];
          const index = list.findIndex((msg) => msg.id === assistantMessage.id);
          if (index === -1) {
            return prev;
          }
          list[index] = { ...list[index], content: result.text };
          return { ...prev, [currentDocument.id]: list };
        });
      }
      return { ok: true };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Chat request failed.";
      setError(message);
      setMessagesByDocument((prev) => {
        const list = [...(prev[currentDocument.id] ?? [])];
        const index = list.findIndex((msg) => msg.id === assistantMessage.id);
        if (index !== -1) {
          list[index] = {
            ...list[index],
            content: `Request failed: ${message}`,
          };
        }
        return { ...prev, [currentDocument.id]: list };
      });
      return { ok: false, error: message };
    } finally {
      setSending(false);
      setAssistantLoadingLabel("Thinking");
    }
  };

  const sendMessage = async () => {
    const text = input.trim();
    const documentId = currentDocument?.id ?? null;
    if (!text || !documentId) {
      return;
    }
    setInput("");
    const userMessage: WorkspaceChatMessage = {
      id: makeId("msg-user"),
      role: "user",
      content: text,
      createdAt: Date.now(),
    };

    const intent = detectWorkspaceIntent(text);

    if (intent === "segment") {
      appendMessages(documentId, [userMessage]);
      await handleSegmentationInChat();
      return;
    }

    if (intent === "crop") {
      appendMessages(documentId, [userMessage]);
      await handleCropInChat(text);
      return;
    }

    if (intent === "extract") {
      appendMessages(documentId, [userMessage]);
      await handleExtractTextInChat();
      return;
    }

    if (intent === "translate") {
      appendMessages(documentId, [userMessage]);
      await handleTranslateInChat(text);
      return;
    }

    if (intent === "entities") {
      appendMessages(documentId, [userMessage]);
      await handleEntityQuestionInChat(text);
      return;
    }

    await sendPromptToChat(text, { loadingLabel: "Thinking" });
  };

  const runSegmentationForCurrentPage = async (): Promise<{ previewUrl: string; coco: CocoPayload } | null> => {
    if (!currentPage) {
      return null;
    }
    setError(null);
    setSegmentingPageId(currentPage.id);
    setSegmentationErrorByPageId((prev) => {
      const next = { ...prev };
      delete next[currentPage.id];
      return next;
    });

    try {
      const pageFile = await pageToFile(currentPage);
      const result = await predictSinglePage(pageFile);
      const suffix = result.annotated_image_url.includes("?") ? "&" : "?";
      const url = `${result.annotated_image_url}${suffix}t=${Date.now()}`;
      setSegmentedPreviewByPageId((prev) => ({ ...prev, [currentPage.id]: url }));
      const coco = (result.coco_json || {}) as CocoPayload;
      setSegmentationCocoByPageId((prev) => ({ ...prev, [currentPage.id]: coco }));
      return { previewUrl: url, coco };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Segmentation failed.";
      setSegmentationErrorByPageId((prev) => ({ ...prev, [currentPage.id]: message }));
      setError(message);
      return null;
    } finally {
      setSegmentingPageId((prev) => (prev === currentPage.id ? null : prev));
    }
  };

  const handleSegmentationInChat = async () => {
    if (!currentPage || !currentDocument) {
      return;
    }
    const currentDocumentId = currentDocument.id;
    const statusMessageId = makeId("msg-status-seg");
    appendMessages(currentDocumentId, [
      {
        id: statusMessageId,
        role: "assistant",
        content: "Segmentation status: running page segmentation...",
        createdAt: Date.now(),
      },
    ]);

    const setSegmentationStatus = (content: string) => {
      setMessagesByDocument((prev) => {
        const list = [...(prev[currentDocumentId] ?? [])];
        const index = list.findIndex((msg) => msg.id === statusMessageId);
        if (index === -1) {
          return prev;
        }
        list[index] = { ...list[index], content };
        return { ...prev, [currentDocumentId]: list };
      });
    };

    setSending(true);
    setAssistantLoadingLabel("Segmenting page");
    try {
      const segmentation = await runSegmentationForCurrentPage();
      if (!segmentation) {
        setSegmentationStatus("Segmentation failed.");
        return;
      }
      appendMessages(currentDocumentId, [
        {
          id: makeId("msg-assistant-segmentation"),
          role: "assistant",
          content: buildSegmentationSummary(segmentation.coco),
          createdAt: Date.now(),
          imageUrl: segmentation.previewUrl,
          imageAlt: currentPage.name,
        },
      ]);
      setSegmentationStatus("Segmentation complete. Summary posted to chat.");
    } finally {
      setSending(false);
      setAssistantLoadingLabel("Thinking");
    }
  };

  const handleCropInChat = async (userText: string) => {
    if (!currentPage || !currentDocument) {
      return;
    }

    const currentDocumentId = currentDocument.id;
    const statusMessageId = makeId("msg-status-crop");
    appendMessages(currentDocumentId, [
      {
        id: statusMessageId,
        role: "assistant",
        content: "Crop status: resolving segmentation labels...",
        createdAt: Date.now(),
      },
    ]);

    const setCropStatus = (content: string) => {
      setMessagesByDocument((prev) => {
        const list = [...(prev[currentDocumentId] ?? [])];
        const index = list.findIndex((msg) => msg.id === statusMessageId);
        if (index === -1) {
          return prev;
        }
        list[index] = { ...list[index], content };
        return { ...prev, [currentDocumentId]: list };
      });
    };

    setSending(true);
    setAssistantLoadingLabel("Cropping label");
    setError(null);

    try {
      let coco = currentSegmentationCoco;
      if (!coco) {
        setCropStatus("Crop status: running segmentation first...");
        const segmentation = await runSegmentationForCurrentPage();
        coco = segmentation?.coco ?? null;
      }

      if (!coco) {
        setCropStatus("Crop failed: segmentation data is unavailable.");
        return;
      }

      const label = resolveCropLabelFromPrompt(userText, coco);
      if (!label) {
        const labels = availableCropLabels(coco);
        appendMessages(currentDocumentId, [
          {
            id: makeId("msg-assistant-crop-no-match"),
            role: "assistant",
            content: labels.length
              ? `Crop failed: no label matched your request. Available labels: ${labels.join(", ")}`
              : "Crop failed: no labels are available on this page.",
            createdAt: Date.now(),
          },
        ]);
        setCropStatus("Crop failed: no matching label found.");
        return;
      }

      setCropStatus(`Crop status: isolating "${label}" from the page...`);
      const cropped = await buildTransparentCropOverlay(currentPage.dataUrl, coco, label);
      appendMessages(currentDocumentId, [
        {
          id: makeId("msg-assistant-crop-image"),
          role: "assistant",
          content: `Cropped ${cropped.matchCount} region${cropped.matchCount === 1 ? "" : "s"} for label "${label}".`,
          createdAt: Date.now(),
          imageUrl: cropped.imageUrl,
          imageAlt: `Crop for ${label}`,
        },
      ]);
      setCropStatus(`Crop complete for "${label}".`);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Crop request failed.";
      setError(message);
      setCropStatus(`Crop failed: ${message}`);
    } finally {
      setSending(false);
      setAssistantLoadingLabel("Thinking");
    }
  };

  const handleExtractTextInChat = async (options?: { includeDebugOutput?: boolean; silent?: boolean }): Promise<{ text: string; runId: string; authorityReport: string } | null> => {
    if (!currentPage || !currentDocument) {
      return null;
    }
    const includeDebugOutput = options?.includeDebugOutput ?? true;
    const silent = options?.silent ?? false;
    const currentDocumentId = currentDocument.id;

    const statusMessageId = makeId("msg-status");
    const assistantMessageId = makeId("msg-assistant-extract");
    const statusMessage: WorkspaceChatMessage = {
      id: statusMessageId,
      role: "assistant",
      content: "Extraction status: preparing OCR request...",
      createdAt: Date.now(),
    };
    const assistantMessage: WorkspaceChatMessage = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      createdAt: Date.now(),
    };
    if (!silent) {
      setMessagesByDocument((prev) => ({
        ...prev,
        [currentDocumentId]: [...(prev[currentDocumentId] ?? []), statusMessage, assistantMessage],
      }));
    }

    const setExtractionStatus = (content: string) => {
      if (silent) {
        return;
      }
      setMessagesByDocument((prev) => {
        const list = [...(prev[currentDocumentId] ?? [])];
        const index = list.findIndex((msg) => msg.id === statusMessageId);
        if (index === -1) {
          return prev;
        }
        list[index] = { ...list[index], content };
        return { ...prev, [currentDocumentId]: list };
      });
    };

    setSending(true);
    setAssistantLoadingLabel("Extracting text");
    try {
      setExtractionStatus("Extraction status: sending full page to SAIA OCR agent...");

      let locationSuggestions = extractLocationSuggestions(currentSegmentationCoco);
      if (!locationSuggestions.length) {
        const segmentation = await runSegmentationForCurrentPage();
        locationSuggestions = extractLocationSuggestions(segmentation?.coco ?? null);
      }

      const traceResult = await extractWithSaiaOcrTrace({
        document_id: currentDocumentId,
        image_id: currentDocumentId,
        page_id: currentPage.id,
        image_b64: toBase64(currentPage.dataUrl),
        location_suggestions: locationSuggestions,
        apply_proofread: true,
      });
      const response = normalizeTraceStartResponse(traceResult);

      const finalText = mergeOCRResultText(response);
      if (!finalText) {
        const status = getExtractionStatus(response);
        setExtractionStatus(`Extraction complete (${status}): no readable text detected.`);
        if (!silent) {
          setMessagesByDocument((prev) => {
            const list = [...(prev[currentDocumentId] ?? [])];
            const index = list.findIndex((msg) => msg.id === assistantMessageId);
            if (index === -1) {
              return prev;
            }
            list[index] = { ...list[index], content: "No readable text detected on this page." };
            return { ...prev, [currentDocumentId]: list };
          });
        }
        return null;
      }

      setOcrTextByPageId((prev) => ({ ...prev, [currentPage.id]: finalText }));
      setOcrRunIdByPageId((prev) => ({ ...prev, [currentPage.id]: traceResult.run_id }));

      if (!silent) {
        setMessagesByDocument((prev) => ({
          ...prev,
          [currentDocumentId]: (prev[currentDocumentId] ?? []).map((msg) =>
            msg.id === assistantMessageId ? { ...msg, content: finalText } : msg,
          ),
        }));
      }
      const status = getExtractionStatus(response);
      let authorityReport = "";
      try {
        const authorityPayload = await fetchAuthorityReport(traceResult.run_id);
        authorityReport = String(authorityPayload.report || "").trim();
      } catch {
        authorityReport = "";
      }
      if (authorityReport) {
        setAuthorityReportByPageId((prev) => ({ ...prev, [currentPage.id]: authorityReport }));
      }
      if (!includeDebugOutput) {
        setExtractionStatus(`Extraction complete (${status}).`);
        return { text: finalText, runId: traceResult.run_id, authorityReport };
      }
      setExtractionStatus(
        `Extraction complete (${status}) [run_id: ${traceResult.run_id}]. Fetching DB table printout...`,
      );

      try {
        const tablePayload = await fetchTraceTables(traceResult.run_id);
        const tableMessage: WorkspaceChatMessage = {
          id: makeId("msg-assistant-trace-table"),
          role: "assistant",
          content: `Pipeline DB printout for run_id ${traceResult.run_id}:\n${formatTraceTablesForChat(tablePayload)}`,
          createdAt: Date.now(),
        };
        setMessagesByDocument((prev) => ({
          ...prev,
          [currentDocumentId]: [...(prev[currentDocumentId] ?? []), tableMessage],
        }));
        const entityReport = authorityReport || formatEntityLinkingReportForChat(tablePayload);
        const entityMessage: WorkspaceChatMessage = {
          id: makeId("msg-assistant-entity-report"),
          role: "assistant",
          content: entityReport,
          createdAt: Date.now(),
        };
        setMessagesByDocument((prev) => ({
          ...prev,
          [currentDocumentId]: [...(prev[currentDocumentId] ?? []), entityMessage],
        }));
        setExtractionStatus(`Extraction complete (${status}) [run_id: ${traceResult.run_id}]. DB table printout posted to chat.`);
      } catch (tableErr: unknown) {
        const tableErrorMessage = tableErr instanceof Error ? tableErr.message : "Failed to fetch DB table printout.";
        const tableMessage: WorkspaceChatMessage = {
          id: makeId("msg-assistant-trace-table-error"),
          role: "assistant",
          content: `Run ${traceResult.run_id} completed, but DB table printout failed: ${tableErrorMessage}`,
          createdAt: Date.now(),
        };
        setMessagesByDocument((prev) => ({
          ...prev,
          [currentDocumentId]: [...(prev[currentDocumentId] ?? []), tableMessage],
        }));
        setExtractionStatus(`Extraction complete (${status}) [run_id: ${traceResult.run_id}], but DB table fetch failed.`);
      }
      return { text: finalText, runId: traceResult.run_id, authorityReport };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "request failed";
      setExtractionStatus(`Extraction failed: ${message}`);
      setError(message);
      return null;
    } finally {
      setSending(false);
      setAssistantLoadingLabel("Thinking");
    }
  };

  const handleTranslateInChat = async (userText: string) => {
    if (!currentPage || !currentDocument) {
      return;
    }
    const userRequest = userText.trim() || "translate to english";
    let sourceText = String(currentExtractedText || "").trim();
    if (!sourceText) {
      sourceText = String((await handleExtractTextInChat({ includeDebugOutput: false, silent: true }))?.text || "").trim();
    }

    if (!sourceText) {
      appendMessages(currentDocument.id, [
        {
          id: makeId("msg-assistant-translate-empty"),
          role: "assistant",
          content: "No extracted text is available to translate.",
          createdAt: Date.now(),
        },
      ]);
      return;
    }

    const prompt = buildContextualUserPrompt(userRequest, {
      sourceText,
      mode: "translation",
    });
    await sendPromptToChat(prompt, {
      displayText: userRequest,
      attachImage: false,
      loadingLabel: "Translating",
      historyForModel: [],
    });
  };

  const handleEntityQuestionInChat = async (userText: string) => {
    if (!currentPage || !currentDocument) {
      return;
    }
    let sourceText = String(currentExtractedText || "").trim();
    let authorityReport = String(currentAuthorityReport || "").trim();
    let runId = String(currentOcrRunId || "").trim();

    if (!sourceText || !runId) {
      const extracted = await handleExtractTextInChat({ includeDebugOutput: false, silent: true });
      sourceText = String(extracted?.text || sourceText || "").trim();
      authorityReport = String(extracted?.authorityReport || authorityReport || "").trim();
      runId = String(extracted?.runId || runId || "").trim();
    } else if (!authorityReport && runId) {
      try {
        const authorityPayload = await fetchAuthorityReport(runId);
        authorityReport = String(authorityPayload.report || "").trim();
        if (authorityReport) {
          setAuthorityReportByPageId((prev) => ({ ...prev, [currentPage.id]: authorityReport }));
        }
      } catch {
        authorityReport = "";
      }
    }

    if (!sourceText) {
      appendMessages(currentDocument.id, [
        {
          id: makeId("msg-assistant-entities-empty"),
          role: "assistant",
          content: "No extracted text is available for entity analysis.",
          createdAt: Date.now(),
        },
      ]);
      return;
    }
    const prompt = buildContextualUserPrompt(userText, {
      sourceText,
      authorityReport: authorityReport || "(no authority report available)",
      mode: "entities",
    });
    await sendPromptToChat(prompt, {
      displayText: userText,
      attachImage: false,
      loadingLabel: "Checking entities",
      historyForModel: [],
    });
  };

  if (!clientReady) {
    return (
      <div className="flex h-screen bg-background text-foreground" suppressHydrationWarning>
        <div className="m-auto text-sm text-muted-foreground">Loading workspace...</div>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      <aside className="flex w-[320px] shrink-0 flex-col border-r bg-muted/30">
        <div className="border-b p-4">
          <h1 className="text-lg font-semibold">ArchAI Workspace</h1>
          <p className="mt-1 text-sm text-muted-foreground">Upload page images and chat with model context.</p>
          <label className="mt-3 inline-flex cursor-pointer rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90">
            Open Document Pages
            <input
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={handleUpload}
            />
          </label>
        </div>

        <div className="border-b p-3">
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">Documents</p>
          <div className="max-h-32 space-y-1 overflow-auto">
            {documents.map((doc) => (
              <button
                key={doc.id}
                type="button"
                onClick={() => selectDocument(doc.id)}
                className={`w-full rounded-md px-2 py-1.5 text-left text-sm ${
                  doc.id === currentDocument?.id ? "bg-primary/10 text-primary" : "hover:bg-accent"
                }`}
              >
                {doc.name}
              </button>
            ))}
            {!documents.length && <p className="text-sm text-muted-foreground">No documents loaded.</p>}
          </div>
        </div>

        <div className="flex min-h-0 flex-1 flex-col p-3">
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">Current Page</p>
          {currentPage ? (
            <>
              <div className="mb-2 flex items-center gap-2 text-xs">
                <button
                  type="button"
                  onClick={() => updateZoom(currentZoom - 0.1)}
                  className="rounded border px-2 py-1 hover:bg-accent"
                >
                  -
                </button>
                <button
                  type="button"
                  onClick={() => updateZoom(currentZoom + 0.1)}
                  className="rounded border px-2 py-1 hover:bg-accent"
                >
                  +
                </button>
                <button
                  type="button"
                  onClick={() => updateZoom(1)}
                  className="rounded border px-2 py-1 hover:bg-accent"
                >
                  Fit
                </button>
                <button
                  type="button"
                  onClick={() => updateZoom(1)}
                  className="rounded border px-2 py-1 hover:bg-accent"
                >
                  100%
                </button>
                <span className="ml-auto text-muted-foreground">{Math.round(currentZoom * 100)}%</span>
              </div>

              <div className="mb-2 flex-1 overflow-auto rounded-md border bg-background p-2">
                <Image
                  src={showSegmentationOverlay && currentSegmentedPreview ? currentSegmentedPreview : currentPage.dataUrl}
                  alt={showSegmentationOverlay && currentSegmentedPreview ? `${currentPage.name} segmented` : currentPage.name}
                  width={1024}
                  height={1024}
                  unoptimized
                  className="mx-auto h-auto max-w-full origin-top transition-transform"
                  style={{ transform: `scale(${currentZoom})` }}
                />
              </div>

              <div className="mb-2 space-y-2">
                <label className="flex items-center gap-2 text-xs text-muted-foreground">
                  <input
                    type="checkbox"
                    checked={showSegmentationOverlay}
                    onChange={(event) => setShowSegmentationOverlay(event.target.checked)}
                    disabled={!currentSegmentedPreview}
                  />
                  Show segmentation overlay
                </label>
                {currentSegmentedPreview && (
                  <p className="text-xs text-emerald-700">Segmented preview ready.</p>
                )}
                {currentSegmentationError && (
                  <p className="text-xs text-red-600">{currentSegmentationError}</p>
                )}
              </div>

              <div className="mb-2 flex items-center gap-2 text-sm">
                <button
                  type="button"
                  onClick={() => updateCurrentPageIndex(currentPageIndex - 1)}
                  disabled={currentPageIndex <= 0}
                  className="rounded border px-2 py-1 disabled:opacity-40"
                >
                  Prev
                </button>
                <select
                  value={currentPageIndex}
                  onChange={(event) => updateCurrentPageIndex(Number(event.target.value))}
                  className="min-w-0 flex-1 rounded border bg-background px-2 py-1"
                >
                  {currentDocument?.pages.map((page, index) => (
                    <option key={page.id} value={index}>
                      Page {index + 1}: {page.name}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() => updateCurrentPageIndex(currentPageIndex + 1)}
                  disabled={Boolean(currentDocument && currentPageIndex >= currentDocument.pages.length - 1)}
                  className="rounded border px-2 py-1 disabled:opacity-40"
                >
                  Next
                </button>
              </div>

              <div className="grid max-h-28 grid-cols-5 gap-1 overflow-auto">
                {currentDocument?.pages.map((page, index) => (
                  <button
                    key={page.id}
                    type="button"
                    onClick={() => updateCurrentPageIndex(index)}
                    className={`overflow-hidden rounded border ${index === currentPageIndex ? "ring-2 ring-primary" : ""}`}
                  >
                    <Image
                      src={page.dataUrl}
                      alt={page.name}
                      width={120}
                      height={56}
                      unoptimized
                      className="h-14 w-full object-cover"
                    />
                  </button>
                ))}
              </div>
            </>
          ) : (
            <p className="text-sm text-muted-foreground">Select or upload a document to begin.</p>
          )}
        </div>
      </aside>

      <section className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-3 border-b px-4 py-3">
          <div>
            <p className="text-sm font-medium">Chat</p>
            <p className="text-xs text-muted-foreground">
              {currentDocument ? `${currentDocument.name} • Page ${currentPageIndex + 1}` : "No document selected"}
            </p>
          </div>

          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              onClick={clearConversation}
              className="rounded border px-3 py-1.5 text-sm hover:bg-accent"
              disabled={!currentDocument || !currentMessages.length}
            >
              New conversation
            </button>
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          {!currentMessages.length ? (
            <p className="mx-auto mt-8 max-w-xl text-center text-sm text-muted-foreground">
              Ask questions about this page, or type `segment this page` / `extract text` to run the pipeline directly in chat.
            </p>
          ) : (
            <div className="mx-auto flex w-full max-w-3xl flex-col gap-4">
              {currentMessages.map((message) => (
                <div
                  key={message.id}
                  className={`rounded-lg px-4 py-3 text-sm leading-relaxed ${
                    message.role === "user"
                      ? "ml-12 bg-primary text-primary-foreground"
                      : "mr-12 border bg-card"
                  }`}
                >
                  <p className="mb-1 text-xs uppercase opacity-70">{message.role}</p>
                  {message.content ? (
                    <p className="whitespace-pre-wrap">{message.content}</p>
                  ) : sending && message.role === "assistant" ? (
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <span className="text-sm">{assistantLoadingLabel}</span>
                      <span className="inline-flex items-center gap-1">
                        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-current" />
                        <span
                          className="h-1.5 w-1.5 animate-bounce rounded-full bg-current"
                          style={{ animationDelay: "0.12s" }}
                        />
                        <span
                          className="h-1.5 w-1.5 animate-bounce rounded-full bg-current"
                          style={{ animationDelay: "0.24s" }}
                        />
                      </span>
                    </div>
                  ) : null}
                  {message.imageUrl ? (
                    <div className="mt-3 overflow-hidden rounded-md border bg-background">
                      <Image
                        src={message.imageUrl}
                        alt={message.imageAlt || "Chat image"}
                        width={1200}
                        height={1200}
                        unoptimized
                        className="h-auto w-full"
                      />
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="border-t p-4">
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-2">
            {error && <p className="text-sm text-red-600">{error}</p>}
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void sendMessage();
                }
              }}
              rows={4}
              placeholder="Ask ArchAI about this page..."
              className="w-full resize-none rounded-md border bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-primary/40"
              disabled={!currentDocument || sending}
            />
            <div className="flex justify-end">
              <button
                type="button"
                onClick={() => void sendMessage()}
                className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                disabled={!currentDocument || sending || !input.trim() || !selectedModel}
              >
                {sending ? "Sending..." : "Send"}
              </button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
