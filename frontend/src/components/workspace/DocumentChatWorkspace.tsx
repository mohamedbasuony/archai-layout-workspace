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
  fetchTraceTables,
  mergeOCRResultText,
  normalizeTraceStartResponse,
  type OCRExtractResponse,
} from "@/lib/api/ocrAgent";
import {
  isRagDebugEnabled,
  getIndexStatus,
  getRetrieveDebug,
  getEvidencePreview,
} from "@/lib/api/rag";
import { getLinkingReport } from "@/lib/api/authorityLinking";
import { predictSinglePage } from "@/lib/api/predict";
import {
  type WorkspaceChatMessage,
  type WorkspaceDocument,
  type WorkspacePage,
  type WorkspacePersistedState,
  type OcrMessageMeta,
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

function getExtractionStatus(result: OCRExtractResponse): "FULL" | "PARTIAL" | "EMPTY" {
  return result.status;
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
  const [segmentingPageId, setSegmentingPageId] = useState<string | null>(null);
  const [showSegmentationOverlay, setShowSegmentationOverlay] = useState(true);
  const [expandedJsonIds, setExpandedJsonIds] = useState<Set<string>>(new Set());

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

  const sendPromptToChat = async (
    text: string,
    options?: {
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

    const explicitImageAttach = Boolean(options?.forceAttachImage);
    const shouldAttachImage = explicitImageAttach || includeCurrentPageImage;
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

    const userMessage: WorkspaceChatMessage = {
      id: makeId("msg-user"),
      role: "user",
      content: text,
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

    const apiMessages: ChatMessagePayload[] = [...priorMessagesForModel, userMessage].map((message) => {
      if (message.id === userMessage.id && shouldAttachImage && imageDataUrl) {
        return {
          role: "user",
          content: [
            { type: "text", text: message.content },
            { type: "image_url", image_url: { url: imageDataUrl } },
          ],
        };
      }
      return {
        role: message.role,
        content: message.content,
      };
    });

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
    if (!text) {
      return;
    }
    setInput("");
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

  const handleSegmentCurrentPage = async () => {
    await runSegmentationForCurrentPage();
  };

  const handleExtractTextInChat = async () => {
    if (!currentPage || !currentDocument) {
      return;
    }
    const currentDocumentId = currentDocument.id;

    const statusMessageId = makeId("msg-status");
    const ocrMessageId = makeId("msg-ocr-result");
    const statusMessage: WorkspaceChatMessage = {
      id: statusMessageId,
      role: "assistant",
      content: "Extracting OCR…",
      createdAt: Date.now(),
    };
    setMessagesByDocument((prev) => ({
      ...prev,
      [currentDocumentId]: [...(prev[currentDocumentId] ?? []), statusMessage],
    }));

    const setExtractionStatus = (content: string) => {
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
      setExtractionStatus("Extracting OCR… running segmentation & sending to SAIA OCR agent…");

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
        return;
      }

      // Build OCR metadata for the inline message
      const ocrMeta: OcrMessageMeta = {
        detected_language: response.detected_language || "unknown",
        confidence: response.confidence,
        warnings: response.warnings || [],
        script_hint: response.script_hint || "unknown",
        lines: response.lines || [],
        raw_json: {
          lines: response.lines,
          text: response.text,
          script_hint: response.script_hint,
          detected_language: response.detected_language,
          confidence: response.confidence,
          warnings: response.warnings,
          fallbacks_used: response.fallbacks_used,
          model_used: response.model_used,
          status: response.status,
          quality_label: response.quality_label,
          sanity_metrics: response.sanity_metrics,
        },
        model_used: response.model_used || "trace-pipeline",
        fallbacks_used: response.fallbacks_used || [],
        status: response.status,
        quality_label: response.quality_label,
        sanity_metrics: response.sanity_metrics,
      };

      // Insert the OCR result as an inline chat message
      const ocrMessage: WorkspaceChatMessage = {
        id: ocrMessageId,
        role: "assistant",
        content: finalText,
        createdAt: Date.now(),
        kind: "ocr",
        ocrMeta,
      };

      // Replace the status message with the result
      setExtractionStatus(`Extraction complete (${response.status}) [run_id: ${traceResult.run_id}]`);
      setMessagesByDocument((prev) => ({
        ...prev,
        [currentDocumentId]: [...(prev[currentDocumentId] ?? []), ocrMessage],
      }));

      // Fetch trace tables in background and add as a DB printout message
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
      }

      // ── RAG VALIDATION REPORT (only when ARCHAI_DEBUG_RAG is enabled) ──
      if (isRagDebugEnabled()) {
        const L: string[] = [];
        const sectionPass: Record<number, boolean> = {};

        // Resolve headline fields
        let assetRef = currentDocumentId;
        let chunksCountSqlite = 0;

        L.push("=== RAG VALIDATION REPORT ===");

        // ── [1] INDEX STATUS ──
        {
          let pass = false;
          try {
            const st = await getIndexStatus(traceResult.run_id);
            chunksCountSqlite = st.chunks_total;
            assetRef = st.asset_ref || currentDocumentId;
            pass = st.missing_chunk_ids.length === 0 && st.chunks_indexed === st.chunks_total && st.chunks_total > 0;
            // Print header fields now that we have them
            L.push(`run_id: ${traceResult.run_id}`);
            L.push(`asset_ref: ${assetRef}`);
            L.push(`chunks_count_sqlite: ${chunksCountSqlite}`);
            L.push("");
            L.push(`[1] INDEX STATUS: ${pass ? "PASS" : "FAIL"}`);
            L.push(JSON.stringify({
              run_id: traceResult.run_id,
              chunks_total: st.chunks_total,
              chunks_indexed: st.chunks_indexed,
              missing_chunk_ids: st.missing_chunk_ids,
            }, null, 2));
          } catch (err: unknown) {
            const e = err as { status?: number };
            L.push(`run_id: ${traceResult.run_id}`);
            L.push(`asset_ref: ${assetRef}`);
            L.push(`chunks_count_sqlite: ${chunksCountSqlite}`);
            L.push("");
            L.push("[1] INDEX STATUS: FAIL");
            L.push(JSON.stringify({
              step: "index_status",
              status: "FAIL",
              http_status: e.status ?? 0,
              error: err instanceof Error ? err.message : String(err),
            }, null, 2));
          }
          sectionPass[1] = pass;
          L.push("");
        }

        // ── [2] RETRIEVE "vilain" ──
        let retrievedChunkIds: string[] = [];
        {
          let pass = false;
          try {
            const probe = await getRetrieveDebug("vilain", 5, traceResult.run_id);
            retrievedChunkIds = probe.results.map((r) => r.chunk_id);
            const hasVilain = probe.results.some(
              (r) => r.text_preview.toLowerCase().includes("vilain"),
            );
            pass = probe.results.length >= 1 && hasVilain;
            L.push(`[2] RETRIEVE "vilain": ${pass ? "PASS" : "FAIL"}`);
            L.push(JSON.stringify({
              query: probe.query,
              k: probe.k,
              filter: { run_id: traceResult.run_id },
              results: probe.results.map((r) => ({
                chunk_id: r.chunk_id,
                chunk_idx: r.chunk_idx,
                offsets: r.offsets,
                score: r.score,
                text_preview: r.text_preview,
              })),
            }, null, 2));
            if (!hasVilain && probe.results.length >= 1) {
              L.push("→ results returned but none contain \"vilain\" (case-insensitive)");
            }
          } catch (err: unknown) {
            const e = err as { status?: number };
            L.push("[2] RETRIEVE \"vilain\": FAIL");
            L.push(JSON.stringify({
              step: "retrieve_debug",
              status: "FAIL",
              http_status: e.status ?? 0,
              error: err instanceof Error ? err.message : String(err),
            }, null, 2));
          }
          sectionPass[2] = pass;
          L.push("");
        }

        // ── [3] EVIDENCE PREVIEW ──
        {
          let pass = false;
          try {
            const ev = await getEvidencePreview("vilain", traceResult.run_id, 5);
            const evIds = new Set(ev.evidence_blocks.map((b) => b.chunk_id));
            const overlap = retrievedChunkIds.some((id) => evIds.has(id));
            pass = ev.hits >= 1 && ev.evidence_blocks.length >= 1 && overlap;
            L.push(`[3] EVIDENCE PREVIEW: ${pass ? "PASS" : "FAIL"}`);
            const blocksToShow = ev.evidence_blocks.slice(0, 2);
            for (const b of blocksToShow) {
              L.push("[EVIDENCE]");
              L.push(`run_id: ${b.run_id}`);
              L.push(`asset_ref: ${b.asset_ref}`);
              L.push(`chunk_id: ${b.chunk_id}`);
              L.push(`chunk_idx: ${b.chunk_idx}`);
              L.push(`offsets: ${b.offsets}`);
              L.push(`text: ${b.text}`);
              L.push("[/EVIDENCE]");
            }
            L.push(`CITATION FORMAT: ${ev.citation_example}`);
          } catch (err: unknown) {
            const e = err as { status?: number };
            L.push("[3] EVIDENCE PREVIEW: FAIL");
            L.push(JSON.stringify({
              step: "evidence_preview",
              status: "FAIL",
              http_status: e.status ?? 0,
              error: err instanceof Error ? err.message : String(err),
            }, null, 2));
          }
          sectionPass[3] = pass;
          L.push("");
        }

        // ── READY ──
        {
          const allPass = [1, 2, 3].every((n) => sectionPass[n]);
          L.push(`READY: ${allPass ? "PASS" : "FAIL"}`);
          if (!allPass) {
            const failing = [1, 2, 3].filter((n) => !sectionPass[n]).map((n) => `[${n}]`);
            L.push(`Failing sections: ${failing.join(", ")}`);
          }
        }

        const ragMessage: WorkspaceChatMessage = {
          id: makeId("msg-assistant-rag-report"),
          role: "assistant",
          content: L.join("\n"),
          createdAt: Date.now(),
        };
        setMessagesByDocument((prev) => ({
          ...prev,
          [currentDocumentId]: [...(prev[currentDocumentId] ?? []), ragMessage],
        }));
      }

      // ── CONSOLIDATED REPORT (single output, replaces separate mention + linking reports) ──
      {
        const consolidatedReportText = (traceResult as unknown as Record<string, unknown>).consolidated_report;
        if (typeof consolidatedReportText === "string" && consolidatedReportText.length > 0) {
          const consolidatedMessage: WorkspaceChatMessage = {
            id: makeId("msg-assistant-consolidated-report"),
            role: "assistant",
            content: consolidatedReportText,
            createdAt: Date.now(),
          };
          setMessagesByDocument((prev) => ({
            ...prev,
            [currentDocumentId]: [...(prev[currentDocumentId] ?? []), consolidatedMessage],
          }));
        } else {
          // Fallback: show mention report + linking report separately if consolidated not available
          const mentionReportText = (traceResult as unknown as Record<string, unknown>).mention_report;
          if (typeof mentionReportText === "string" && mentionReportText.length > 0) {
            const mentionReportMessage: WorkspaceChatMessage = {
              id: makeId("msg-assistant-mention-report"),
              role: "assistant",
              content: mentionReportText,
              createdAt: Date.now(),
            };
            setMessagesByDocument((prev) => ({
              ...prev,
              [currentDocumentId]: [...(prev[currentDocumentId] ?? []), mentionReportMessage],
            }));
          }
          try {
            const lr = await getLinkingReport(traceResult.run_id);
            const reportText = lr.report || "(no report available)";
            const linkingMessage: WorkspaceChatMessage = {
              id: makeId("msg-assistant-linking-report"),
              role: "assistant",
              content: reportText,
              createdAt: Date.now(),
            };
            setMessagesByDocument((prev) => ({
              ...prev,
              [currentDocumentId]: [...(prev[currentDocumentId] ?? []), linkingMessage],
            }));
          } catch {
            // silently skip linking report if unavailable
          }
        }
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "request failed";
      setExtractionStatus(`Extraction failed: ${message}`);
      setError(message);
    } finally {
      setSending(false);
      setAssistantLoadingLabel("Thinking");
    }
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
                <button
                  type="button"
                  onClick={() => void handleSegmentCurrentPage()}
                  className="w-full rounded-md border px-3 py-2 text-sm font-medium hover:bg-accent disabled:opacity-50"
                  disabled={currentPageIsSegmenting}
                >
                  {currentPageIsSegmenting ? "Segmenting..." : "Segment current page"}
                </button>
                <button
                  type="button"
                  onClick={() => void handleExtractTextInChat()}
                  className="w-full rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                  disabled={currentPageIsSegmenting || sending}
                >
                  Extract with SAIA OCR
                </button>
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
            <select
              value={selectedModel}
              disabled
              className="max-w-64 rounded border bg-background px-2 py-1 text-sm"
            >
              <option value={LOCKED_MODEL_ID}>{LOCKED_MODEL_ID}</option>
            </select>

            <label className="flex items-center gap-1 text-xs text-muted-foreground">
              <input
                type="checkbox"
                checked={includeCurrentPageImage}
                disabled={!selectedModel || !visionModelIds.has(selectedModel)}
                onChange={(event) => setIncludeCurrentPageImage(event.target.checked)}
              />
              Include current page image
            </label>

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
              Ask questions about this page, request summaries, or ask for structured extraction output.
            </p>
          ) : (
            <div className="mx-auto flex w-full max-w-3xl flex-col gap-4">
              {currentMessages.map((message) => {
                const isOcr = message.kind === "ocr" && message.ocrMeta;
                return (
                <div
                  key={message.id}
                  className={`rounded-lg px-4 py-3 text-sm leading-relaxed ${
                    message.role === "user"
                      ? "ml-12 bg-primary text-primary-foreground"
                      : "mr-12 border bg-card"
                  }`}
                >
                  <p className="mb-1 text-xs uppercase opacity-70">
                    {isOcr ? "OCR result" : message.role}
                  </p>
                  {isOcr && message.content ? (
                    <>
                      {/* Proofread OCR text */}
                      <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed select-text">
                        {message.content}
                      </pre>

                      {/* Compact metadata */}
                      <div className="mt-3 flex flex-wrap items-center gap-3 border-t pt-2 text-xs text-muted-foreground">
                        {message.ocrMeta!.quality_label && (
                          <span className={
                            message.ocrMeta!.quality_label === "HIGH"
                              ? "font-semibold text-green-600"
                              : message.ocrMeta!.quality_label === "MEDIUM"
                                ? "font-semibold text-amber-600"
                                : "font-semibold text-red-600"
                          }>
                            Quality: {message.ocrMeta!.quality_label}
                          </span>
                        )}
                        <span>
                          Language: <strong>{message.ocrMeta!.detected_language}</strong>
                        </span>
                        {message.ocrMeta!.confidence != null && (
                          <span>
                            Confidence: <strong>{(message.ocrMeta!.confidence * 100).toFixed(0)}%</strong>
                          </span>
                        )}
                        {message.ocrMeta!.warnings.length > 0 && (
                          <span className="text-amber-600">
                            Warnings: {message.ocrMeta!.warnings.slice(0, 3).join(", ")}
                            {message.ocrMeta!.warnings.length > 3 && ` (+${message.ocrMeta!.warnings.length - 3})`}
                          </span>
                        )}
                      </div>

                      {/* Action buttons */}
                      <div className="mt-2 flex items-center gap-2">
                        <button
                          type="button"
                          onClick={async () => {
                            try {
                              await navigator.clipboard.writeText(message.content);
                            } catch {
                              // Ignore clipboard failures.
                            }
                          }}
                          className="rounded border px-2 py-1 text-xs hover:bg-accent"
                        >
                          Copy
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setExpandedJsonIds((prev) => {
                              const next = new Set(prev);
                              if (next.has(message.id)) {
                                next.delete(message.id);
                              } else {
                                next.add(message.id);
                              }
                              return next;
                            });
                          }}
                          className="rounded border px-2 py-1 text-xs hover:bg-accent"
                        >
                          {expandedJsonIds.has(message.id) ? "Hide details" : "Show details"}
                        </button>
                      </div>

                      {/* Collapsible details section */}
                      {expandedJsonIds.has(message.id) && (
                        <div className="mt-2 space-y-2 rounded border bg-muted/40 p-2 text-xs">
                          {message.ocrMeta!.sanity_metrics && (
                            <div>
                              <p className="mb-1 font-semibold">Sanity metrics</p>
                              <table className="w-full text-left">
                                <tbody>
                                  {Object.entries(message.ocrMeta!.sanity_metrics).map(([key, val]) => (
                                    <tr key={key} className="border-b last:border-0">
                                      <td className="py-0.5 pr-3 text-muted-foreground">{key.replace(/_/g, " ")}</td>
                                      <td className="py-0.5 font-mono">
                                        {typeof val === "number" ? (val * 100).toFixed(1) + "%" : String(val)}
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          )}
                          {message.ocrMeta!.raw_json && (
                            <details>
                              <summary className="cursor-pointer font-semibold">Raw JSON</summary>
                              <pre className="mt-1 max-h-48 overflow-auto">
                                {JSON.stringify(message.ocrMeta!.raw_json, null, 2)}
                              </pre>
                            </details>
                          )}
                        </div>
                      )}
                    </>
                  ) : message.content ? (
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
                </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="border-t p-4">
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-2">
            {error && <p className="text-sm text-red-600">{error}</p>}
            {!visionModelIds.has(selectedModel) && includeCurrentPageImage && (
              <p className="text-xs text-amber-600">
                The selected model is not vision-capable. Disable image attachment or choose a vision model.
              </p>
            )}
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
