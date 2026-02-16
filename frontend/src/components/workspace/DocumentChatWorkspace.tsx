"use client";

import { ChangeEvent, useEffect, useMemo, useState } from "react";
import Image from "next/image";

import {
  createChatCompletion,
  getChatModels,
  type ChatMessagePayload,
} from "@/lib/api/chat";
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
const LEGACY_EXTRACTION_MODEL_CANDIDATES = [
  "qwen2.5-vl-72b-instruct",
  "qwen2.5-vl",
  "internvl2-large",
  "internvl2",
];

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
  const response = await fetch(page.dataUrl);
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

function pickLegacyExtractionModel(
  models: string[],
  visionModelIds: Set<string>,
  selectedModel: string,
): string | null {
  const byLower = new Map(models.map((model) => [model.toLowerCase(), model]));

  for (const candidate of LEGACY_EXTRACTION_MODEL_CANDIDATES) {
    const match = byLower.get(candidate.toLowerCase());
    if (match) {
      return match;
    }
  }

  if (selectedModel && visionModelIds.has(selectedModel)) {
    return selectedModel;
  }

  const firstVision = models.find((model) => visionModelIds.has(model));
  if (firstVision) {
    return firstVision;
  }

  if (selectedModel) {
    return selectedModel;
  }

  return LEGACY_EXTRACTION_MODEL_CANDIDATES[0];
}

function normalizeExtractedText(raw: string): string {
  const trimmed = String(raw || "").trim();
  if (!trimmed) {
    return "";
  }

  const withoutFence = trimmed.replace(/^```[a-z]*\s*/i, "").replace(/\s*```$/, "").trim();
  if (!withoutFence) {
    return "";
  }

  try {
    const parsed = JSON.parse(withoutFence) as unknown;
    if (typeof parsed === "string") {
      return parsed.trim();
    }
    if (parsed && typeof parsed === "object") {
      const maybeText = (parsed as Record<string, unknown>).text;
      if (typeof maybeText === "string") {
        return maybeText.trim();
      }
      const strings: string[] = [];
      const collect = (value: unknown) => {
        if (typeof value === "string") {
          const v = value.trim();
          if (v) {
            strings.push(v);
          }
          return;
        }
        if (Array.isArray(value)) {
          for (const item of value) {
            collect(item);
          }
          return;
        }
        if (value && typeof value === "object") {
          for (const item of Object.values(value as Record<string, unknown>)) {
            collect(item);
          }
        }
      };
      collect(parsed);
      return strings.join("\n").trim();
    }
  } catch {
    // Keep non-JSON content as-is.
  }

  return withoutFence;
}

export function DocumentChatWorkspace({ initialDocumentId }: DocumentChatWorkspaceProps) {
  const [hydrated, setHydrated] = useState(false);
  const [documents, setDocuments] = useState<WorkspaceDocument[]>([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);
  const [pageIndexByDocument, setPageIndexByDocument] = useState<Record<string, number>>({});
  const [zoomByDocument, setZoomByDocument] = useState<Record<string, number>>({});
  const [messagesByDocument, setMessagesByDocument] = useState<Record<string, WorkspaceChatMessage[]>>({});

  const [models, setModels] = useState<string[]>([]);
  const [visionModelIds, setVisionModelIds] = useState<Set<string>>(new Set());
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [includeCurrentPageImage, setIncludeCurrentPageImage] = useState(false);

  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [assistantLoadingLabel, setAssistantLoadingLabel] = useState("Thinking");
  const [error, setError] = useState<string | null>(null);
  const [segmentedPreviewByPageId, setSegmentedPreviewByPageId] = useState<Record<string, string>>({});
  const [segmentationErrorByPageId, setSegmentationErrorByPageId] = useState<Record<string, string>>({});
  const [segmentingPageId, setSegmentingPageId] = useState<string | null>(null);

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
  const currentSegmentationError = currentPage ? (segmentationErrorByPageId[currentPage.id] ?? null) : null;
  const currentPageIsSegmenting = Boolean(currentPage && segmentingPageId === currentPage.id);

  useEffect(() => {
    let cancelled = false;
    getChatModels()
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setModels(payload.models);
        setVisionModelIds(new Set(payload.vision_models));
        setSelectedModel((prev) => {
          if (prev) {
            return prev;
          }
          const firstVision = payload.models.find((model) => payload.vision_models.includes(model));
          return firstVision || payload.default_model || payload.models[0] || "";
        });
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
      if (parsed.selectedModel) {
        setSelectedModel(parsed.selectedModel);
      }
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
    const requestedModel = (options?.modelOverride || selectedModel || "").trim();
    let modelForRequest = requestedModel;
    const priorMessagesForModel = [...(options?.historyForModel ?? currentMessages)];

    if (shouldAttachImage && !imageDataUrl) {
      setError("No current page image available to attach.");
      return { ok: false, error: "No current page image available to attach." };
    }
    if (shouldAttachImage) {
      if (!modelForRequest || !visionModelIds.has(modelForRequest)) {
        const autoVisionModel = models.find((model) => visionModelIds.has(model));
        if (!autoVisionModel) {
          setError("No vision-capable model is available. Pick a model with vision support.");
          return { ok: false, error: "No vision-capable model is available." };
        }
        modelForRequest = autoVisionModel;
        setSelectedModel(autoVisionModel);
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

  const handleSegmentCurrentPage = async () => {
    if (!currentPage) {
      return;
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
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Segmentation failed.";
      setSegmentationErrorByPageId((prev) => ({ ...prev, [currentPage.id]: message }));
      setError(message);
    } finally {
      setSegmentingPageId((prev) => (prev === currentPage.id ? null : prev));
    }
  };

  const handleExtractTextInChat = async () => {
    if (!currentPage || !currentDocument) {
      return;
    }
    const currentDocumentId = currentDocument.id;
    if (!currentSegmentedPreview) {
      setError("Segment the current page first, then extract text.");
      return;
    }

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
    setMessagesByDocument((prev) => ({
      ...prev,
      [currentDocumentId]: [...(prev[currentDocumentId] ?? []), statusMessage, assistantMessage],
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

    const extractionModel = pickLegacyExtractionModel(models, visionModelIds, selectedModel);
    if (!extractionModel) {
      setExtractionStatus("Extraction failed: no available model.");
      setError("No extraction model is available.");
      return;
    }
    setSelectedModel(extractionModel);
    setExtractionStatus(`Extraction status: using model ${extractionModel}.`);

    setSending(true);
    setAssistantLoadingLabel("Extracting text");
    try {
      setExtractionStatus("Extraction status: running OCR...");
      const response = await createChatCompletion(
        {
          model: extractionModel,
          stream: true,
          temperature: 0,
          context: {
            document_id: currentDocumentId,
            filename: currentPage.name,
            current_page_index: currentPageIndex,
          },
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text:
                    "Read this manuscript page and transcribe all readable text in natural reading order. " +
                    "Return plain text only. Do not return JSON, markdown, labels, coordinates, or explanations. " +
                    "If no readable text is visible, return an empty string.",
                },
                { type: "image_url", image_url: { url: currentPage.dataUrl } },
              ],
            },
          ],
        },
        (delta) => {
          setMessagesByDocument((prev) => {
            const list = [...(prev[currentDocumentId] ?? [])];
            const index = list.findIndex((msg) => msg.id === assistantMessageId);
            if (index === -1) {
              return prev;
            }
            list[index] = { ...list[index], content: `${list[index].content}${delta}` };
            return { ...prev, [currentDocumentId]: list };
          });
        },
      );

      const finalText = normalizeExtractedText(response.text);
      if (!finalText) {
        setExtractionStatus("Extraction complete: no readable text detected.");
        setMessagesByDocument((prev) => {
          const list = [...(prev[currentDocumentId] ?? [])];
          const index = list.findIndex((msg) => msg.id === assistantMessageId);
          if (index === -1) {
            return prev;
          }
          list[index] = { ...list[index], content: "No readable text detected on this page." };
          return { ...prev, [currentDocumentId]: list };
        });
        return;
      }

      setMessagesByDocument((prev) => ({
        ...prev,
        [currentDocumentId]: (prev[currentDocumentId] ?? []).map((msg) =>
          msg.id === assistantMessageId ? { ...msg, content: finalText } : msg,
        ),
      }));
      setExtractionStatus("Extraction complete.");
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "request failed";
      setExtractionStatus(`Extraction failed: ${message}`);
      setError(message);
    } finally {
      setSending(false);
      setAssistantLoadingLabel("Thinking");
    }
  };

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
                  src={currentSegmentedPreview || currentPage.dataUrl}
                  alt={currentSegmentedPreview ? `${currentPage.name} segmented` : currentPage.name}
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
                  disabled={!currentSegmentedPreview || currentPageIsSegmenting || sending}
                >
                  Extract text in chat
                </button>
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
              onChange={(event) => setSelectedModel(event.target.value)}
              className="max-w-64 rounded border bg-background px-2 py-1 text-sm"
            >
              {models.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
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
                </div>
              ))}
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
