"use client";

import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import Image from "next/image";

import {
  analyzeSegmentLabel,
  createChatCompletion,
  getChatModels,
  type ChatMessagePayload,
} from "@/lib/api/chat";
import {
  extractPageText,
  fetchAuthorityReport,
  mergeOCRResultText,
  type OCRExtractResponse,
} from "@/lib/api/ocrAgent";
import { predictSinglePage } from "@/lib/api/predict";
import {
  type WorkspaceChatMessage,
  type WorkspaceDocument,
  type WorkspaceDocumentMetadata,
  type WorkspacePage,
  type WorkspacePersistedState,
} from "@/lib/workspace/types";
import {
  DEFAULT_CHAT_MODEL_ID,
  DEFAULT_TASK_MODELS,
  EMPTY_DOCUMENT_METADATA,
  SEGMENTATION_LABEL_VOCAB,
  STORAGE_KEY,
} from "@/features/workspace/constants";
import { clamp, makeId, pageToFile, readFileAsDataUrl, sortedImages, toBase64 } from "@/features/workspace/files";
import {
  detectWorkspaceIntent,
  isLabelAnalysisPrompt,
  resolveOcrPromptHints,
  translationLanguageHint,
} from "@/features/workspace/intents";
import {
  availableStoredLabels,
  bestMatchingSegmentationLabel,
  buildSegmentationSummary,
  buildStoredLabelRegions,
  buildTransparentCropOverlay,
  resolveCropLabelFromPrompt,
} from "@/features/workspace/segmentation";
import type { CocoPayload, PendingUploadDocument, StoredLabelRegionMap } from "@/features/workspace/types";

interface DocumentChatWorkspaceProps {
  initialDocumentId?: string;
}

function removeTextUncertainties(text: string): string {
  return text
    .split("\n")
    .map((line) =>
      line
        .replace(/\[(?:…|\.{3})\]/g, "")
        .replace(/…/g, "")
        .replace(/\?/g, "")
        .replace(/\s{2,}/g, " ")
        .trim(),
    )
    .filter((line, index, lines) => line.length > 0 || index < lines.length - 1)
    .join("\n")
    .trim();
}

function getExtractionStatus(result: OCRExtractResponse): "FULL" | "PARTIAL" | "EMPTY" {
  return result.status;
}

function buildEnglishTranslationPrompt(
  userRequest: string,
  sourceText: string,
  metadata: WorkspaceDocumentMetadata | null | undefined,
): string {
  const sourceLanguage = translationLanguageHint(metadata);
  const sourceLanguageLower = sourceLanguage.toLowerCase();
  const year = String(metadata?.year || "").trim();
  const scriptFamily = String(metadata?.scriptFamily || "").trim();
  const placeOrOrigin = String(metadata?.placeOrOrigin || "").trim();
  const documentType = String(metadata?.documentType || "").trim();
  const notes = String(metadata?.notes || "").trim();
  return [
    `Translate the extracted manuscript passage from ${sourceLanguage} into fluent, faithful English.`,
    "This is a translation request, not an OCR request.",
    "Use the OCR-extracted text below as the only text to translate.",
    `Source language: ${sourceLanguage}.`,
    "Target language: English.",
    year ? `The manuscript year is: ${year}. Use this only as a weak historical-language hint.` : "",
    scriptFamily ? `The manuscript script family is: ${scriptFamily}. Use this only as a weak reading hint.` : "",
    placeOrOrigin ? `The manuscript place or origin is: ${placeOrOrigin}. Use this only as a weak dialect hint.` : "",
    documentType ? `The manuscript document type is: ${documentType}. Use this only as a weak genre/context hint.` : "",
    notes ? `Additional manuscript notes: ${notes}. Use this only as a weak contextual hint and never to invent text.` : "",
    "Treat the extracted transcript as the authoritative source text to interpret and render into English.",
    sourceLanguageLower.includes("old french")
      ? "Treat spelling variation, abbreviation, and likely OCR distortions as expected features of Old French and resolve them contextually when the intended reading is reasonably clear."
      : "Use sentence-level and passage-level context to resolve obvious orthographic or OCR-like distortions where the intended sense is reasonably clear.",
    "Translate at the level of clauses, sentences, and the whole passage, not as a word-by-word gloss.",
    "Produce the best coherent English rendering that the passage supports.",
    "Prefer fluent English syntax over literal token-by-token paraphrase whenever the context makes the intended sense reasonably clear.",
    "Keep the translation roughly proportional to the source passage; do not expand a damaged or repetitive passage into a longer narrative than the source supports.",
    "Silently normalize obvious OCR-like distortions internally when the likely source reading is clear from context.",
    "Do not turn an unclear token into a confident person, place, or plot detail unless the passage clearly supports that reading.",
    "If a clause remains too corrupt to interpret confidently, mark only that local span with [unclear] instead of inventing connective narrative or repeated moral commentary.",
    "Do not summarize, paraphrase away content, or answer questions about the text.",
    "Preserve uncertainty only where it is genuinely unavoidable after contextual interpretation.",
    "If a word or phrase remains unresolved, mark only that local span with [unclear] or [token?].",
    "Do not invent meaning for genuinely unclear words or spans.",
    "Do not echo opaque source tokens as if they were valid English unless they are clearly names or untranslated forms.",
    "Do not describe the page, layout, decoration, or image.",
    "Do not repeat the source text unchanged unless a token is genuinely unreadable.",
    "Do not add a translator's note, note, explanation, or editorial commentary.",
    "Return only the English translation.",
    "Do not return JSON.",
    "Do not explain your reasoning.",
    "",
    "User request:",
    userRequest,
    "",
    "OCR-extracted source text:",
    sourceText,
  ].filter(Boolean).join("\n");
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
      "This is always a best-effort translation request.",
      "Never refuse, gate, or say the OCR is too uncertain.",
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

export function DocumentChatWorkspace({ initialDocumentId }: DocumentChatWorkspaceProps) {
  const [clientReady, setClientReady] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const [documents, setDocuments] = useState<WorkspaceDocument[]>([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);
  const [pageIndexByDocument, setPageIndexByDocument] = useState<Record<string, number>>({});
  const [zoomByDocument, setZoomByDocument] = useState<Record<string, number>>({});
  const [messagesByDocument, setMessagesByDocument] = useState<Record<string, WorkspaceChatMessage[]>>({});

  const [visionModelIds, setVisionModelIds] = useState<Set<string>>(new Set());
  const [taskModels, setTaskModels] = useState(DEFAULT_TASK_MODELS);
  const [selectedModel, setSelectedModel] = useState<string>(DEFAULT_CHAT_MODEL_ID);
  const [includeCurrentPageImage, setIncludeCurrentPageImage] = useState(false);

  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [assistantLoadingLabel, setAssistantLoadingLabel] = useState("Thinking");
  const [error, setError] = useState<string | null>(null);
  const [segmentedPreviewByPageId, setSegmentedPreviewByPageId] = useState<Record<string, string>>({});
  const [segmentationLabelsByPageId, setSegmentationLabelsByPageId] = useState<Record<string, StoredLabelRegionMap>>({});
  const [segmentationErrorByPageId, setSegmentationErrorByPageId] = useState<Record<string, string>>({});
  const [ocrTextByPageId, setOcrTextByPageId] = useState<Record<string, string>>({});
  const [ocrRunIdByPageId, setOcrRunIdByPageId] = useState<Record<string, string>>({});
  const [authorityReportByPageId, setAuthorityReportByPageId] = useState<Record<string, string>>({});
  const [, setSegmentingPageId] = useState<string | null>(null);
  const [showSegmentationOverlay, setShowSegmentationOverlay] = useState(true);
  const [pendingUpload, setPendingUpload] = useState<PendingUploadDocument | null>(null);
  const [metadataDraft, setMetadataDraft] = useState<WorkspaceDocumentMetadata>(EMPTY_DOCUMENT_METADATA);
  const [metadataError, setMetadataError] = useState<string | null>(null);
  const chatScrollContainerRef = useRef<HTMLDivElement | null>(null);
  const chatScrollAnchorRef = useRef<HTMLDivElement | null>(null);

  const currentDocument = useMemo(
    () => documents.find((doc) => doc.id === selectedDocumentId) ?? null,
    [documents, selectedDocumentId],
  );

  const currentPageIndex = currentDocument
    ? clamp(pageIndexByDocument[currentDocument.id] ?? 0, 0, Math.max(0, currentDocument.pages.length - 1))
    : 0;

  const currentPage = currentDocument ? currentDocument.pages[currentPageIndex] ?? null : null;
  const currentZoom = currentDocument ? zoomByDocument[currentDocument.id] ?? 1 : 1;
  const currentMessages = useMemo(
    () => (currentDocument ? messagesByDocument[currentDocument.id] ?? [] : []),
    [currentDocument, messagesByDocument],
  );
  const currentSegmentedPreview = currentPage ? (segmentedPreviewByPageId[currentPage.id] ?? null) : null;
  const currentSegmentationLabels = currentPage ? (segmentationLabelsByPageId[currentPage.id] ?? null) : null;
  const currentSegmentationError = currentPage ? (segmentationErrorByPageId[currentPage.id] ?? null) : null;
  const currentExtractedText = currentPage ? (ocrTextByPageId[currentPage.id] ?? "") : "";
  const currentOcrRunId = currentPage ? (ocrRunIdByPageId[currentPage.id] ?? "") : "";
  const currentAuthorityReport = currentPage ? (authorityReportByPageId[currentPage.id] ?? "") : "";
  const currentMetadata = currentDocument?.metadata ?? null;

  useEffect(() => {
    const container = chatScrollContainerRef.current;
    const anchor = chatScrollAnchorRef.current;
    if (!container || !anchor) {
      return;
    }
    requestAnimationFrame(() => {
      anchor.scrollIntoView({ block: "end", behavior: "auto" });
    });
  }, [currentMessages, sending, currentDocument?.id]);

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
        setTaskModels(payload.task_models || DEFAULT_TASK_MODELS);
        setSelectedModel(payload.default_model || payload.task_models?.chat_rag_model || DEFAULT_CHAT_MODEL_ID);
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
      setSelectedModel(DEFAULT_CHAT_MODEL_ID);
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
      setPendingUpload({ baseName, pages });
      setMetadataDraft({
        ...EMPTY_DOCUMENT_METADATA,
        scriptFamily: "medieval latin script",
        documentType: "manuscript",
      });
      setMetadataError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load selected files.";
      setError(message);
    } finally {
      event.target.value = "";
    }
  };

  const commitPendingUpload = () => {
    if (!pendingUpload) {
      return;
    }

    const language = metadataDraft.language.trim();
    const year = metadataDraft.year.trim();
    if (!language || !year) {
      setMetadataError("Language and year are required before extraction can begin.");
      return;
    }

    const document: WorkspaceDocument = {
      id: makeId("doc"),
      name: pendingUpload.pages.length > 1 ? `${pendingUpload.baseName} (${pendingUpload.pages.length} pages)` : pendingUpload.baseName,
      pages: pendingUpload.pages,
      createdAt: Date.now(),
      metadata: {
        language,
        year,
        placeOrOrigin: metadataDraft.placeOrOrigin.trim(),
        scriptFamily: metadataDraft.scriptFamily.trim(),
        documentType: metadataDraft.documentType.trim(),
        notes: metadataDraft.notes.trim(),
      },
    };

    setDocuments((prev) => [document, ...prev]);
    setPageIndexByDocument((prev) => ({ ...prev, [document.id]: 0 }));
    setZoomByDocument((prev) => ({ ...prev, [document.id]: 1 }));
    setMessagesByDocument((prev) => ({ ...prev, [document.id]: [] }));
    setSelectedDocumentId(document.id);
    setPendingUpload(null);
    setMetadataDraft(EMPTY_DOCUMENT_METADATA);
    setMetadataError(null);
  };

  const cancelPendingUpload = () => {
    setPendingUpload(null);
    setMetadataDraft(EMPTY_DOCUMENT_METADATA);
    setMetadataError(null);
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
      chatStage?: string;
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
    const requestedModel = (options?.modelOverride || selectedModel || taskModels.chat_rag_model || DEFAULT_CHAT_MODEL_ID).trim();
    let modelForRequest = requestedModel;
    const priorMessagesForModel = [...(options?.historyForModel ?? currentMessages)];
    const chatStage = options?.chatStage || (shouldAttachImage ? "visual_chat" : "rag_chat");

    if (shouldAttachImage && !imageDataUrl) {
      setError("No current page image available to attach.");
      return { ok: false, error: "No current page image available to attach." };
    }
    if (shouldAttachImage) {
      if (!modelForRequest || !visionModelIds.has(modelForRequest)) {
        const fallbackVisionModel =
          [taskModels.label_visual_model, taskModels.label_visual_fallback_model]
            .find((candidate) => candidate && visionModelIds.has(candidate))
          || "";
        if (!fallbackVisionModel) {
          setError("No vision-capable chat model is available on the backend.");
          return { ok: false, error: "No vision-capable chat model is available." };
        }
        modelForRequest = fallbackVisionModel;
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
            chat_stage: chatStage,
            document_id: currentDocument.id,
            filename: currentPage?.name || currentDocument.name,
            current_page_index: currentPageIndex,
            page_count: currentDocument.pages.length,
            transcript: currentExtractedText || undefined,
            authority_report: chatStage === "translation" ? undefined : (currentAuthorityReport || undefined),
            ocr_run_id: currentOcrRunId || undefined,
            document_language: currentMetadata?.language || undefined,
            document_year: currentMetadata?.year || undefined,
            place_or_origin: currentMetadata?.placeOrOrigin || undefined,
            script_family: currentMetadata?.scriptFamily || undefined,
            document_type: currentMetadata?.documentType || undefined,
            document_notes: currentMetadata?.notes || undefined,
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
    const matchedLabel =
      resolveCropLabelFromPrompt(text, currentSegmentationLabels)
      || bestMatchingSegmentationLabel(text, SEGMENTATION_LABEL_VOCAB);

    if (!intent && matchedLabel && isLabelAnalysisPrompt(text)) {
      appendMessages(documentId, [userMessage]);
      await handleLabelAnalysisInChat(text);
      return;
    }

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
      await handleExtractTextInChat({ userPrompt: text });
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

  const runSegmentationForCurrentPage = async (): Promise<{ previewUrl: string; coco: CocoPayload; labelsByName: StoredLabelRegionMap } | null> => {
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
      const labelsByName = buildStoredLabelRegions(coco);
      setSegmentationLabelsByPageId((prev) => ({ ...prev, [currentPage.id]: labelsByName }));
      return { previewUrl: url, coco, labelsByName };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Segmentation failed.";
      setSegmentationErrorByPageId((prev) => ({ ...prev, [currentPage.id]: message }));
      setError(message);
      return null;
    } finally {
      setSegmentingPageId((prev) => (prev === currentPage.id ? null : prev));
    }
  };

  const handleLabelAnalysisInChat = async (userText: string) => {
    if (!currentPage || !currentDocument) {
      return;
    }

    const currentDocumentId = currentDocument.id;
    const statusMessageId = makeId("msg-status-label-analysis");
    appendMessages(currentDocumentId, [
      {
        id: statusMessageId,
        role: "assistant",
        content: "Label analysis status: preparing cropped label image...",
        createdAt: Date.now(),
      },
    ]);

    const setLabelStatus = (content: string) => {
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

    setError(null);
    setSending(true);
    setAssistantLoadingLabel("Analyzing label");

    try {
      let labelsByName = currentSegmentationLabels;
      if (!labelsByName) {
        setLabelStatus("Label analysis status: running segmentation first...");
        const segmentation = await runSegmentationForCurrentPage();
        labelsByName = segmentation?.labelsByName ?? null;
      }

      if (!labelsByName) {
        setLabelStatus("Label analysis failed: segmentation data is unavailable.");
        return;
      }

      const label = resolveCropLabelFromPrompt(userText, labelsByName);
      if (!label) {
        const labels = availableStoredLabels(labelsByName);
        appendMessages(currentDocumentId, [
          {
            id: makeId("msg-assistant-label-analysis-no-match"),
            role: "assistant",
            content: labels.length
              ? `Label analysis failed: no label matched your request. Available labels: ${labels.join(", ")}`
              : "Label analysis failed: no labels are available on this page.",
            createdAt: Date.now(),
          },
        ]);
        setLabelStatus("Label analysis failed: no matching label found.");
        return;
      }

      const regions = labelsByName[label] || [];
      if (!regions.length) {
        setLabelStatus(`Label analysis failed: no stored coordinates found for "${label}".`);
        return;
      }

      setLabelStatus(`Label analysis status: cropping "${label}" and analyzing it...`);
      const response = await analyzeSegmentLabel({
        question: userText,
        label_name: label,
        image_b64: toBase64(currentPage.dataUrl),
        regions,
        filename: currentPage.name,
        page_id: currentPage.id,
        document_id: currentDocumentId,
      });
      appendMessages(currentDocumentId, [
        {
          id: makeId("msg-assistant-label-analysis"),
          role: "assistant",
          content: response.text,
          createdAt: Date.now(),
          imageUrl: `data:image/png;base64,${response.crop_image_b64}`,
          imageAlt: `${label} crop`,
        },
      ]);
      if (response.warnings.length) {
        setError(response.warnings.join(" | "));
      }
      setLabelStatus(
        response.analysis_mode
          ? `Label analysis complete for "${label}" in ${response.analysis_mode} mode using ${response.model_used}.`
          : `Label analysis complete for "${label}" using ${response.model_used}.`,
      );
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Label analysis request failed.";
      setError(message);
      setLabelStatus(`Label analysis failed: ${message}`);
    } finally {
      setSending(false);
      setAssistantLoadingLabel("Thinking");
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
      let labelsByName = currentSegmentationLabels;
      if (!labelsByName) {
        setCropStatus("Crop status: running segmentation first...");
        const segmentation = await runSegmentationForCurrentPage();
        labelsByName = segmentation?.labelsByName ?? null;
      }

      if (!labelsByName) {
        setCropStatus("Crop failed: segmentation data is unavailable.");
        return;
      }

      const label = resolveCropLabelFromPrompt(userText, labelsByName);
      if (!label) {
        const labels = availableStoredLabels(labelsByName);
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
      const cropped = await buildTransparentCropOverlay(currentPage.dataUrl, labelsByName, label);
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

  // Coordinates the browser-side OCR workflow for the selected manuscript
  // page. Recognition and trace persistence remain backend responsibilities;
  // this function sends the source image and reflects the returned state in
  // the workspace chat.
  const handleExtractTextInChat = async (options?: {
    includeDebugOutput?: boolean;
    silent?: boolean;
    userPrompt?: string;
  }): Promise<{ text: string; runId: string; authorityReport: string } | null> => {
    if (!currentPage || !currentDocument) {
      return null;
    }
    const silent = options?.silent ?? false;
    const userPrompt = options?.userPrompt ?? "";
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
      setExtractionStatus("Extraction status: running GLM OCR...");

      // Prompt hints provide runtime context for script and language handling.
      // The request still sends the selected page image for live OCR; the
      // frontend does not supply a replacement transcript.
      const ocrHints = resolveOcrPromptHints(userPrompt, currentMetadata);
      const response = await extractPageText({
        document_id: currentDocumentId,
        image_id: currentDocumentId,
        page_id: currentPage.id,
        image_b64: toBase64(currentPage.dataUrl),
        script_hint_seed: ocrHints.scriptHintSeed,
        language_hint: ocrHints.languageHint,
        ocr_backend: "glmocr",
        apply_proofread: false,
        metadata: currentMetadata ? {
          language: currentMetadata.language,
          year: currentMetadata.year,
          place_or_origin: currentMetadata.placeOrOrigin,
          script_family: currentMetadata.scriptFamily,
          document_type: currentMetadata.documentType,
          notes: currentMetadata.notes,
        } : undefined,
      });

      // Keep the recognized text and trace metadata together. The run ID and
      // reports let the UI expose the backend's persisted analysis for the same
      // request that produced the visible transcription.
      const rawExtractedText = mergeOCRResultText(response);
      const finalText = removeTextUncertainties(rawExtractedText);
      const storedText = finalText || "";
      const runId = String(response.run_id || "").trim();
      const authorityReport = String(
        response.consolidated_report
        || response.authority_report
        || response.mention_report
        || "",
      ).trim();
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

      setOcrTextByPageId((prev) => ({ ...prev, [currentPage.id]: storedText }));
      setOcrRunIdByPageId((prev) => ({ ...prev, [currentPage.id]: runId }));
      setAuthorityReportByPageId((prev) => ({ ...prev, [currentPage.id]: authorityReport }));

      if (!silent) {
        setMessagesByDocument((prev) => ({
          ...prev,
          [currentDocumentId]: (prev[currentDocumentId] ?? []).map((msg) =>
            msg.id === assistantMessageId
              ? { ...msg, content: finalText }
              : msg,
          ),
        }));
      }
      const status = getExtractionStatus(response);
      const pipelineBits = [
        typeof response.chunks_count === "number" ? `${response.chunks_count} chunks` : "",
        typeof response.mentions_count === "number" ? `${response.mentions_count} mentions` : "",
      ].filter(Boolean);
      setExtractionStatus(
        pipelineBits.length
          ? `Extraction complete (${status}). Knowledge pipeline: ${pipelineBits.join(", ")}.`
          : `Extraction complete (${status}).`,
      );
      return { text: storedText, runId, authorityReport };
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

    const prompt = buildEnglishTranslationPrompt(userRequest, sourceText, currentMetadata);
    await sendPromptToChat(prompt, {
      displayText: userRequest,
      attachImage: false,
      loadingLabel: "Translating",
      modelOverride: taskModels.translation_model,
      chatStage: "translation",
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
      chatStage: "entity_qa",
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

        {currentMetadata ? (
          <div className="border-b p-3 text-xs text-muted-foreground">
            <p className="mb-2 font-medium uppercase tracking-wide">Metadata</p>
            <div className="space-y-1">
              <p><span className="font-medium text-foreground">Language:</span> {currentMetadata.language}</p>
              <p><span className="font-medium text-foreground">Year:</span> {currentMetadata.year}</p>
              {currentMetadata.placeOrOrigin ? <p><span className="font-medium text-foreground">Origin:</span> {currentMetadata.placeOrOrigin}</p> : null}
              {currentMetadata.scriptFamily ? <p><span className="font-medium text-foreground">Script:</span> {currentMetadata.scriptFamily}</p> : null}
              {currentMetadata.documentType ? <p><span className="font-medium text-foreground">Type:</span> {currentMetadata.documentType}</p> : null}
            </div>
          </div>
        ) : null}

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
                {/* The segmentation preview is a display aid. OCR requests use
                    currentPage.dataUrl so toggling this overlay never changes
                    the source image sent to the backend. */}
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

        <div ref={chatScrollContainerRef} className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
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
              <div ref={chatScrollAnchorRef} />
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
      {pendingUpload ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4">
          <div className="w-full max-w-2xl rounded-xl border bg-background p-6 shadow-2xl">
            <h2 className="text-lg font-semibold">Document metadata</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Language and year are required. These values will guide Kraken routing now and stay with the document for later retrieval and entity work.
            </p>
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <label className="text-sm">
                <span className="mb-1 block font-medium">Language</span>
                <input
                  value={metadataDraft.language}
                  onChange={(event) => setMetadataDraft((prev) => ({ ...prev, language: event.target.value }))}
                  placeholder="latin, middle_french, anglo_norman..."
                  className="w-full rounded-md border bg-background px-3 py-2 outline-none focus:ring-2 focus:ring-primary/40"
                />
              </label>
              <label className="text-sm">
                <span className="mb-1 block font-medium">Year</span>
                <input
                  value={metadataDraft.year}
                  onChange={(event) => setMetadataDraft((prev) => ({ ...prev, year: event.target.value }))}
                  placeholder="e.g. 1248 or c. 1250"
                  className="w-full rounded-md border bg-background px-3 py-2 outline-none focus:ring-2 focus:ring-primary/40"
                />
              </label>
              <label className="text-sm">
                <span className="mb-1 block font-medium">Place / origin</span>
                <input
                  value={metadataDraft.placeOrOrigin}
                  onChange={(event) => setMetadataDraft((prev) => ({ ...prev, placeOrOrigin: event.target.value }))}
                  placeholder="Lausanne, Paris, Toledo..."
                  className="w-full rounded-md border bg-background px-3 py-2 outline-none focus:ring-2 focus:ring-primary/40"
                />
              </label>
              <label className="text-sm">
                <span className="mb-1 block font-medium">Script family</span>
                <input
                  value={metadataDraft.scriptFamily}
                  onChange={(event) => setMetadataDraft((prev) => ({ ...prev, scriptFamily: event.target.value }))}
                  placeholder="medieval latin script, caroline, textualis..."
                  className="w-full rounded-md border bg-background px-3 py-2 outline-none focus:ring-2 focus:ring-primary/40"
                />
              </label>
              <label className="text-sm md:col-span-2">
                <span className="mb-1 block font-medium">Document type</span>
                <input
                  value={metadataDraft.documentType}
                  onChange={(event) => setMetadataDraft((prev) => ({ ...prev, documentType: event.target.value }))}
                  placeholder="gospel, charter, cartulary, liturgical manuscript..."
                  className="w-full rounded-md border bg-background px-3 py-2 outline-none focus:ring-2 focus:ring-primary/40"
                />
              </label>
              <label className="text-sm md:col-span-2">
                <span className="mb-1 block font-medium">Notes</span>
                <textarea
                  value={metadataDraft.notes}
                  onChange={(event) => setMetadataDraft((prev) => ({ ...prev, notes: event.target.value }))}
                  rows={4}
                  placeholder="Repository, shelfmark, dating notes, provenance, paleography notes..."
                  className="w-full rounded-md border bg-background px-3 py-2 outline-none focus:ring-2 focus:ring-primary/40"
                />
              </label>
            </div>
            {metadataError ? <p className="mt-3 text-sm text-red-600">{metadataError}</p> : null}
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                onClick={cancelPendingUpload}
                className="rounded-md border px-4 py-2 text-sm hover:bg-accent"
              >
                Cancel upload
              </button>
              <button
                type="button"
                onClick={commitPendingUpload}
                className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
              >
                Save metadata and open document
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
