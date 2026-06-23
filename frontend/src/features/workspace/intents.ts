import type { WorkspaceDocumentMetadata } from "@/lib/workspace/types";

import type { OcrPromptHints, WorkspaceIntent } from "./types";

export function normalizeCommandText(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .replace(/4/g, "a")
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ");
}

export function normalizeLabelText(text: string): string {
  return normalizeCommandText(text)
    .replace(/\bembelished\b/g, "embellished")
    .replace(/\blabels\b/g, "label")
    .replace(/\bregions\b/g, "region");
}

export function isSegmentationIntent(text: string): boolean {
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

export function isExtractionIntent(text: string): boolean {
  const value = normalizeCommandText(text);
  return (
    value === "extract text" ||
    value === "extract the text" ||
    value === "extract text in chat" ||
    value === "ocr this page" ||
    value === "run ocr" ||
    value.includes("extract text") ||
    /\bextract\b.*\btext\b/.test(value) ||
    /\bextract\b.*\b(manuscript|page|latin|french|english|old french|middle french|anglo norman|italian|spanish|iberian|portuguese|catalan)\b/.test(value) ||
    value.includes("transcribe") ||
    value.includes("ocr") ||
    value.includes("read the text") ||
    value.includes("what does it say")
  );
}

export function isTranslationIntent(text: string): boolean {
  const value = normalizeCommandText(text);
  return (
    value.startsWith("translate") ||
    value.includes("translate") ||
    value === "in english please" ||
    value === "english please" ||
    (value.includes("english") && (value.includes("text") || value.includes("page") || value.includes("into") || value.includes("to ")))
  );
}

export function isCropIntent(text: string): boolean {
  const value = normalizeLabelText(text);
  return value.includes("crop") || value.includes("cut out") || value.includes("isolate");
}

export function isLabelAnalysisPrompt(text: string): boolean {
  const value = normalizeLabelText(text);
  return /(what is this|what is that|explain|style|art style|origin|origins|motif|ornament|ornamental|decorative|decoration|shape|symbol|design|embellished|initial)/.test(value);
}

export function isEntityIntent(text: string): boolean {
  const value = normalizeCommandText(text);
  const hasEntityTopic = /(entity|entities|person|persons|people|place|places|location|locations|name|names|mention|mentioned|mentions|who|where)/.test(value);
  const hasQuestionShape = /(are there|there any|any\b|which|what|who|where|mentioned|mentions|mention|named|names)/.test(value);
  return hasEntityTopic && hasQuestionShape;
}

export function detectWorkspaceIntent(text: string): WorkspaceIntent {
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

function languageHintFor(value: string, fallbackScriptHint: string): OcrPromptHints {
  if (/\banglo norman\b/.test(value)) {
    return { languageHint: "anglo_norman", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bold french\b/.test(value)) {
    return { languageHint: "old_french", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bmiddle french\b|\bmedieval french\b/.test(value)) {
    return { languageHint: "middle_french", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\blatin\b/.test(value)) {
    return { languageHint: "latin", scriptHintSeed: fallbackScriptHint, ocrBackend: "auto" };
  }
  if (/\bfrench\b/.test(value)) {
    return { languageHint: "french", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\boccitan\b|\bold occitan\b|\bprovencal\b|\bprovençal\b/.test(value)) {
    return { languageHint: "occitan", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bitalian\b/.test(value)) {
    return { languageHint: "italian", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bspanish\b|\biberian\b|\bportuguese\b|\bcatalan\b/.test(value)) {
    return { languageHint: "spanish", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bmiddle english\b/.test(value)) {
    return { languageHint: "middle_english", scriptHintSeed: fallbackScriptHint, ocrBackend: "auto" };
  }
  if (/\bold english\b/.test(value)) {
    return { languageHint: "old_english", scriptHintSeed: "insular_old_english", ocrBackend: "auto" };
  }
  if (/\bmiddle high german\b/.test(value)) {
    return { languageHint: "middle_high_german", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bold high german\b/.test(value)) {
    return { languageHint: "old_high_german", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bgerman\b|\bdutch\b|\bflemish\b/.test(value)) {
    return { languageHint: "german", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  return {};
}

export function extractOcrPromptHints(text: string): OcrPromptHints {
  return languageHintFor(normalizeCommandText(text), "latin");
}

export function metadataToOcrHints(metadata: WorkspaceDocumentMetadata | null | undefined): OcrPromptHints {
  if (!metadata) {
    return {};
  }

  const languageValue = normalizeCommandText(metadata.language || "");
  const scriptValue = normalizeCommandText(metadata.scriptFamily || "");
  const fallbackScriptHint = scriptValue.includes("insular") ? "insular_old_english" : "latin";
  const languageHints = languageHintFor(languageValue, fallbackScriptHint);
  if (languageHints.languageHint || languageHints.scriptHintSeed) {
    return languageHints;
  }
  return scriptValue ? { scriptHintSeed: fallbackScriptHint, ocrBackend: "auto" } : {};
}

export function resolveOcrPromptHints(
  text: string,
  metadata: WorkspaceDocumentMetadata | null | undefined,
): OcrPromptHints {
  const promptHints = extractOcrPromptHints(text);
  return promptHints.languageHint || promptHints.scriptHintSeed ? promptHints : metadataToOcrHints(metadata);
}

export function translationLanguageHint(metadata: WorkspaceDocumentMetadata | null | undefined): string {
  const language = String(metadata?.language || "").trim();
  if (language) {
    return language;
  }
  const hints = metadataToOcrHints(metadata);
  return hints.languageHint ? hints.languageHint.replace(/_/g, " ") : "unknown";
}
