import type { WorkspaceDocumentMetadata } from "@/lib/workspace/types";

export const STORAGE_KEY = "archai_workspace_state_v1";
export const DEFAULT_CHAT_MODEL_ID = "qwen3-30b-a3b-instruct-2507";
export const DEFAULT_TASK_MODELS = {
  ocr_model: "glm-ocr:latest",
  chat_rag_model: DEFAULT_CHAT_MODEL_ID,
  translation_model: "llama-3.3-70b-instruct",
  label_visual_model: "qwen3-vl-30b-a3b-instruct",
  label_visual_fallback_model: "internvl3.5-30b-a3b",
  verifier_model: "qwen3-235b-a22b",
  embedding_model: "multilingual-e5-large-instruct",
};

export const TEXT_LABEL_INCLUDE_TOKENS = [
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

export const TEXT_LABEL_EXCLUDE_TOKENS = [
  "border",
  "column",
  "table",
  "diagram",
  "illustration",
  "graphic",
  "music",
  "zone",
];

export const SEGMENTATION_LABEL_VOCAB = [
  "Border",
  "Table",
  "Diagram",
  "Main script black",
  "Main script coloured",
  "Variant script black",
  "Variant script coloured",
  "Historiated",
  "Inhabited",
  "Zoo - Anthropomorphic",
  "Embellished",
  "Plain initial- coloured",
  "Plain initial - Highlighted",
  "Plain initial - Black",
  "Page Number",
  "Quire Mark",
  "Running header",
  "Catchword",
  "Gloss",
  "Illustrations",
  "Column",
  "GraphicZone",
  "MusicLine",
  "MusicZone",
  "Music",
];

export const EMPTY_DOCUMENT_METADATA: WorkspaceDocumentMetadata = {
  language: "",
  year: "",
  placeOrOrigin: "",
  scriptFamily: "",
  documentType: "",
  notes: "",
};
