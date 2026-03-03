export type ChatRole = "user" | "assistant" | "system";
export type ChatMessageKind = "text" | "ocr";

export interface OcrMessageMeta {
  detected_language: string;
  confidence: number | null;
  warnings: string[];
  script_hint: string;
  lines: string[];
  raw_json: Record<string, unknown> | null;
  model_used: string;
  fallbacks_used: string[];
  status: string;
  quality_label?: string;
  sanity_metrics?: Record<string, number>;
}

export interface WorkspacePage {
  id: string;
  name: string;
  dataUrl: string;
  mimeType: string;
}

export interface WorkspaceDocument {
  id: string;
  name: string;
  pages: WorkspacePage[];
  createdAt: number;
}

export interface WorkspaceChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
  kind?: ChatMessageKind;
  ocrMeta?: OcrMessageMeta;
}

export interface WorkspacePersistedState {
  selectedModel: string | null;
  includeCurrentPageImage: boolean;
}
