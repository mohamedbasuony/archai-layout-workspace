export type ChatRole = "user" | "assistant" | "system";

export interface WorkspacePage {
  id: string;
  name: string;
  dataUrl: string;
  mimeType: string;
}

export interface WorkspaceDocumentMetadata {
  language: string;
  year: string;
  placeOrOrigin: string;
  scriptFamily: string;
  documentType: string;
  notes: string;
}

export interface WorkspaceDocument {
  id: string;
  name: string;
  pages: WorkspacePage[];
  createdAt: number;
  metadata: WorkspaceDocumentMetadata;
}

export interface WorkspaceChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
  imageUrl?: string;
  imageAlt?: string;
}

export interface WorkspacePersistedState {
  selectedModel: string | null;
  includeCurrentPageImage: boolean;
}
