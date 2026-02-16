export type ChatRole = "user" | "assistant" | "system";

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
}

export interface WorkspacePersistedState {
  selectedModel: string | null;
  includeCurrentPageImage: boolean;
}
