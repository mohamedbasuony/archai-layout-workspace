import type { LabelRegionPayload } from "@/lib/api/chat";
import type { WorkspacePage } from "@/lib/workspace/types";

export interface CocoCategory {
  id: number;
  name: string;
}

export interface CocoAnnotation {
  id: number;
  category_id: number;
  bbox: [number, number, number, number];
  segmentation?: number[] | number[][];
}

export interface CocoPayload {
  categories?: CocoCategory[];
  annotations?: CocoAnnotation[];
}

export interface OCRLocationSuggestion {
  region_id: string;
  category: string;
  bbox_xywh: [number, number, number, number];
}

export type StoredLabelRegion = LabelRegionPayload;
export type StoredLabelRegionMap = Record<string, StoredLabelRegion[]>;

export interface PendingUploadDocument {
  baseName: string;
  pages: WorkspacePage[];
}

export type WorkspaceIntent = "segment" | "extract" | "translate" | "entities" | "crop" | null;

export interface OcrPromptHints {
  scriptHintSeed?: string;
  languageHint?: string;
  ocrBackend?: "auto" | "kraken_mccatmus" | "kraken_catmus" | "kraken_cremma_medieval" | "kraken_cremma_lat";
}
