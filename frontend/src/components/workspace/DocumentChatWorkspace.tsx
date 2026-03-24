"use client";

import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import Image from "next/image";

import {
  createChatCompletion,
  getChatModels,
  type ChatMessagePayload,
} from "@/lib/api/chat";
import {
  extractWithSaiaOcr,
  fetchAuthorityReport,
  mergeOCRResultText,
  type OCRRegionPayload,
  type OCRExtractResponse,
  type OCRExtractRequestPayload,
} from "@/lib/api/ocrAgent";
import { predictSinglePage } from "@/lib/api/predict";
import {
  type WorkspaceChatMessage,
  type WorkspaceDocument,
  type WorkspaceDocumentMetadata,
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

interface PendingUploadDocument {
  baseName: string;
  pages: WorkspacePage[];
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

function polygonArea(flatPoints: number[]): number {
  if (!Array.isArray(flatPoints) || flatPoints.length < 6) {
    return 0;
  }
  let area = 0;
  for (let index = 0; index < flatPoints.length; index += 2) {
    const nextIndex = (index + 2) % flatPoints.length;
    area += (flatPoints[index] ?? 0) * (flatPoints[nextIndex + 1] ?? 0);
    area -= (flatPoints[nextIndex] ?? 0) * (flatPoints[index + 1] ?? 0);
  }
  return Math.abs(area) / 2;
}

function bestAnnotationPolygon(annotation: CocoAnnotation): [number, number][] | undefined {
  const polygons = annotationPolygons(annotation);
  if (!polygons.length) {
    return undefined;
  }
  const best = [...polygons].sort((left, right) => polygonArea(right) - polygonArea(left))[0] ?? null;
  if (!best) {
    return undefined;
  }
  const points: [number, number][] = [];
  for (let index = 0; index < best.length; index += 2) {
    const x = Number(best[index]);
    const y = Number(best[index + 1]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      continue;
    }
    points.push([x, y]);
  }
  return points.length >= 3 ? points : undefined;
}

function regionSpecificity(label: string): number {
  const key = label.toLowerCase();
  if (key.includes("line")) {
    return 0;
  }
  if (
    key.includes("main script") ||
    key.includes("variant script") ||
    key.includes("gloss") ||
    key.includes("header") ||
    key.includes("catchword") ||
    key.includes("page number") ||
    key.includes("quire")
  ) {
    return 1;
  }
  return 2;
}

function isPreferredLineLabel(label: string): boolean {
  const key = label.toLowerCase();
  return key.includes("line") || key.includes("main script") || key.includes("variant script");
}

function bboxArea(bbox: [number, number, number, number]): number {
  return Math.max(0, bbox[2] - bbox[0]) * Math.max(0, bbox[3] - bbox[1]);
}

function bboxCoverage(left: [number, number, number, number], right: [number, number, number, number]): number {
  const interX1 = Math.max(left[0], right[0]);
  const interY1 = Math.max(left[1], right[1]);
  const interX2 = Math.min(left[2], right[2]);
  const interY2 = Math.min(left[3], right[3]);
  const interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
  const minArea = Math.max(1, Math.min(bboxArea(left), bboxArea(right)));
  return interArea / minArea;
}

function extractStructuredRegions(coco: CocoPayload | null | undefined): OCRRegionPayload[] {
  if (!coco) {
    return [];
  }

  function isColumnLikeLabel(label: string): boolean {
    const key = label.toLowerCase();
    return /column|mainzone|main zone|text zone|main text area/.test(key);
  }

  function bboxOverlapRatio(left: [number, number, number, number], right: [number, number, number, number]): number {
    const interX1 = Math.max(left[0], right[0]);
    const interY1 = Math.max(left[1], right[1]);
    const interX2 = Math.min(left[2], right[2]);
    const interY2 = Math.min(left[3], right[3]);
    const interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
    const leftArea = Math.max(1, bboxArea(left));
    return interArea / leftArea;
  }

  function clusterRowsByColumns(
    rows: Array<{
      region_id: string;
      label: string;
      bbox_xyxy: [number, number, number, number];
      polygon?: [number, number][];
      specificity: number;
    }>,
    columnBoxes: Array<[number, number, number, number]>,
  ) {
    const assigned = new Map<number, typeof rows>();
    for (let index = 0; index < columnBoxes.length; index += 1) {
      assigned.set(index, []);
    }
    const overflow: typeof rows = [];
    for (const row of rows) {
      const centerX = (row.bbox_xyxy[0] + row.bbox_xyxy[2]) / 2;
      let bestIndex = -1;
      let bestScore = -1;
      for (let index = 0; index < columnBoxes.length; index += 1) {
        const columnBBox = columnBoxes[index];
        const containsCenter = columnBBox[0] <= centerX && centerX <= columnBBox[2];
        const overlap = bboxOverlapRatio(row.bbox_xyxy, columnBBox);
        const score = overlap + (containsCenter ? 0.25 : 0);
        if (score > bestScore) {
          bestScore = score;
          bestIndex = index;
        }
      }
      if (bestIndex >= 0 && bestScore > 0.05) {
        assigned.get(bestIndex)?.push(row);
      } else {
        overflow.push(row);
      }
    }
    const ordered: typeof rows = [];
    columnBoxes.forEach((_box, index) => {
      const columnRows = [...(assigned.get(index) ?? [])].sort((left, right) => {
        if (Math.abs(left.bbox_xyxy[1] - right.bbox_xyxy[1]) > 8) {
          return left.bbox_xyxy[1] - right.bbox_xyxy[1];
        }
        return left.bbox_xyxy[0] - right.bbox_xyxy[0];
      });
      ordered.push(...columnRows);
    });
    ordered.push(...overflow.sort((left, right) => {
      if (Math.abs(left.bbox_xyxy[1] - right.bbox_xyxy[1]) > 8) {
        return left.bbox_xyxy[1] - right.bbox_xyxy[1];
      }
      return left.bbox_xyxy[0] - right.bbox_xyxy[0];
    }));
    return ordered;
  }

  function clusterRowsWithoutColumns(
    rows: Array<{
      region_id: string;
      label: string;
      bbox_xyxy: [number, number, number, number];
      polygon?: [number, number][];
      specificity: number;
    }>,
  ) {
    const sortedRows = [...rows].sort((left, right) => {
      const leftCenter = (left.bbox_xyxy[0] + left.bbox_xyxy[2]) / 2;
      const rightCenter = (right.bbox_xyxy[0] + right.bbox_xyxy[2]) / 2;
      if (Math.abs(leftCenter - rightCenter) > 8) {
        return leftCenter - rightCenter;
      }
      return left.bbox_xyxy[1] - right.bbox_xyxy[1];
    });
    const pageWidth =
      Math.max(...sortedRows.map((row) => row.bbox_xyxy[2])) -
      Math.min(...sortedRows.map((row) => row.bbox_xyxy[0]));
    const mergeGap = Math.max(80, pageWidth * 0.08);
    const columns: Array<{
      x1: number;
      x2: number;
      centerX: number;
      rows: typeof rows;
    }> = [];
    for (const row of sortedRows) {
      const centerX = (row.bbox_xyxy[0] + row.bbox_xyxy[2]) / 2;
      let match: typeof columns[number] | null = null;
      let matchDistance: number | null = null;
      for (const column of columns) {
        const overlap = Math.min(row.bbox_xyxy[2], column.x2) - Math.max(row.bbox_xyxy[0], column.x1);
        const distance = overlap > 0 ? Math.abs(centerX - column.centerX) : Math.max(row.bbox_xyxy[0] - column.x2, column.x1 - row.bbox_xyxy[2], 0);
        if (distance <= mergeGap && (matchDistance === null || distance < matchDistance)) {
          match = column;
          matchDistance = distance;
        }
      }
      if (!match) {
        columns.push({
          x1: row.bbox_xyxy[0],
          x2: row.bbox_xyxy[2],
          centerX,
          rows: [row],
        });
        continue;
      }
      match.rows.push(row);
      match.x1 = Math.min(match.x1, row.bbox_xyxy[0]);
      match.x2 = Math.max(match.x2, row.bbox_xyxy[2]);
      match.centerX = match.rows.reduce((sum, item) => sum + (item.bbox_xyxy[0] + item.bbox_xyxy[2]) / 2, 0) / match.rows.length;
    }
    columns.sort((left, right) => left.x1 - right.x1);
    return columns.flatMap((column) =>
      [...column.rows].sort((left, right) => {
        if (Math.abs(left.bbox_xyxy[1] - right.bbox_xyxy[1]) > 8) {
          return left.bbox_xyxy[1] - right.bbox_xyxy[1];
        }
        return left.bbox_xyxy[0] - right.bbox_xyxy[0];
      }),
    );
  }

  const categories = Array.isArray(coco.categories) ? coco.categories : [];
  const annotations = Array.isArray(coco.annotations) ? coco.annotations : [];
  const categoryById = new Map<number, string>();
  for (const category of categories) {
    if (typeof category?.id === "number" && typeof category?.name === "string") {
      categoryById.set(category.id, category.name);
    }
  }

  const columnBoxes = annotations
    .map((annotation) => {
      if (!annotation || !Array.isArray(annotation.bbox) || annotation.bbox.length < 4) {
        return null;
      }
      const label = categoryById.get(Number(annotation.category_id)) || "";
      if (!isColumnLikeLabel(label)) {
        return null;
      }
      const [x, y, w, h] = annotation.bbox;
      if (![x, y, w, h].every((value) => Number.isFinite(value)) || w < 8 || h < 8) {
        return null;
      }
      return [x, y, x + w, y + h] as [number, number, number, number];
    })
    .filter((value): value is [number, number, number, number] => Boolean(value))
    .sort((left, right) => {
      if (Math.abs(left[0] - right[0]) > 8) {
        return left[0] - right[0];
      }
      return left[1] - right[1];
    });

  const candidates = annotations
    .map((annotation) => {
      if (!annotation || !Array.isArray(annotation.bbox) || annotation.bbox.length < 4) {
        return null;
      }
      const label = categoryById.get(Number(annotation.category_id)) || "";
      if (isColumnLikeLabel(label)) {
        return null;
      }
      if (!label || !isRelevantTextLabel(label)) {
        return null;
      }
      const [x, y, w, h] = annotation.bbox;
      if (![x, y, w, h].every((value) => Number.isFinite(value)) || w < 8 || h < 8) {
        return null;
      }
      return {
        region_id: String(annotation.id),
        label,
        bbox_xyxy: [x, y, x + w, y + h] as [number, number, number, number],
        polygon: bestAnnotationPolygon(annotation),
        specificity: regionSpecificity(label),
      };
    })
    .filter((value): value is {
      region_id: string;
      label: string;
      bbox_xyxy: [number, number, number, number];
      polygon?: [number, number][];
      specificity: number;
    } => Boolean(value));

  const preferredCandidates = candidates.filter((candidate) => isPreferredLineLabel(candidate.label));
  const workingSet = preferredCandidates.length ? preferredCandidates : candidates;
  const preSortedWorkingSet = [...workingSet].sort((left, right) => {
    if (left.specificity !== right.specificity) {
      return left.specificity - right.specificity;
    }
    const areaDelta = bboxArea(left.bbox_xyxy) - bboxArea(right.bbox_xyxy);
    if (Math.abs(areaDelta) > 1) {
      return areaDelta;
    }
    if (Math.abs(left.bbox_xyxy[1] - right.bbox_xyxy[1]) > 8) {
      return left.bbox_xyxy[1] - right.bbox_xyxy[1];
    }
    return left.bbox_xyxy[0] - right.bbox_xyxy[0];
  });

  const kept = preSortedWorkingSet.filter((candidate, index) => {
    for (let cursor = 0; cursor < index; cursor += 1) {
      const prior = preSortedWorkingSet[cursor];
      if (prior.specificity > candidate.specificity) {
        continue;
      }
      if (bboxCoverage(prior.bbox_xyxy, candidate.bbox_xyxy) >= 0.85) {
        return false;
      }
    }
    return true;
  });

  const ordered = columnBoxes.length
    ? clusterRowsByColumns(kept, columnBoxes)
    : clusterRowsWithoutColumns(kept);

  return ordered.map((region, index) => ({
    region_id: region.region_id,
    bbox_xyxy: region.bbox_xyxy,
    polygon: region.polygon,
    label: region.label,
    reading_order: index,
  }));
}

function toBase64(dataUrl: string): string {
  const index = dataUrl.indexOf(",");
  if (index === -1) {
    return dataUrl;
  }
  return dataUrl.slice(index + 1);
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

type WorkspaceIntent = "segment" | "extract" | "translate" | "entities" | "crop" | "label_analysis" | null;
type OcrPromptHints = {
  scriptHintSeed?: string;
  languageHint?: string;
  ocrBackend?: "auto" | "kraken_mccatmus" | "kraken_catmus" | "kraken_cremma_medieval" | "kraken_cremma_lat";
};
type ExtractionEngineCardId = "kraken" | "calamari" | "glmocr";
const EMPTY_DOCUMENT_METADATA: WorkspaceDocumentMetadata = {
  language: "",
  year: "",
  placeOrOrigin: "",
  scriptFamily: "",
  documentType: "",
  notes: "",
};

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
    /\bextract\b.*\btext\b/.test(value) ||
    /\bextract\b.*\b(manuscript|page|latin|french|english|old french|middle french|anglo norman|italian|spanish|iberian|portuguese|catalan)\b/.test(value) ||
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

function isLabelAnalysisPrompt(text: string): boolean {
  const value = normalizeLabelText(text);
  return /(what is this|what is that|explain|style|art style|origin|origins|motif|ornament|ornamental|decorative|decoration|shape|symbol|design|embellished|initial)/.test(value);
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

function extractOcrPromptHints(text: string): OcrPromptHints {
  const value = normalizeCommandText(text);
  if (/\banglo norman\b/.test(value)) {
    return { languageHint: "anglo_norman", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bold french\b/.test(value)) {
    return { languageHint: "old_french", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bmiddle french\b/.test(value)) {
    return { languageHint: "middle_french", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bmedieval french\b/.test(value)) {
    return { languageHint: "middle_french", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\blatin\b/.test(value)) {
    return { languageHint: "latin", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bmedieval latin\b/.test(value)) {
    return { languageHint: "latin", scriptHintSeed: "latin", ocrBackend: "auto" };
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
    return { languageHint: "middle_english", scriptHintSeed: "latin", ocrBackend: "auto" };
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

function metadataToOcrHints(metadata: WorkspaceDocumentMetadata | null | undefined): OcrPromptHints {
  if (!metadata) {
    return {};
  }

  const languageValue = normalizeCommandText(metadata.language || "");
  const scriptValue = normalizeCommandText(metadata.scriptFamily || "");
  const fallbackScriptHint = scriptValue.includes("insular") ? "insular_old_english" : "latin";

  if (/\banglo norman\b/.test(languageValue)) {
    return { languageHint: "anglo_norman", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bold french\b/.test(languageValue)) {
    return { languageHint: "old_french", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bmiddle french\b/.test(languageValue)) {
    return { languageHint: "middle_french", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bmedieval french\b/.test(languageValue)) {
    return { languageHint: "middle_french", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\blatin\b/.test(languageValue)) {
    return { languageHint: "latin", scriptHintSeed: fallbackScriptHint, ocrBackend: "auto" };
  }
  if (/\bmedieval latin\b/.test(languageValue)) {
    return { languageHint: "latin", scriptHintSeed: fallbackScriptHint, ocrBackend: "auto" };
  }
  if (/\bfrench\b/.test(languageValue)) {
    return { languageHint: "french", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\boccitan\b|\bold occitan\b|\bprovencal\b|\bprovençal\b/.test(languageValue)) {
    return { languageHint: "occitan", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bitalian\b/.test(languageValue)) {
    return { languageHint: "italian", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bspanish\b|\biberian\b|\bportuguese\b|\bcatalan\b/.test(languageValue)) {
    return { languageHint: "spanish", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bmiddle english\b/.test(languageValue)) {
    return { languageHint: "middle_english", scriptHintSeed: fallbackScriptHint, ocrBackend: "auto" };
  }
  if (/\bold english\b/.test(languageValue)) {
    return { languageHint: "old_english", scriptHintSeed: "insular_old_english", ocrBackend: "auto" };
  }
  if (/\bmiddle high german\b/.test(languageValue)) {
    return { languageHint: "middle_high_german", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bold high german\b/.test(languageValue)) {
    return { languageHint: "old_high_german", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bgerman\b|\bdutch\b|\bflemish\b/.test(languageValue)) {
    return { languageHint: "german", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  return scriptValue ? { scriptHintSeed: fallbackScriptHint, ocrBackend: "auto" } : {};
}

function isPrintLikeMetadata(metadata: WorkspaceDocumentMetadata | null | undefined): boolean {
  const scriptValue = normalizeCommandText(metadata?.scriptFamily || "");
  const typeValue = normalizeCommandText(metadata?.documentType || "");
  const notesValue = normalizeCommandText(metadata?.notes || "");
  return /(print|printed|imprint|fraktur|antiqua|humanistic|roman type)/.test(`${scriptValue} ${typeValue} ${notesValue}`);
}

function getExtractionEngineRecommendation(
  userPrompt: string,
  metadata: WorkspaceDocumentMetadata | null | undefined,
): {
  recommended: ExtractionEngineCardId;
  detectedLanguage: string;
  detectedScript: string;
  autoRecommendation: string;
} {
  const hints = resolveOcrPromptHints(userPrompt, metadata);
  const detectedLanguage = hints.languageHint || normalizeCommandText(metadata?.language || "") || "unknown";
  const detectedScript = hints.scriptHintSeed || normalizeCommandText(metadata?.scriptFamily || "") || "unknown";
  if (isPrintLikeMetadata(metadata)) {
    if (/(old_french|middle_french|anglo_norman|french)/.test(detectedLanguage)) {
      return {
        recommended: "calamari",
        detectedLanguage,
        detectedScript,
        autoRecommendation: "Calamari -> historical_french",
      };
    }
    if (/(german|old_high_german|middle_high_german|dutch|flemish)/.test(detectedLanguage) || /(fraktur|blackletter)/.test(detectedScript)) {
      return {
        recommended: "calamari",
        detectedLanguage,
        detectedScript,
        autoRecommendation: "Calamari -> fraktur_historical",
      };
    }
    if (/(latin|italian|spanish|portuguese|catalan|occitan)/.test(detectedLanguage) || /(antiqua|humanistic|roman)/.test(detectedScript)) {
      return {
        recommended: "calamari",
        detectedLanguage,
        detectedScript,
        autoRecommendation: "Calamari -> antiqua_historical / gt4histocr",
      };
    }
  }
  if (detectedLanguage === "latin") {
    return {
      recommended: "kraken",
      detectedLanguage,
      detectedScript,
      autoRecommendation: "Kraken family -> CREMMA-Medieval-LAT, then CATMuS, then McCATMuS",
    };
  }
  if (/(old_french|middle_french|anglo_norman|french)/.test(detectedLanguage)) {
    return {
      recommended: "kraken",
      detectedLanguage,
      detectedScript,
      autoRecommendation: "Kraken family -> CREMMA Medieval, then CATMuS, then McCATMuS",
    };
  }
  if (/(spanish|portuguese|catalan|iberian|italian|occitan)/.test(detectedLanguage)) {
    return {
      recommended: "kraken",
      detectedLanguage,
      detectedScript,
      autoRecommendation: "Kraken family -> CATMuS, then McCATMuS",
    };
  }
  if (/(german|old_high_german|middle_high_german|dutch|flemish|english|old_english|middle_english)/.test(detectedLanguage)) {
    return {
      recommended: "kraken",
      detectedLanguage,
      detectedScript,
      autoRecommendation: "Kraken family -> McCATMuS, then CATMuS",
    };
  }
  return {
    recommended: "kraken",
    detectedLanguage,
    detectedScript,
    autoRecommendation: "Kraken family -> CATMuS, then McCATMuS",
  };
}

function resolveOcrPromptHints(text: string, metadata: WorkspaceDocumentMetadata | null | undefined): OcrPromptHints {
  const promptHints = extractOcrPromptHints(text);
  if (promptHints.languageHint || promptHints.scriptHintSeed) {
    return promptHints;
  }
  return metadataToOcrHints(metadata);
}

function buildEnglishTranslationPrompt(sourceText: string): string {
  return [
    "Translate the following OCR-extracted manuscript text into English.",
    "This is a best-effort translation task.",
    "Never refuse, block, or gate the translation.",
    "Never answer with policy text such as 'the OCR is too uncertain' or similar.",
    "Return plain English only.",
    "Do not return JSON.",
    "Do not explain what you are doing.",
    "If parts are uncertain, translate what you can and preserve uncertainty inline with markers like [unclear] or [?].",
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
  const [pendingUpload, setPendingUpload] = useState<PendingUploadDocument | null>(null);
  const [metadataDraft, setMetadataDraft] = useState<WorkspaceDocumentMetadata>(EMPTY_DOCUMENT_METADATA);
  const [metadataError, setMetadataError] = useState<string | null>(null);
  const [showEngineSelector, setShowEngineSelector] = useState(false);
  const [pendingExtractionOptions, setPendingExtractionOptions] = useState<{
    userPrompt: string;
    compareBackends?: ("calamari" | "glmocr")[];
    ocrBackend?: "auto" | "calamari" | "glmocr";
  } | null>(null);
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
  const currentMessages = currentDocument ? (messagesByDocument[currentDocument.id] ?? []) : [];
  const currentSegmentedPreview = currentPage ? (segmentedPreviewByPageId[currentPage.id] ?? null) : null;
  const currentSegmentationCoco = currentPage ? (segmentationCocoByPageId[currentPage.id] ?? null) : null;
  const currentSegmentationError = currentPage ? (segmentationErrorByPageId[currentPage.id] ?? null) : null;
  const currentExtractedText = currentPage ? (ocrTextByPageId[currentPage.id] ?? "") : "";
  const currentOcrRunId = currentPage ? (ocrRunIdByPageId[currentPage.id] ?? "") : "";
  const currentAuthorityReport = currentPage ? (authorityReportByPageId[currentPage.id] ?? "") : "";
  const currentPageIsSegmenting = Boolean(currentPage && segmentingPageId === currentPage.id);
  const currentMetadata = currentDocument?.metadata ?? null;
  const extractionEngineRecommendation = getExtractionEngineRecommendation(
    pendingExtractionOptions?.userPrompt || "",
    currentMetadata,
  );

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
    const matchedLabel = resolveCropLabelFromPrompt(text, currentSegmentationCoco);

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
      setPendingExtractionOptions({ userPrompt: text });
      setShowEngineSelector(true);
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

    try {
      let coco = currentSegmentationCoco;
      if (!coco) {
        setLabelStatus("Label analysis status: running segmentation first...");
        const segmentation = await runSegmentationForCurrentPage();
        coco = segmentation?.coco ?? null;
      }

      if (!coco) {
        setLabelStatus("Label analysis failed: segmentation data is unavailable.");
        return;
      }

      const label = resolveCropLabelFromPrompt(userText, coco);
      if (!label) {
        const labels = availableCropLabels(coco);
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

      setLabelStatus(`Label analysis status: cropping "${label}" and sending it to the model...`);
      const cropped = await buildTransparentCropOverlay(currentPage.dataUrl, coco, label);
      const prompt = [
        "Answer the user's question about the attached cropped manuscript label image.",
        "The attached image contains only the requested label in its original page position.",
        "Label name:",
        label,
        "",
        "User request:",
        userText,
        "",
        "Focus on visible form, ornament, likely art style, decorative function, and plausible historical/manuscript context.",
        "Do not mention internal pipeline steps.",
      ].join("\n");

      await sendPromptToChat(prompt, {
        displayText: userText,
        attachImage: true,
        forcedImageDataUrl: cropped.imageUrl,
        loadingLabel: "Analyzing label",
        historyForModel: [],
      });
      setLabelStatus(`Label analysis complete for "${label}".`);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Label analysis request failed.";
      setError(message);
      setLabelStatus(`Label analysis failed: ${message}`);
    } finally {
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

  const handleExtractTextInChat = async (options?: {
    includeDebugOutput?: boolean;
    silent?: boolean;
    userPrompt?: string;
    ocrBackend?: "auto" | "calamari" | "glmocr";
    compareBackends?: ("calamari" | "glmocr")[];
  }): Promise<{ text: string; runId: string; authorityReport: string } | null> => {
    if (!currentPage || !currentDocument) {
      return null;
    }
    const includeDebugOutput = options?.includeDebugOutput ?? true;
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
      setExtractionStatus(
        options?.compareBackends?.length
          ? "Extraction status: running segmentation-guided OCR comparison..."
          : "Extraction status: running segmentation-guided OCR..."
      );

      let locationSuggestions = extractLocationSuggestions(currentSegmentationCoco);
      let structuredRegions = extractStructuredRegions(currentSegmentationCoco);
      if (!locationSuggestions.length) {
        const segmentation = await runSegmentationForCurrentPage();
        locationSuggestions = extractLocationSuggestions(segmentation?.coco ?? null);
        structuredRegions = extractStructuredRegions(segmentation?.coco ?? null);
      }

      const ocrHints = resolveOcrPromptHints(userPrompt, currentMetadata);
      const response = await extractWithSaiaOcr({
        document_id: currentDocumentId,
        image_id: currentDocumentId,
        page_id: currentPage.id,
        image_b64: toBase64(currentPage.dataUrl),
        script_hint_seed: ocrHints.scriptHintSeed,
        language_hint: ocrHints.languageHint,
        ocr_backend: (options?.ocrBackend as OCRExtractRequestPayload["ocr_backend"]) ?? "auto",
        compare_backends: (options?.compareBackends as OCRExtractRequestPayload["compare_backends"]) ?? [],
        location_suggestions: locationSuggestions,
        regions: structuredRegions,
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

      const rawExtractedText = mergeOCRResultText(response);
      const finalText = removeTextUncertainties(rawExtractedText);
      const comparisonText = (response.comparison_runs ?? [])
        .map((run) => {
          const cleaned = removeTextUncertainties(String(run.text || "").trim());
          const warnings = (run.warnings ?? []).filter(Boolean);
          const notes = (run.notes ?? []).filter(Boolean);
          const confidence = typeof run.confidence === "number" ? ` confidence=${run.confidence.toFixed(3)}` : "";
          const hintText = [run.language_hint ? `language=${run.language_hint}` : "", run.script_family ? `script=${run.script_family}` : ""]
            .filter(Boolean)
            .join(" ");
          const warningText = warnings.length ? `\nWarnings: ${warnings.join("; ")}` : "";
          const notesText = notes.length ? `\nNotes: ${notes.join("; ")}` : "";
          return [
            `${run.selected ? "Selected" : "Compared"} OCR: ${run.backend_name} (${run.model_name})${confidence}${hintText ? ` ${hintText}` : ""}`,
            cleaned || "(no readable text detected)",
            warningText,
            notesText,
          ].filter(Boolean).join("\n");
        })
        .join("\n\n---\n\n")
        .trim();
      const fallbackStateText = (response.comparison_runs ?? [])
        .map((run) => removeTextUncertainties(String(run.text || "").trim()))
        .find((value) => Boolean(value));
      const storedText = finalText || fallbackStateText || "";
      const hasComparisonText = Boolean(comparisonText);
      if (!finalText && !hasComparisonText) {
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
      setOcrRunIdByPageId((prev) => ({ ...prev, [currentPage.id]: "" }));
      setAuthorityReportByPageId((prev) => ({ ...prev, [currentPage.id]: "" }));

      if (!silent) {
        setMessagesByDocument((prev) => ({
          ...prev,
          [currentDocumentId]: (prev[currentDocumentId] ?? []).map((msg) =>
            msg.id === assistantMessageId
              ? { ...msg, content: (options?.compareBackends?.length ? comparisonText : finalText) || finalText }
              : msg,
          ),
        }));
      }
      const status = getExtractionStatus(response);
      setExtractionStatus(
        options?.compareBackends?.length
          ? `Extraction comparison complete (${status}).`
          : `Extraction complete (${status}).`
      );
      return { text: storedText, runId: "", authorityReport: "" };
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
      {showEngineSelector && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4">
          <div className="w-full max-w-3xl rounded-xl border bg-background p-6 shadow-2xl">
            <h3 className="text-lg font-semibold mb-1">Select OCR Engine</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Choose which recognition engine to use for text extraction. Detected language/script hint:
              {` ${extractionEngineRecommendation.detectedLanguage || "unknown"}`}
              {extractionEngineRecommendation.detectedScript && extractionEngineRecommendation.detectedScript !== "unknown"
                ? ` / ${extractionEngineRecommendation.detectedScript}`
                : ""}
              . Auto recommendation: {extractionEngineRecommendation.autoRecommendation}.
            </p>
            <div className="grid gap-3 md:grid-cols-3">
              <button
                type="button"
                onClick={() => {
                  setShowEngineSelector(false);
                  handleExtractTextInChat({ ...pendingExtractionOptions, ocrBackend: "auto", compareBackends: [] });
                  setPendingExtractionOptions(null);
                }}
                className={`rounded-lg border p-4 hover:bg-muted transition text-left ${
                  extractionEngineRecommendation.recommended === "kraken" ? "border-primary ring-2 ring-primary/30" : ""
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="font-medium">Kraken family</div>
                  <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary">Recommended</span>
                </div>
                <div className="text-xs text-muted-foreground mt-2">
                  Best manuscript OCR path in this repo using segmented line crops and medieval-trained models.
                </div>
                <div className="text-xs mt-2"><span className="font-medium">Best for:</span> Latin, Old French, Middle French, Anglo-Norman</div>
                <div className="text-xs mt-1"><span className="font-medium">Also good for:</span> Spanish/Iberian, Italian</div>
                <div className="text-xs mt-2 text-muted-foreground">
                  Best choice for medieval handwritten Latin-script material. Uses segmentation-driven line OCR.
                </div>
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowEngineSelector(false);
                  handleExtractTextInChat({ ...pendingExtractionOptions, ocrBackend: "calamari", compareBackends: [] });
                  setPendingExtractionOptions(null);
                }}
                className={`rounded-lg border p-4 hover:bg-muted transition text-left ${
                  extractionEngineRecommendation.recommended === "calamari" ? "border-primary ring-2 ring-primary/30" : ""
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="font-medium">Calamari</div>
                  <span className="rounded-full bg-muted px-2 py-0.5 text-[11px] font-medium text-muted-foreground">Historical print baseline</span>
                </div>
                <div className="text-xs text-muted-foreground mt-2">
                  Line OCR backend best suited to historical print and broad fallback experiments.
                </div>
                <div className="text-xs mt-2"><span className="font-medium">Best for:</span> historical printed German/Fraktur, historical printed Latin, later historical printed French</div>
                <div className="text-xs mt-2 text-muted-foreground">
                  Not the best default for medieval handwritten manuscripts. Useful for comparison and print-like material.
                </div>
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowEngineSelector(false);
                  handleExtractTextInChat({ ...pendingExtractionOptions, ocrBackend: "glmocr", compareBackends: [] });
                  setPendingExtractionOptions(null);
                }}
                className="rounded-lg border p-4 hover:bg-muted transition text-left"
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="font-medium">GLM-OCR</div>
                  <span className="rounded-full bg-amber-500/10 px-2 py-0.5 text-[11px] font-medium text-amber-700">Experimental</span>
                </div>
                <div className="text-xs text-muted-foreground mt-2">
                  General multimodal OCR for full-page or column-level document extraction.
                </div>
                <div className="text-xs mt-2"><span className="font-medium">Best for:</span> broad multilingual documents, complex modern layouts, page/column OCR experiments</div>
                <div className="text-xs mt-2 text-muted-foreground">
                  Not specialized for medieval manuscript line transcription.
                </div>
              </button>
            </div>
            <button
              type="button"
              onClick={() => {
                setShowEngineSelector(false);
                handleExtractTextInChat({ ...pendingExtractionOptions, ocrBackend: "auto", compareBackends: ["calamari", "glmocr"] });
                setPendingExtractionOptions(null);
              }}
              className="mt-4 w-full rounded-lg border p-4 text-left hover:bg-muted transition"
            >
              <div className="font-medium">Compare all three</div>
              <div className="text-xs text-muted-foreground mt-1">Run Kraken family, Calamari, and GLM-OCR together and return the three assembled transcriptions with backend, model, warnings, and confidence metadata.</div>
            </button>
            <button
              type="button"
              onClick={() => { setShowEngineSelector(false); setPendingExtractionOptions(null); }}
              className="mt-4 text-sm text-muted-foreground hover:underline"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
