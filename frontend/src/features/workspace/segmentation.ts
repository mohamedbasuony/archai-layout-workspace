import { TEXT_LABEL_EXCLUDE_TOKENS, TEXT_LABEL_INCLUDE_TOKENS } from "./constants";
import { normalizeLabelText } from "./intents";
import type { CocoAnnotation, CocoPayload, OCRLocationSuggestion, StoredLabelRegionMap } from "./types";

export function extractLocationSuggestions(coco: CocoPayload | null | undefined): OCRLocationSuggestion[] {
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
    const [x, y, width, height] = annotation.bbox;
    if (![x, y, width, height].every((value) => Number.isFinite(value))) {
      continue;
    }
    if (width < 8 || height < 8) {
      continue;
    }
    suggestions.push({
      region_id: String(annotation.id),
      category,
      bbox_xywh: [x, y, width, height],
    });
  }

  suggestions.sort((first, second) => {
    const [firstX, firstY] = first.bbox_xywh;
    const [secondX, secondY] = second.bbox_xywh;
    return Math.abs(firstY - secondY) > 8 ? firstY - secondY : firstX - secondX;
  });

  return suggestions.slice(0, 60);
}

export function buildSegmentationSummary(coco: CocoPayload | null | undefined): string {
  if (!coco) {
    return "Segmentation completed, but no COCO payload was returned.";
  }

  const annotations = Array.isArray(coco.annotations) ? coco.annotations : [];
  const categoryById = cocoCategoryMap(coco);
  const counts = new Map<string, number>();
  for (const annotation of annotations) {
    const label = categoryById.get(Number(annotation?.category_id)) || `category_${String(annotation?.category_id ?? "unknown")}`;
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }

  const lines = Array.from(counts.entries())
    .sort((first, second) => second[1] - first[1] || first[0].localeCompare(second[0]))
    .map(([label, count]) => `- ${label}: ${count}`);

  return [
    "Segmentation complete.",
    "",
    `Total regions: ${annotations.length}`,
    `Text-like regions: ${extractLocationSuggestions(coco).length}`,
    "",
    "Discovered labels:",
    ...(lines.length ? lines : ["- none"]),
  ].join("\n");
}

export function bestMatchingSegmentationLabel(text: string, labels: string[]): string | null {
  const prompt = normalizeLabelText(text);
  let bestLabel: string | null = null;
  let bestScore = 0;

  for (const label of labels) {
    const normalized = normalizeLabelText(label);
    let score = prompt.includes(normalized) ? 100 : 0;
    for (const token of normalized.split(" ").filter((token) => token.length >= 3)) {
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

export function buildStoredLabelRegions(coco: CocoPayload | null | undefined): StoredLabelRegionMap {
  const labelsById = cocoCategoryMap(coco);
  const regionsByLabel: StoredLabelRegionMap = {};
  for (const annotation of coco?.annotations || []) {
    if (!annotation || !Array.isArray(annotation.bbox) || annotation.bbox.length < 4) {
      continue;
    }
    const label = labelsById.get(Number(annotation.category_id));
    if (!label) {
      continue;
    }
    const [x, y, width, height] = annotation.bbox;
    if (![x, y, width, height].every((value) => Number.isFinite(value))) {
      continue;
    }
    const regions = regionsByLabel[label] || [];
    regions.push({
      region_id: String(annotation.id),
      bbox_xyxy: [x, y, x + width, y + height],
      polygons: annotationPolygons(annotation),
    });
    regionsByLabel[label] = regions;
  }
  return regionsByLabel;
}

export function availableStoredLabels(labelsByName: StoredLabelRegionMap | null | undefined): string[] {
  return Object.keys(labelsByName || {}).sort((first, second) => first.localeCompare(second));
}

export function resolveCropLabelFromPrompt(
  text: string,
  labelsByName: StoredLabelRegionMap | null | undefined,
): string | null {
  return bestMatchingSegmentationLabel(text, availableStoredLabels(labelsByName));
}

export async function buildTransparentCropOverlay(
  pageDataUrl: string,
  labelsByName: StoredLabelRegionMap,
  label: string,
): Promise<{ imageUrl: string; matchCount: number }> {
  const matches = labelsByName[label] || [];
  if (!matches.length) {
    throw new Error(`No regions found for label: ${label}`);
  }

  const source = await loadImageElement(pageDataUrl);
  const canvas = document.createElement("canvas");
  canvas.width = source.naturalWidth || source.width;
  canvas.height = source.naturalHeight || source.height;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Failed to create crop canvas.");
  }

  context.clearRect(0, 0, canvas.width, canvas.height);
  for (const annotation of matches) {
    const polygons = annotation.polygons || [];
    if (polygons.length) {
      for (const polygon of polygons) {
        context.save();
        context.beginPath();
        for (let index = 0; index < polygon.length; index += 2) {
          const x = polygon[index] ?? 0;
          const y = polygon[index + 1] ?? 0;
          if (index === 0) {
            context.moveTo(x, y);
          } else {
            context.lineTo(x, y);
          }
        }
        context.closePath();
        context.clip();
        context.drawImage(source, 0, 0, canvas.width, canvas.height);
        context.restore();
      }
      continue;
    }

    const [x1, y1, x2, y2] = annotation.bbox_xyxy;
    context.save();
    context.beginPath();
    context.rect(x1, y1, x2 - x1, y2 - y1);
    context.clip();
    context.drawImage(source, 0, 0, canvas.width, canvas.height);
    context.restore();
  }

  return { imageUrl: canvas.toDataURL("image/png"), matchCount: matches.length };
}

function isRelevantTextLabel(label: string): boolean {
  const key = label.toLowerCase();
  return !TEXT_LABEL_EXCLUDE_TOKENS.some((token) => key.includes(token))
    && TEXT_LABEL_INCLUDE_TOKENS.some((token) => key.includes(token));
}

function cocoCategoryMap(coco: CocoPayload | null | undefined): Map<number, string> {
  const labels = new Map<number, string>();
  for (const category of coco?.categories || []) {
    if (typeof category?.id === "number" && typeof category?.name === "string") {
      labels.set(category.id, category.name);
    }
  }
  return labels;
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
    return (raw as number[][]).filter((polygon) => Array.isArray(polygon) && polygon.length >= 6);
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
