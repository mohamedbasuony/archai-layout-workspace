import { apiUrl } from "./client";

export interface PredictSingleResponse {
  task_id: string;
  coco_json: Record<string, unknown>;
  stats: Record<string, number>;
  annotated_image_url: string;
}

function parseErrorText(status: number, raw: string): string {
  try {
    const parsed = JSON.parse(raw) as { detail?: string };
    if (typeof parsed.detail === "string" && parsed.detail.trim()) {
      return `HTTP ${status}: ${parsed.detail}`;
    }
  } catch {
    // ignore parse errors
  }
  return `HTTP ${status}: ${raw || "Request failed"}`;
}

export async function predictSinglePage(
  image: File,
  confidence = 0.25,
  iou = 0.3,
): Promise<PredictSingleResponse> {
  const form = new FormData();
  form.append("image", image);
  form.append("confidence", String(confidence));
  form.append("iou", String(iou));

  const response = await fetch(apiUrl("/predict/single"), {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    const raw = await response.text();
    throw new Error(parseErrorText(response.status, raw));
  }

  return (await response.json()) as PredictSingleResponse;
}
