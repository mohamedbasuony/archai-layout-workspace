/**
 * Authority Linking API client — entity linking report calls.
 */

import { apiFetch } from "./client";

/* ── Types ─────────────────────────────────────────────────────────── */

export interface LinkingReportResult {
  run_id: string;
  asset_ref: string;
  mentions_total: number;
  type_counts: Record<string, number>;
  candidates_total: number;
  source_counts: Record<string, number>;
  linked_total: number;
  unresolved_total: number;
  ambiguous_total: number;
  api_calls: number;
  cache_hits: number;
  took_ms: number;
  mention_results: LinkingMentionResult[];
  report?: string;
}

export interface LinkingMentionResult {
  mention_id: string;
  surface: string;
  ent_type: string;
  chunk_id: string | null;
  start_offset: number;
  end_offset: number;
  evidence_text: string;
  status: "linked" | "ambiguous" | "unresolved";
  reason: string;
  selected: {
    qid: string;
    label: string;
    description: string;
    score: number;
    viaf_id: string;
    geonames_id: string;
  } | null;
  top_candidates: {
    qid: string;
    label: string;
    score: number;
  }[];
}

/* ── API calls ─────────────────────────────────────────────────────── */

export async function getLinkingReport(
  runId: string,
): Promise<LinkingReportResult> {
  return apiFetch<LinkingReportResult>(
    `/authority/report/${encodeURIComponent(runId)}`,
    { method: "GET" },
  );
}
