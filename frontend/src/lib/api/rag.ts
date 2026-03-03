/**
 * RAG API client – index & debug retrieval calls.
 *
 * All functions follow the same `apiFetch` pattern used by the rest of the
 * frontend (CORS-safe, proxy-fallback).
 */

import { apiFetch } from "./client";

/* ── Types ─────────────────────────────────────────────────────────── */

export interface RagIndexRunResult {
  run_id: string;
  asset_ref: string;
  chunks_total: number;
  chunks_indexed: number;
  chunks_skipped: number;
  collection_name: string;
  collection_count_after: number;
  took_ms: number;
  status: string;
}

export interface RagIndexStatus {
  run_id: string;
  asset_ref: string;
  chunks_total: number;
  chunks_indexed: number;
  indexed: boolean;
  missing_chunk_ids: string[];
}

export interface RagRetrieveHit {
  chunk_id: string;
  run_id: string;
  asset_ref: string;
  chunk_idx: number;
  offsets: string;
  score: number;
  text_preview: string;
}

export interface RagRetrieveResult {
  query: string;
  k: number;
  filter: { run_id: string | null; asset_ref: string | null };
  results: RagRetrieveHit[];
}

export interface RagEvidenceBlock {
  run_id: string;
  asset_ref: string;
  chunk_id: string;
  chunk_idx: number;
  offsets: string;
  text: string;
}

export interface RagEvidencePreview {
  query: string;
  k: number;
  hits: number;
  evidence_ids: { chunk_id: string; chunk_idx: number; offsets: string }[];
  evidence_blocks: RagEvidenceBlock[];
  citation_example: string;
  evidence_text: string;
  rag_instruction: string;
}

/* ── API calls ─────────────────────────────────────────────────────── */

export async function postIndexRun(runId: string): Promise<RagIndexRunResult> {
  return apiFetch<RagIndexRunResult>(`/index/run/${encodeURIComponent(runId)}`, {
    method: "POST",
  });
}

export async function getIndexStatus(runId: string): Promise<RagIndexStatus> {
  return apiFetch<RagIndexStatus>(`/index/run/${encodeURIComponent(runId)}/status`, {
    method: "GET",
  });
}

export async function getRetrieveDebug(
  query: string,
  k: number,
  runId?: string,
): Promise<RagRetrieveResult> {
  const params = new URLSearchParams({ query, k: String(k) });
  if (runId) params.set("run_id", runId);
  return apiFetch<RagRetrieveResult>(`/rag/debug/retrieve?${params.toString()}`, {
    method: "GET",
  });
}

export async function getEvidencePreview(
  query: string,
  runId: string,
  k: number = 8,
): Promise<RagEvidencePreview> {
  const params = new URLSearchParams({ query, run_id: runId, k: String(k) });
  return apiFetch<RagEvidencePreview>(`/rag/debug/evidence-preview?${params.toString()}`, {
    method: "GET",
  });
}

/* ── Debug-flag helper ─────────────────────────────────────────────── */

/**
 * Returns `true` when `NEXT_PUBLIC_ARCHAI_DEBUG_RAG` is a truthy value
 * ("1", "true", "yes") **or** when the backend has set `ARCHAI_DEBUG_RAG`
 * and the browser URL contains `?rag_debug=1`.
 *
 * This keeps the debug blocks invisible for normal users.
 */
export function isRagDebugEnabled(): boolean {
  // Build-time env (Next.js)
  const envVal = (
    typeof process !== "undefined"
      ? process.env?.NEXT_PUBLIC_ARCHAI_DEBUG_RAG ?? ""
      : ""
  ).trim().toLowerCase();
  if (["1", "true", "yes"].includes(envVal)) return true;

  // Runtime URL param fallback (handy for ad-hoc testing)
  if (typeof window !== "undefined") {
    const params = new URLSearchParams(window.location.search);
    const urlVal = (params.get("rag_debug") ?? "").trim().toLowerCase();
    if (["1", "true", "yes"].includes(urlVal)) return true;
  }

  return false;
}
