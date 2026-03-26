"""Authority Linking orchestrator (v2 — precision-first).

Reads entity_mentions for a run, queries Wikidata for candidates,
scores/disambiguates with hard type gates, enriches with VIAF/GeoNames,
and persists results back into entity_candidates (with ``is_selected``
semantics via ``meta_json``).

**Key rules (v2):**

* ``type_compatible`` from wikidata_client.is_type_compatible() is a
  **hard gate** — candidates that fail can never be auto-selected.
* ``AUTO_SELECT_THRESHOLD = 0.75`` and ``MIN_MARGIN = 0.10`` from
  entity_scoring.py are enforced.
* Gate D in the validation report flags type-mismatch detections.
* VIAF / GeoNames IDs are only surfaced when backed by a Wikidata
  property value (P214 / P1566).
* Separate ``api_calls_search`` and ``api_calls_get`` counters.

Usage::

    from app.services.authority_linking import run_authority_linking
    result = run_authority_linking(run_id)
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from app.db import pipeline_db
from app.services.entity_scoring import (
    context_similarity,
    compute_score,
    disambiguate,
    get_thresholds,
    rescore_with_canonical,
    string_similarity,
)
from app.services.authority_sources import search_geonames, search_viaf
from app.services.wikidata_client import (
    enrich_wikidata_item,
    is_type_compatible,
    search_wikidata,
    cache_check,
    cache_get,
    normalise_surface,
)
from app.services.text_normalization import (
    normalize_for_search,
    text_quality_label,
    token_quality_score,
)

log = logging.getLogger("archai.authority_linking")


def _fetch_viaf_profile(viaf_id: str) -> dict[str, Any]:
    viaf_id = str(viaf_id or "").strip()
    if not viaf_id:
        return {"aliases": [], "source_url": ""}

    import urllib.error
    import urllib.request

    source_url = f"https://viaf.org/viaf/{viaf_id}/viaf.json"
    request = urllib.request.Request(source_url, headers={"User-Agent": "Archai-OCR-Pipeline/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        log.debug("VIAF HTTP error for %s: %s", viaf_id, exc)
        return {"aliases": [], "source_url": source_url}
    except Exception as exc:  # noqa: BLE001
        log.debug("VIAF lookup failed for %s: %s", viaf_id, exc)
        return {"aliases": [], "source_url": source_url}

    aliases: list[dict[str, str]] = []
    seen: set[str] = set()
    main_headings = payload.get("mainHeadings", {}).get("data", [])
    if isinstance(main_headings, dict):
        main_headings = [main_headings]
    for item in main_headings if isinstance(main_headings, list) else []:
        if not isinstance(item, dict):
            continue
        value = str(item.get("text") or "").strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        aliases.append({"lang": "", "value": value})

    return {
        "aliases": aliases,
        "source_url": source_url,
        "name_type": str(payload.get("nameType") or "").strip(),
        "titles": payload.get("titles"),
    }


def _clear_structured_links_for_run(run_id: str) -> None:
    pipeline_db._init_db_if_needed()
    with pipeline_db._connect() as conn:
        conn.execute(
            """
            DELETE FROM mention_links
            WHERE mention_id IN (
                SELECT mention_id FROM entity_mentions WHERE run_id=?
            )
            """,
            (run_id,),
        )
        conn.execute(
            """
            DELETE FROM evidence_spans
            WHERE run_id=? AND mention_id IS NOT NULL
            """,
            (run_id,),
        )
        conn.commit()


def _score_breakdown_for_mention(
    *,
    mention: dict[str, Any],
    query_details: list[dict[str, Any]],
    name_likeness: float,
    canonical_match: dict[str, Any] | None,
    selected: dict[str, Any] | None,
    top_candidates: list[dict[str, Any]],
    ocr_quality: str,
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "mention_surface": str(mention.get("surface") or ""),
        "mention_confidence": float(mention.get("confidence", 0.0)),
        "ent_type": str(mention.get("ent_type") or "unknown"),
        "status": status,
        "reason": reason,
        "ocr_quality": ocr_quality,
        "name_likeness": name_likeness,
        "canonical_match": canonical_match,
        "query_details": query_details,
        "selected_candidate": selected,
        "top_candidates": top_candidates,
    }


def _persist_mention_link(
    *,
    run_id: str,
    asset_ref: str,
    mention: dict[str, Any],
    evidence_text: str,
    evidence_span_id: str | None,
    status: str,
    reason: str,
    score_breakdown: dict[str, Any],
    selected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entity_id = _selected_entity_id(selected)
    return pipeline_db.upsert_mention_link(
        {
            "mention_id": mention.get("mention_id"),
            "entity_id": entity_id,
            "confidence": selected.get("score") if selected else mention.get("confidence"),
            "link_status": status,
            "selected_by": "authority_linking.v3",
            "type_compatible": selected.get("type_compatible") if selected else None,
            "score_breakdown_json": score_breakdown,
            "evidence_span_id": evidence_span_id,
            "reason": reason,
        }
    )


def _is_unresolved_status(status: str) -> bool:
    return str(status or "").strip().lower().startswith("unresolved")


def persist_unresolved_mentions(
    run_id: str,
    *,
    reason: str,
    status: str = "unresolved_low_quality",
    selected_by: str = "authority_linking.deferred",
) -> dict[str, Any]:
    run = pipeline_db.get_run(run_id)
    base_text = str(run.get("proofread_text") or run.get("ocr_text") or "") if run else ""
    asset_ref = str(run.get("asset_ref") or "") if run else ""
    mentions = pipeline_db.list_entity_mentions(run_id)
    if not mentions:
        empty = _empty_result(run_id, "no mentions found")
        empty["_base_text"] = base_text
        empty["asset_ref"] = asset_ref
        return empty

    ocr_quality = text_quality_label(base_text) if base_text else "LOW"
    mention_results: list[dict[str, Any]] = []
    type_counts = Counter(str(mention.get("ent_type", "unknown")) for mention in mentions)

    for mention in mentions:
        mention_id = str(mention.get("mention_id") or "")
        surface = str(mention.get("surface") or "")
        ent_type = str(mention.get("ent_type") or "unknown")
        chunk_id = mention.get("chunk_id")
        start_off = int(mention.get("start_offset", 0))
        end_off = int(mention.get("end_offset", 0))
        evidence_text = base_text[start_off:end_off] if base_text and end_off <= len(base_text) else surface
        evidence_context = _mention_context_window(base_text, start_off, end_off) if base_text else evidence_text
        mention_reason = _deferred_reason_for_mention(surface, ent_type, evidence_context or evidence_text, reason)
        evidence_span = pipeline_db.upsert_evidence_span(
            {
                "run_id": run_id,
                "asset_ref": asset_ref,
                "page_id": asset_ref or None,
                "chunk_id": chunk_id,
                "mention_id": mention_id,
                "start_offset": start_off,
                "end_offset": end_off,
                "raw_text": evidence_text,
                "normalized_text": normalize_for_search(evidence_text) if evidence_text else "",
                "meta_json": {
                    "source": "authority_linking.deferred",
                    "ent_type": ent_type,
                    "surface": surface,
                    "reason": mention_reason,
                },
            }
        )
        score_breakdown = _score_breakdown_for_mention(
            mention=mention,
            query_details=[],
            name_likeness=0.0,
            canonical_match=None,
            selected=None,
            top_candidates=[],
            ocr_quality=ocr_quality,
            status=status,
            reason=mention_reason,
        )
        pipeline_db.upsert_mention_link(
            {
                "mention_id": mention_id,
                "entity_id": None,
                "confidence": min(float(mention.get("confidence", 0.0)), 0.35),
                "link_status": status,
                "selected_by": selected_by,
                "type_compatible": None,
                "score_breakdown_json": score_breakdown,
                "evidence_span_id": evidence_span.get("span_id"),
                "reason": mention_reason,
            }
        )
        mention_results.append(
            {
                "mention_id": mention_id,
                "surface": surface,
                "ent_type": ent_type,
                "chunk_id": chunk_id,
                "start_offset": start_off,
                "end_offset": end_off,
                "evidence_text": evidence_text,
                "queries_attempted": [],
                "query_details": [],
                "status": status,
                "reason": mention_reason,
                "name_likeness": None,
                "canonical_match": None,
                "selected": None,
                "top_candidates": [],
            }
        )

    return {
        "run_id": run_id,
        "asset_ref": asset_ref,
        "mentions_total": len(mentions),
        "type_counts": dict(type_counts),
        "candidates_total": 0,
        "source_counts": {},
        "linked_total": 0,
        "unresolved_total": len(mentions),
        "ambiguous_total": 0,
        "skipped_total": 0,
        "quality_skipped": len(mentions),
        "canonical_matched": 0,
        "type_mismatch_count": 0,
        "api_calls_search": 0,
        "api_calls_viaf": 0,
        "api_calls_geonames": 0,
        "api_calls_get": 0,
        "api_calls": 0,
        "cache_hits": 0,
        "took_ms": 0,
        "ocr_quality": ocr_quality,
        "mention_results": mention_results,
        "_base_text": base_text,
    }


def _selected_entity_id(selected: dict[str, Any] | None) -> str | None:
    if not selected:
        return None
    source = str(selected.get("source") or ("wikidata" if selected.get("qid") else "")).strip()
    authority_id = str(
        selected.get("authority_id")
        or selected.get("qid")
        or selected.get("viaf_id")
        or selected.get("geonames_id")
        or ""
    ).strip()
    if not source or not authority_id:
        return None
    return f"{source}:{authority_id}"


def _score_candidate(
    *,
    mention: dict[str, Any],
    candidate: dict[str, Any],
    context: str,
    canonical_norm: str,
    domain_bonus: float,
    type_ok: bool,
    resolved_tokens: set[str],
) -> tuple[float, dict[str, Any]]:
    surface = str(mention.get("surface") or "")
    candidate_label = str(candidate.get("label") or "")
    compare_surface = canonical_norm or surface
    label_similarity = string_similarity(compare_surface, candidate_label)

    aliases_raw = list(candidate.get("aliases", []) or [])
    alias_values = [str(item.get("value") or "").strip() for item in aliases_raw if isinstance(item, dict)]
    alias_values = [value for value in alias_values if value]
    alias_match_quality = max((string_similarity(compare_surface, alias) for alias in alias_values), default=0.0)

    description_parts = [
        str(candidate.get("description") or ""),
        str(candidate.get("canonical_description") or ""),
        str(candidate.get("parent_location") or ""),
        " ".join(str(item) for item in list(candidate.get("titles") or [])[:5]),
    ]
    description_text = " ".join(part for part in description_parts if part).strip()
    document_context_compatibility = context_similarity(context, description_text)

    source = str(candidate.get("source") or "wikidata").strip() or "wikidata"
    source_confidence = float(candidate.get("source_confidence") or {"wikidata": 1.0, "viaf": 0.92, "geonames": 0.95}.get(source, 0.8))

    mention_label = str(mention.get("label") or "").lower()
    segmentation_label_prior = 0.0
    if source == "geonames" and mention.get("ent_type") == "place":
        segmentation_label_prior = 0.04
    elif source == "viaf" and mention.get("ent_type") in {"person", "work"}:
        segmentation_label_prior = 0.03
    elif mention_label.startswith("person") or mention_label.startswith("place") or mention_label.startswith("work"):
        segmentation_label_prior = 0.02

    cooccurrence_bonus = 0.0
    if resolved_tokens:
        haystack = " ".join(
            [
                candidate_label.lower(),
                description_text.lower(),
                str(candidate.get("country_name") or "").lower(),
                str(candidate.get("admin1_name") or "").lower(),
            ]
        )
        if any(token and token in haystack for token in resolved_tokens):
            cooccurrence_bonus = 0.05

    chronology_bonus = 0.0
    if any(token in description_text.lower() for token in ("medieval", "arthurian", "saint", "abbey", "monastery", "diocese")):
        chronology_bonus = 0.03

    base_score = compute_score(
        surface,
        candidate_label,
        context,
        description_text,
        type_compatible=type_ok,
        canonical_norm=canonical_norm,
        domain_bonus=domain_bonus,
    )

    final_score = (
        base_score * 0.68
        + alias_match_quality * 0.12
        + document_context_compatibility * 0.08
        + source_confidence * 0.06
        + segmentation_label_prior
        + cooccurrence_bonus
        + chronology_bonus
    )
    final_score = max(0.0, min(1.0, round(final_score, 4)))

    return final_score, {
        "label_similarity": round(label_similarity, 4),
        "alias_match_quality": round(alias_match_quality, 4),
        "document_context_compatibility": round(document_context_compatibility, 4),
        "source_confidence": round(source_confidence, 4),
        "segmentation_label_prior": round(segmentation_label_prior, 4),
        "cooccurrence_bonus": round(cooccurrence_bonus, 4),
        "chronology_bonus": round(chronology_bonus, 4),
        "domain_bonus": round(domain_bonus, 4),
        "base_score": round(base_score, 4),
        "final_score": final_score,
    }


def _search_all_sources_for_query(
    query: str,
    *,
    ent_type: str,
    top_k: int,
    wikidata_mode: str,
) -> tuple[dict[str, list[dict[str, Any]]], int, int, int]:
    results: dict[str, list[dict[str, Any]]] = {"wikidata": [], "viaf": [], "geonames": []}
    api_calls_search = 0
    api_calls_viaf = 0
    api_calls_geonames = 0

    if wikidata_mode == "cache":
        cached = cache_get("wikidata", query, max_age_hours=6.0) or []
        results["wikidata"] = cached[:top_k]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures: dict[Any, str] = {}
        if wikidata_mode == "api":
            futures[executor.submit(search_wikidata, query, k=top_k, force_refresh=True)] = "wikidata"
        if ent_type in {"person", "work"}:
            futures[executor.submit(search_viaf, query, k=top_k, ent_type=ent_type)] = "viaf"
        if ent_type == "place":
            futures[executor.submit(search_geonames, query, k=top_k, ent_type=ent_type)] = "geonames"

        for future in as_completed(futures):
            source = futures[future]
            try:
                payload = future.result() or []
            except Exception as exc:  # noqa: BLE001
                log.debug("authority source %s failed for %r: %s", source, query, exc)
                payload = []
            if source == "wikidata":
                results["wikidata"] = payload
                if wikidata_mode == "api":
                    api_calls_search += 1
            elif source == "viaf":
                results["viaf"] = payload
                api_calls_viaf += 1
            elif source == "geonames":
                results["geonames"] = payload
                api_calls_geonames += 1

    return results, api_calls_search, api_calls_viaf, api_calls_geonames


def _candidate_source(candidate: dict[str, Any]) -> str:
    source = str(candidate.get("source") or "").strip().lower()
    if source:
        return source
    if candidate.get("qid"):
        return "wikidata"
    if candidate.get("viaf_id"):
        return "viaf"
    if candidate.get("geonames_id"):
        return "geonames"
    return "unknown"


def _candidate_authority_id(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("authority_id")
        or candidate.get("qid")
        or candidate.get("viaf_id")
        or candidate.get("geonames_id")
        or candidate.get("label")
        or ""
    ).strip()


def _candidate_identity_key(candidate: dict[str, Any]) -> str:
    source = _candidate_source(candidate)
    authority_id = _candidate_authority_id(candidate)
    if authority_id:
        return f"{source}:{authority_id}"
    return f"{source}:{normalise_surface(str(candidate.get('label') or ''))}"


def _merge_candidate(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if key == "aliases":
            aliases: list[dict[str, Any]] = []
            seen: set[tuple[str, str]] = set()
            for item in list(existing.get("aliases") or []) + list(incoming.get("aliases") or []):
                if not isinstance(item, dict):
                    continue
                alias_value = str(item.get("value") or "").strip()
                alias_lang = str(item.get("lang") or "").strip()
                if not alias_value:
                    continue
                alias_key = (alias_value.casefold(), alias_lang)
                if alias_key in seen:
                    continue
                seen.add(alias_key)
                aliases.append({"value": alias_value, "lang": alias_lang})
            merged["aliases"] = aliases
            continue
        if value not in (None, "", [], {}):
            merged[key] = value
    return merged


def _selected_candidate_payload(selected: dict[str, Any] | None) -> dict[str, Any] | None:
    if not selected:
        return None
    return {
        "source": _candidate_source(selected),
        "authority_id": _candidate_authority_id(selected),
        "qid": selected.get("qid", ""),
        "label": selected.get("label", ""),
        "description": selected.get("description", ""),
        "score": selected.get("score", 0.0),
        "viaf_id": selected.get("viaf_id", ""),
        "geonames_id": selected.get("geonames_id", ""),
        "canonical_label": selected.get("canonical_label", ""),
        "aliases": selected.get("aliases", []),
        "lat": selected.get("lat"),
        "lon": selected.get("lon"),
        "country_qids": selected.get("country_qids", []),
        "admin_qids": selected.get("admin_qids", []),
        "country_name": selected.get("country_name", ""),
        "admin1_name": selected.get("admin1_name", ""),
        "parent_location": selected.get("parent_location", ""),
        "type_compatible": selected.get("type_compatible", False),
    }


def _top_candidate_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": _candidate_source(candidate),
        "authority_id": _candidate_authority_id(candidate),
        "qid": candidate.get("qid", ""),
        "label": candidate.get("label", ""),
        "score": candidate.get("score", 0.0),
        "type_compatible": candidate.get("type_compatible", False),
    }


def _persist_selected_entity(
    *,
    selected: dict[str, Any],
    surface: str,
    ent_type: str,
) -> tuple[dict[str, Any], int]:
    source = _candidate_source(selected)
    authority_id = _candidate_authority_id(selected)
    canonical_label = str(
        selected.get("canonical_label")
        or selected.get("label")
        or authority_id
        or selected.get("qid")
        or ""
    ).strip()
    description = str(selected.get("canonical_description") or selected.get("description") or "").strip()

    viaf_profile: dict[str, Any] = {}
    if selected.get("viaf_id") and ent_type in {"person", "work"}:
        try:
            viaf_profile = _fetch_viaf_profile(str(selected.get("viaf_id") or ""))
        except Exception as exc:  # noqa: BLE001
            log.debug("viaf profile fetch failed for %s: %s", selected.get("viaf_id"), exc)
            viaf_profile = {}

    country_name = str(selected.get("country_name") or "").strip()
    admin1_name = str(selected.get("admin1_name") or "").strip()
    country_qids = list(selected.get("country_qids", []) or [])
    admin_qids = list(selected.get("admin_qids", []) or [])
    api_calls_get_delta = 0

    if source == "wikidata":
        if country_qids and not country_name:
            try:
                country_info = enrich_wikidata_item(str(country_qids[0]))
                api_calls_get_delta += 1
                country_name = str(country_info.get("canonical_label") or country_qids[0] or "").strip()
            except Exception:  # noqa: BLE001
                country_name = str(country_qids[0] or "").strip()
        if admin_qids and not admin1_name:
            try:
                admin_info = enrich_wikidata_item(str(admin_qids[0]))
                api_calls_get_delta += 1
                admin1_name = str(admin_info.get("canonical_label") or admin_qids[0] or "").strip()
            except Exception:  # noqa: BLE001
                admin1_name = str(admin_qids[0] or "").strip()

    parent_location = str(selected.get("parent_location") or "").strip()
    if not parent_location:
        parent_location = " > ".join(part for part in (admin1_name, country_name) if part)

    entity_record = pipeline_db.upsert_authority_entity(
        {
            "entity_id": f"{source}:{authority_id}",
            "authority_source": source,
            "authority_id": authority_id,
            "wikidata_qid": selected.get("qid"),
            "viaf_id": selected.get("viaf_id"),
            "geonames_id": selected.get("geonames_id"),
            "canonical_label": canonical_label or authority_id or surface,
            "normalized_label": normalise_surface(canonical_label or authority_id or surface),
            "entity_type": ent_type,
            "description": description or None,
            "lat": selected.get("lat"),
            "lon": selected.get("lon"),
            "country_name": country_name or None,
            "admin1_name": admin1_name or None,
            "parent_location": parent_location or None,
            "meta_json": {
                "source_url": selected.get("url", ""),
                "instance_of_qids": selected.get("instance_of_qids", []),
                "country_qids": country_qids,
                "admin_qids": admin_qids,
                "titles": selected.get("titles"),
                "name_type": selected.get("name_type"),
                "feature_code": selected.get("feature_code"),
                "source_confidence": selected.get("source_confidence"),
            },
        }
    )

    alias_rows = [
        {"alias": canonical_label or authority_id or surface, "alias_lang": "", "alias_source": source},
        {"alias": surface, "alias_lang": "", "alias_source": "mention_surface"},
    ]
    for alias in list(selected.get("aliases", []) or []):
        if not isinstance(alias, dict):
            continue
        alias_rows.append(
            {
                "alias": str(alias.get("value") or ""),
                "alias_lang": str(alias.get("lang") or ""),
                "alias_source": source,
            }
        )
    for alias in list(viaf_profile.get("aliases", []) or []):
        if not isinstance(alias, dict):
            continue
        alias_rows.append(
            {
                "alias": str(alias.get("value") or ""),
                "alias_lang": str(alias.get("lang") or ""),
                "alias_source": "viaf",
            }
        )
    pipeline_db.replace_authority_aliases(entity_record["entity_id"], alias_rows)

    assertion_rows = [
        {
            "source_name": source,
            "property_name": "authority_id",
            "property_value": authority_id,
            "source_json": {"source_url": selected.get("url", "")},
        },
    ]
    if selected.get("qid"):
        assertion_rows.append(
            {
                "source_name": "wikidata",
                "property_name": "wikidata_qid",
                "property_value": str(selected.get("qid") or ""),
                "source_json": {"url": selected.get("url", "")},
            }
        )
    if description:
        assertion_rows.append(
            {
                "source_name": source,
                "property_name": "description",
                "property_value": description,
                "source_json": {"description": description},
            }
        )
    if entity_record.get("lat") is not None and entity_record.get("lon") is not None:
        assertion_rows.append(
            {
                "source_name": source,
                "property_name": "coordinates",
                "property_value": f"{entity_record['lat']},{entity_record['lon']}",
                "source_json": {"lat": entity_record["lat"], "lon": entity_record["lon"]},
            }
        )
    if selected.get("viaf_id"):
        assertion_rows.append(
            {
                "source_name": "viaf",
                "property_name": "viaf_id",
                "property_value": str(selected.get("viaf_id") or ""),
                "source_json": viaf_profile or {"viaf_id": selected.get("viaf_id")},
            }
        )
    if selected.get("geonames_id"):
        assertion_rows.append(
            {
                "source_name": "geonames",
                "property_name": "geonames_id",
                "property_value": str(selected.get("geonames_id") or ""),
                "source_json": {
                    "geonames_id": selected.get("geonames_id"),
                    "country_name": entity_record.get("country_name"),
                    "admin1_name": entity_record.get("admin1_name"),
                    "parent_location": entity_record.get("parent_location"),
                },
            }
        )
    if entity_record.get("country_name"):
        assertion_rows.append(
            {
                "source_name": source,
                "property_name": "country_name",
                "property_value": str(entity_record.get("country_name") or ""),
                "source_json": {"country_name": entity_record.get("country_name")},
            }
        )
    if entity_record.get("admin1_name"):
        assertion_rows.append(
            {
                "source_name": source,
                "property_name": "admin1_name",
                "property_value": str(entity_record.get("admin1_name") or ""),
                "source_json": {"admin1_name": entity_record.get("admin1_name")},
            }
        )
    pipeline_db.replace_authority_source_assertions(entity_record["entity_id"], assertion_rows)
    return entity_record, api_calls_get_delta

# ── API call caps ──────────────────────────────────────────────────────
_MAX_SEARCH_CALLS_PER_RUN = 30    # max wbsearchentities calls per run
_MAX_ENRICH_PER_MENTION = 3       # max wbgetentities calls per mention

# ── Modern-occupation keywords to reject from medieval-entity linking ──
_MODERN_DESCRIPTION_REJECTS: set[str] = {
    "singer", "footballer", "basketball", "baseball", "hockey",
    "rapper", "actress", "actor", "politician", "president",
    "musician", "composer", "band", "album", "song",
    "software", "programming", "computer", "video game",
    "film", "television", "tv series", "anime",
    "bacterium", "bacteria", "virus", "protein", "enzyme",
    "chemical", "compound", "mineral", "drug",
    "asteroid", "galaxy", "planet", "star system",
    "erectile", "dysfunction",
    "operating system", "linux",
    "port ", "seaport", "airport",
    "brand", "company", "corporation",
    "football club", "sports team",
}

# ── Domain keywords that boost medieval entity confidence ──────────────
_MEDIEVAL_DOMAIN_KEYWORDS: set[str] = {
    "arthurian", "round table", "table ronde", "chevalier",
    "knight", "grail", "graal", "camelot", "legendary",
    "chrétien", "chretien", "troyes", "medieval", "médiéval",
    "fictional character", "literary character", "mythological",
    "legendary king", "queen consort",
    "literature", "literary work", "roman de",
    "chanson de geste", "matière de bretagne",
}

# ── Editorial / philology blacklist (non-entity markers) ──────────────
# These surface forms should never be sent to Wikidata as entity queries.
_EDITORIAL_BLACKLIST: set[str] = {
    "lacune", "lacuna", "lacunae",
    "[...]", "\u2026", "illegible", "gap", "missing",
    "illisible", "effacé", "effac\u00e9", "rature",
    "trou", "déchirure", "dechirure",
}

# ── Known-place gazetteer for conditional linking ─────────────────────
_KNOWN_PLACE_GAZETTEER: set[str] = {
    "lausanne", "paris", "lyon", "rome", "avignon", "geneve", "genève",
    "camelot", "logres", "bretagne", "cornouailles", "gaules",
    "tintagel", "carduel", "carlion", "winchester", "londres",
    "jérusalem", "jerusalem", "constantinople", "acre",
    "norgales", "gorre", "sarras", "benoic", "gaule",
}

_WEAK_PLACE_LEXICAL_BLACKLIST: set[str] = {
    "enfant",
    "enfans",
    "enfants",
    "vilanie",
    "vilenie",
    "villanie",
    "amor",
    "amour",
    "courtoisie",
    "felonie",
    "honte",
    "honor",
    "honneur",
    "merci",
    "paor",
    "peor",
    "peur",
    "proece",
    "proesce",
    "chevalerie",
}

_PLACE_CONTEXT_PATTERNS: tuple[str, ...] = (
    " a {surface} ",
    " a la {surface} ",
    " au {surface} ",
    " aux {surface} ",
    " as {surface} ",
    " en {surface} ",
    " vers {surface} ",
    " sur {surface} ",
    " sor {surface} ",
    " de la cite de {surface} ",
    " de la ville de {surface} ",
    " del pais de {surface} ",
    " du pais de {surface} ",
    " en la cite de {surface} ",
    " en la ville de {surface} ",
)

_PLACE_LOCATIVE_PREPOSITIONS: tuple[str, ...] = (
    "a",
    "au",
    "aux",
    "en",
    "vers",
    "sur",
    "sor",
    "devers",
    "lez",
    "pres",
    "près",
)

_PLACE_HEADWORDS: tuple[str, ...] = (
    "cite",
    "cité",
    "ville",
    "castel",
    "chastel",
    "pais",
    "pays",
    "terre",
    "roiaume",
    "reaume",
    "contree",
    "contrée",
    "abbaye",
    "monastere",
    "monastère",
    "eglise",
    "église",
    "diocese",
    "diocèse",
)

# ── Canonical Arthurian / medieval entities ────────────────────────────
# Maps normalised canonical name → preferred query + qualifier strings.
# Used to (a) detect when an OCR mention matches a known entity, and
# (b) build high-quality Wikidata queries for that entity.

_CANONICAL_ENTITIES: dict[str, dict[str, Any]] = {
    "arthur":     {"canon": "Arthur",     "queries": ["King Arthur", "Arthur Arthurian legend"], "type": "person"},
    "artus":      {"canon": "Arthur",     "queries": ["King Arthur", "Arthur Arthurian legend"], "type": "person"},
    "lancelot":   {"canon": "Lancelot",   "queries": ["Lancelot", "Lancelot du Lac"], "type": "person"},
    "leantlote":  {"canon": "Lancelot",   "queries": ["Lancelot", "Lancelot du Lac"], "type": "person"},
    "leantilote": {"canon": "Lancelot",   "queries": ["Lancelot", "Lancelot du Lac"], "type": "person"},
    "lancelote":  {"canon": "Lancelot",   "queries": ["Lancelot", "Lancelot du Lac"], "type": "person"},
    "lanselot":   {"canon": "Lancelot",   "queries": ["Lancelot", "Lancelot du Lac"], "type": "person"},
    "lanceloc":   {"canon": "Lancelot",   "queries": ["Lancelot", "Lancelot du Lac"], "type": "person"},
    "guenievre":  {"canon": "Guinevere",  "queries": ["Guinevere", "Guenièvre"], "type": "person"},
    "guenievr":   {"canon": "Guinevere",  "queries": ["Guinevere", "Guenièvre"], "type": "person"},
    "perceval":   {"canon": "Perceval",   "queries": ["Perceval", "Percival Arthurian"], "type": "person"},
    "parsival":   {"canon": "Perceval",   "queries": ["Perceval", "Percival Arthurian"], "type": "person"},
    "merlin":     {"canon": "Merlin",     "queries": ["Merlin", "Merlin enchanteur"], "type": "person"},
    "galahad":    {"canon": "Galahad",    "queries": ["Galahad", "Galaad chevalier"], "type": "person"},
    "galaad":     {"canon": "Galahad",    "queries": ["Galahad", "Galaad chevalier"], "type": "person"},
    "tristan":    {"canon": "Tristan",    "queries": ["Tristan", "Tristan Iseut"], "type": "person"},
    "gauvain":    {"canon": "Gauvain",    "queries": ["Gauvain", "Gawain chevalier"], "type": "person"},
    "gawain":     {"canon": "Gauvain",    "queries": ["Gauvain", "Gawain chevalier"], "type": "person"},
    "yvain":      {"canon": "Yvain",      "queries": ["Yvain", "Yvain chevalier au lion"], "type": "person"},
    "ivain":      {"canon": "Yvain",      "queries": ["Yvain", "Yvain chevalier au lion"], "type": "person"},
    "graal":      {"canon": "Graal",      "queries": ["Saint Graal", "Holy Grail"], "type": "work"},
    "bohort":     {"canon": "Bohort",     "queries": ["Bohort", "Bors chevalier"], "type": "person"},
    "erec":       {"canon": "Érec",       "queries": ["Érec", "Erec et Enide"], "type": "person"},
}

# ── Name-likeness quality gate ─────────────────────────────────────────

_VOWELS_SET = set("aeiouyàâäéèêëïîôùûüœæ")

_NAME_QUALITY_THRESHOLD = 0.30  # below this → skip Wikidata


def _levenshtein_quick(s: str, t: str) -> int:
    """Levenshtein edit distance (pure-Python, small strings)."""
    n, m = len(s), len(t)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def _compute_name_likeness(surface: str, ent_type: str) -> float:
    """Return a 0-1 quality score for a mention surface.

    A high score means the surface looks like a plausible entity name.
    A low score (< ``_NAME_QUALITY_THRESHOLD``) means the surface is
    likely OCR garbage and should NOT be sent to Wikidata.

    Factors:
      * Blacklisted common word → 0.0
      * Canonical match (normalized distance + bigram overlap) → 1.0 / 0.90
      * Vowel ratio within [0.20, 0.80]
      * No runs of ≥ 4 consecutive consonants
      * Token length within [3, 20]
      * Not purely stopwords
    """
    import re as _re
    from app.services.text_normalization import (
        is_blacklisted_token,
        normalized_edit_distance as _ned,
        bigram_overlap as _bo,
    )
    norm = _re.sub(r"[^\w\s]", " ", surface, flags=_re.UNICODE)
    norm = " ".join(norm.split()).strip().lower()
    if not norm:
        return 0.0

    tokens = norm.split()

    # Fast path: canonical match on any token → excellent quality
    # Uses normalized edit distance + bigram overlap (not raw dist)
    for tok in tokens:
        # Reject blacklisted tokens from contributing to quality
        if is_blacklisted_token(tok):
            continue
        if tok in _CANONICAL_ENTITIES:
            return 1.0
        for canon in _CANONICAL_ENTITIES:
            if len(tok) < 5:
                continue  # short tokens: exact match only (above)
            nd = _ned(tok, canon)
            bo = _bo(tok, canon)
            if nd <= 0.25 and bo >= 0.40:
                return 0.90

    # Remove role triggers / stopwords for scoring
    content = [
        t for t in tokens
        if t not in _QUERY_STOPWORDS and len(t) >= 2
        and not is_blacklisted_token(t)
    ]
    if not content:
        return 0.05  # pure stopwords / blacklisted → low

    score = 0.0
    n_good = 0
    for tok in content:
        tok_score = 0.5  # base
        # Length
        if len(tok) < 3 or len(tok) > 20:
            tok_score -= 0.30
        # Vowel ratio
        vowels = sum(1 for c in tok if c in _VOWELS_SET)
        ratio = vowels / len(tok)
        if ratio < 0.15 or ratio > 0.85:
            tok_score -= 0.30
        elif 0.25 <= ratio <= 0.65:
            tok_score += 0.15  # ideal range
        # Consonant clusters
        max_run = 0
        run = 0
        for c in tok:
            if c in _VOWELS_SET:
                run = 0
            else:
                run += 1
                max_run = max(max_run, run)
        if max_run >= 4:
            tok_score -= 0.35  # heavy penalty

        if tok_score > 0:
            n_good += 1
        score += max(0.0, tok_score)

    avg = score / len(content) if content else 0.0
    # Bonus for having multiple good tokens (reinforces signal)
    if n_good >= 2:
        avg = min(1.0, avg + 0.10)
    return round(min(1.0, max(0.0, avg)), 3)


def _check_canonical_match(surface: str) -> dict[str, Any] | None:
    """If any token in *surface* fuzzy-matches a canonical entity, return it.

    Uses length-aware matching:
      - Tokens < 5 chars: exact match only
      - Tokens >= 5 chars: normalized_edit_distance <= 0.25 AND
        bigram_overlap >= 0.40
      - Blacklisted common words are never matched.

    Returns a dict ``{"canon": …, "queries": …, "type": …, "token": …,
    "dist": …, "nd": …, "bo": …}`` or ``None``.
    """
    import re as _re
    from app.services.text_normalization import (
        is_blacklisted_token,
        normalized_edit_distance as _ned,
        bigram_overlap as _bo,
    )
    norm = _re.sub(r"[^\w\s]", " ", surface, flags=_re.UNICODE)
    tokens = norm.split()

    best: dict[str, Any] | None = None
    best_nd: float = 1.0

    for tok in tokens:
        tok_low = tok.lower()
        if len(tok_low) < 3:
            continue
        if is_blacklisted_token(tok_low):
            continue
        for canon_key, info in _CANONICAL_ENTITIES.items():
            nd = _ned(tok_low, canon_key)
            # Short tokens (< 5 chars): exact match only
            if len(tok_low) < 5:
                if nd == 0.0 and nd < best_nd:
                    best = {**info, "token": tok_low, "dist": 0, "nd": 0.0, "bo": 1.0}
                    best_nd = nd
                continue
            # Tokens >= 5 chars: normalized distance + bigram overlap
            bo = _bo(tok_low, canon_key)
            if nd <= 0.25 and bo >= 0.40 and nd < best_nd:
                raw_dist = _levenshtein_quick(tok_low, canon_key)
                best = {**info, "token": tok_low, "dist": raw_dist, "nd": nd, "bo": bo}
                best_nd = nd
    return best


def _has_strong_place_context(surface: str, evidence_text: str) -> bool:
    import re as _re

    surface_norm = normalize_for_search(surface)
    evidence_norm = f" {normalize_for_search(evidence_text)} "
    if not surface_norm or not evidence_norm.strip():
        return False
    for pattern in _PLACE_CONTEXT_PATTERNS:
        if pattern.format(surface=surface_norm) in evidence_norm:
            return True
    escaped_surface = _re.escape(surface_norm)
    locatives = "|".join(_re.escape(token) for token in _PLACE_LOCATIVE_PREPOSITIONS)
    headwords = "|".join(_re.escape(normalize_for_search(token)) for token in _PLACE_HEADWORDS)
    if _re.search(
        rf"\b(?:{locatives})\b(?:\s+\w+){{0,3}}\s+{escaped_surface}\b",
        evidence_norm,
    ):
        return True
    if _re.search(
        rf"\b(?:{headwords})\b(?:\s+\w+){{0,2}}\s+(?:de|del|du|des)\s+{escaped_surface}\b",
        evidence_norm,
    ):
        return True
    return False


def _mention_context_window(base_text: str, start_offset: int, end_offset: int, *, radius: int = 40) -> str:
    if not base_text:
        return ""
    start = max(0, int(start_offset) - radius)
    end = min(len(base_text), int(end_offset) + radius)
    return str(base_text[start:end] or "")


def _display_probable_type(ent_type: str, reason: str, surface: str) -> str:
    ent = str(ent_type or "").strip().lower() or "unknown"
    reason_text = str(reason or "")
    surface_low = str(surface or "").strip().lower()
    if ent == "place":
        if "low_evidence_place" in reason_text and ("lexical=True" in reason_text or surface_low in _WEAK_PLACE_LEXICAL_BLACKLIST):
            return "lexical/unknown"
        if "low_evidence_place" in reason_text and "context=False" in reason_text:
            return "possible_place_unresolved"
    return ent


def _deferred_reason_for_mention(surface: str, ent_type: str, evidence_text: str, base_reason: str) -> str:
    ent_low = str(ent_type or "").strip().lower()
    surface_text = str(surface or "").strip()
    surface_low = normalize_for_search(surface_text)
    reason_text = str(base_reason or "").strip()
    if ent_low != "place" or not surface_low:
        return reason_text
    is_known_place = surface_low in _KNOWN_PLACE_GAZETTEER
    is_capitalized = surface_text[:1].isupper()
    lexical = surface_low in _WEAK_PLACE_LEXICAL_BLACKLIST
    context_ok = _has_strong_place_context(surface_text, evidence_text)
    minimal_shape = len(surface_low) >= 4 and token_quality_score(surface_low) >= 0.45
    if is_known_place or ((is_capitalized or context_ok) and not lexical and minimal_shape):
        return reason_text
    suffix = (
        f"low_evidence_place: cap={is_capitalized} context={context_ok} "
        f"lexical={lexical} shape={minimal_shape}"
    )
    if not reason_text:
        return suffix
    if suffix in reason_text:
        return reason_text
    return f"{reason_text}; {suffix}"


def _summarize_authority_reason(reason: str, surface: str, ent_type: str) -> str:
    reason_text = str(reason or "").strip()
    surface_low = str(surface or "").strip().lower()
    ent_low = str(ent_type or "").strip().lower()
    if not reason_text:
        return "no reliable authority match"
    if "low_evidence_place" in reason_text:
        if surface_low in _WEAK_PLACE_LEXICAL_BLACKLIST or "lexical=True" in reason_text:
            return "insufficient evidence to treat this lexical item as a place"
        return "insufficient contextual evidence to support place linking"
    if "token_search_allowed=False" in reason_text:
        return "linking deferred because the passage quality was too risky for reliable authority lookup"
    if "name_likeness=" in reason_text:
        return "surface form is too noisy for reliable authority lookup"
    if reason_text == "no wikidata candidates":
        return "no reliable authority candidate found"
    if reason_text == "no candidate marked as selected":
        return "no candidate met the selection threshold"
    if reason_text == "marked ambiguous":
        return "multiple candidates remained ambiguous"
    if ent_low == "date" and "date" in reason_text:
        return "date detected but not authority-linked"
    return reason_text[:120]

# ── Role-aware decomposition ───────────────────────────────────────────

# Trigger tokens that denote a title/role preceding a personal name.
_ROLE_TRIGGERS: set[str] = {
    "roi", "roy", "roya", "rex", "evesque", "évesque", "euesque",
    "duc", "conte", "sire", "pape", "dame", "monsieur",
}

_QUERY_STOPWORDS: set[str] = {
    "et", "de", "la", "le", "les", "des", "du", "en", "a", "au",
    "que", "qui", "ne", "pas", "par", "son", "ses", "sa",
    "li", "lo", "el", "al", "ou", "un", "une",
    # Also include role triggers so they get stripped in query simplification
    "roi", "roy", "roya", "rex", "sire", "monsieur", "dame",
    "evesque", "évesque", "euesque", "duc", "conte", "pape",
}


def _prefilter_candidates(
    candidates: list[dict[str, Any]],
    surface: str,
    ent_type: str,
) -> list[dict[str, Any]]:
    """Pre-filter Wikidata candidates BEFORE calling wbgetentities.

    Removes candidates whose description contains modern-occupation
    keywords that are obviously irrelevant to medieval entity linking.
    Also sorts by relevance (domain keyword bonus) so the enrichment
    cap processes the best candidates first.
    """
    from app.services.text_normalization import normalize_for_search

    surface_norm = normalize_for_search(surface)
    kept: list[tuple[float, dict[str, Any]]] = []

    for wd in candidates:
        desc = str(wd.get("description", "")).lower()
        # Reject modern-occupation descriptions
        rejected = False
        for reject_kw in _MODERN_DESCRIPTION_REJECTS:
            if reject_kw in desc:
                log.debug(
                    "pre-filter: rejecting %s (%s) — description contains '%s'",
                    wd.get("qid", "?"), wd.get("label", "?"), reject_kw,
                )
                rejected = True
                break
        if rejected:
            continue

        # Domain keyword bonus for sorting
        bonus = 0.0
        for kw in _MEDIEVAL_DOMAIN_KEYWORDS:
            if kw in desc:
                bonus += 0.10
        # String proximity bonus
        label_norm = normalize_for_search(str(wd.get("label", "")))
        if label_norm and surface_norm and label_norm == surface_norm:
            bonus += 0.30
        elif label_norm and surface_norm and surface_norm in label_norm:
            bonus += 0.15
        kept.append((bonus, wd))

    # Sort by bonus descending, return just the candidates
    kept.sort(key=lambda x: -x[0])
    return [wd for _, wd in kept]


def _build_query_variants(surface: str, method: str) -> list[str]:
    """Generate progressively simplified queries for noisy medieval OCR.

    Strategy (always applied, regardless of method):
      a) normalised full surface (whitespace normalised, punct removed)
      b) drop stopwords/roles
      c) longest token(s) with len >= 5

    For ``rule:salvage_trigger`` mentions that start with a role trigger,
    the role-stripped form is always the *first* query so that Wikidata
    searches the personal name directly.

    Returns 1-3 deduplicated query strings.
    """
    # Normalise: collapse newlines/whitespace, strip punctuation
    import re as _re
    norm = _re.sub(r"[^\w\s]", " ", surface, flags=_re.UNICODE)
    norm = " ".join(norm.split()).strip()
    if not norm:
        return [surface]

    tokens = norm.split()

    queries: list[str] = []

    # ── (a) normalised full surface ───────────────────────────────────
    queries.append(norm)

    # ── (b) drop stopwords / roles ────────────────────────────────────
    content_tokens = [
        t for t in tokens
        if t.lower() not in _QUERY_STOPWORDS and len(t) >= 2
    ]
    if content_tokens and " ".join(content_tokens) != norm:
        queries.append(" ".join(content_tokens))

    # ── (c) longest tokens with len >= 5 ──────────────────────────────
    long_tokens = sorted(
        [t for t in tokens if len(t) >= 5],
        key=len, reverse=True,
    )[:2]
    if long_tokens:
        lt_str = " ".join(long_tokens)
        queries.append(lt_str)

    # ── For salvage_trigger role mentions: inject name-only first ─────
    if method == "rule:salvage_trigger" and len(tokens) >= 2:
        first_low = tokens[0].lower()
        if first_low in _ROLE_TRIGGERS:
            name_only = [
                t for t in tokens[1:]
                if t.lower() not in _QUERY_STOPWORDS and len(t) >= 2
            ]
            if name_only:
                # Insert name-only query at the front (highest priority)
                queries.insert(0, " ".join(name_only))

    # Deduplicate preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        q_low = q.strip().lower()
        if q_low and q_low not in seen:
            seen.add(q_low)
            deduped.append(q.strip())
    return deduped or [surface]


# ── Public API ─────────────────────────────────────────────────────────


def run_authority_linking(
    run_id: str,
    *,
    top_k: int = 5,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Run the full authority-linking pipeline for *run_id*.

    1. Load entity_mentions + chunks from SQLite
    2. For each mention, search Wikidata for candidates
    3. Score (with hard type gate), disambiguate, enrich selected candidate
    4. Persist candidates into entity_candidates
    5. Return a summary dict suitable for the chat report

    The function is idempotent: existing candidates are cleared for the
    run_id before inserting new ones.
    """
    t0 = time.monotonic()

    # ── 1. Load data ──────────────────────────────────────────────────
    run = pipeline_db.get_run(run_id)
    base_text = str(run.get("proofread_text") or run.get("ocr_text") or "") if run else ""
    asset_ref = str(run.get("asset_ref") or "") if run else ""

    # ── Determine OCR quality level ───────────────────────────────────
    ocr_quality = "MEDIUM"  # default
    if run:
        # Check warnings_json for explicit quality_label
        warnings_raw = run.get("warnings_json") or ""
        if isinstance(warnings_raw, str):
            try:
                warnings_data = json.loads(warnings_raw) if warnings_raw.strip() else {}
            except Exception:
                warnings_data = {}
        else:
            warnings_data = warnings_raw
        explicit_label = ""
        if isinstance(warnings_data, dict):
            explicit_label = str(warnings_data.get("quality_label", ""))
        if explicit_label.upper() in ("HIGH", "MEDIUM", "LOW"):
            ocr_quality = explicit_label.upper()
        elif base_text:
            # Compute from text content
            ocr_quality = text_quality_label(base_text)

    log.info("run %s ocr_quality=%s", run_id, ocr_quality)

    mentions = pipeline_db.list_entity_mentions(run_id)
    if not mentions:
        empty = _empty_result(run_id, "no mentions found")
        empty["_base_text"] = base_text
        empty["asset_ref"] = asset_ref
        empty["ocr_quality"] = ocr_quality
        return empty

    chunks = pipeline_db.list_chunks(run_id)
    chunk_text_by_id: dict[str, str] = {
        str(ch["chunk_id"]): str(ch.get("text") or "") for ch in chunks
    }

    # ── 2. Clear old candidates ───────────────────────────────────────
    _clear_candidates_for_run(run_id)
    _clear_structured_links_for_run(run_id)

    # ── 3. Process each mention ───────────────────────────────────────
    api_calls_search = 0
    api_calls_viaf = 0
    api_calls_geonames = 0
    api_calls_get = 0
    cache_hits = 0
    type_mismatch_count = 0
    quality_skipped = 0
    canonical_matched = 0
    all_candidate_rows: list[dict[str, Any]] = []
    mention_results: list[dict[str, Any]] = []
    resolved_tokens: set[str] = set()

    for mention in mentions:
        mention_id = str(mention["mention_id"])
        surface = str(mention.get("surface") or "")
        ent_type = str(mention.get("ent_type") or "unknown")
        method = str(mention.get("method") or "")
        start_off = int(mention.get("start_offset", 0))
        end_off = int(mention.get("end_offset", 0))
        chunk_id = mention.get("chunk_id")
        mention_confidence = float(mention.get("confidence", 0.0))
        evidence_text = base_text[start_off:end_off] if base_text and end_off <= len(base_text) else surface
        evidence_span = pipeline_db.upsert_evidence_span(
            {
                "run_id": run_id,
                "asset_ref": asset_ref,
                "page_id": asset_ref or None,
                "chunk_id": chunk_id,
                "mention_id": mention_id,
                "start_offset": start_off,
                "end_offset": end_off,
                "raw_text": evidence_text,
                "normalized_text": normalize_for_search(evidence_text) if evidence_text else "",
                "meta_json": {
                    "source": "authority_linking",
                    "ent_type": ent_type,
                    "surface": surface,
                },
            }
        )
        evidence_span_id = str(evidence_span.get("span_id") or "")

        # Context: chunk text + nearby base text
        context = _build_context(
            base_text, start_off, end_off,
            chunk_text_by_id.get(str(chunk_id) if chunk_id else "", ""),
        )

        # ── Quality gate: skip non-linkable types and garbage surfaces ─
        if ent_type == "role":
            # Role mentions (trigger without plausible name) → skip
            quality_skipped += 1
            score_breakdown = _score_breakdown_for_mention(
                mention=mention,
                query_details=[],
                name_likeness=0.0,
                canonical_match=None,
                selected=None,
                top_candidates=[],
                ocr_quality=ocr_quality,
                status="skipped",
                reason="ent_type=role (not linkable)",
            )
            _persist_mention_link(
                run_id=run_id,
                asset_ref=asset_ref,
                mention=mention,
                evidence_text=evidence_text,
                evidence_span_id=evidence_span_id,
                status="skipped",
                reason="ent_type=role (not linkable)",
                score_breakdown=score_breakdown,
            )
            mention_results.append({
                "mention_id": mention_id,
                "surface": surface,
                "ent_type": ent_type,
                "chunk_id": chunk_id,
                "start_offset": start_off,
                "end_offset": end_off,
                "evidence_text": evidence_text,
                "queries_attempted": [],
                "query_details": [],
                "status": "skipped",
                "reason": "ent_type=role (not linkable)",
                "name_likeness": 0.0,
                "canonical_match": None,
                "selected": None,
                "top_candidates": [],
            })
            log.info(
                "mention %s surface=%r SKIPPED: ent_type=role",
                mention_id, surface,
            )
            continue

        # ── Req 3a: DATE mentions → never query Wikidata ─────────────
        if ent_type == "date":
            quality_skipped += 1
            score_breakdown = _score_breakdown_for_mention(
                mention=mention,
                query_details=[],
                name_likeness=0.0,
                canonical_match=None,
                selected=None,
                top_candidates=[],
                ocr_quality=ocr_quality,
                status="skipped",
                reason="ent_type=date (dates never query Wikidata)",
            )
            _persist_mention_link(
                run_id=run_id,
                asset_ref=asset_ref,
                mention=mention,
                evidence_text=evidence_text,
                evidence_span_id=evidence_span_id,
                status="skipped",
                reason="ent_type=date (dates never query Wikidata)",
                score_breakdown=score_breakdown,
            )
            mention_results.append({
                "mention_id": mention_id,
                "surface": surface,
                "ent_type": ent_type,
                "chunk_id": chunk_id,
                "start_offset": start_off,
                "end_offset": end_off,
                "evidence_text": evidence_text,
                "queries_attempted": [],
                "query_details": [],
                "status": "skipped",
                "reason": "ent_type=date (dates never query Wikidata)",
                "name_likeness": 0.0,
                "canonical_match": None,
                "selected": None,
                "top_candidates": [],
            })
            log.info(
                "mention %s surface=%r SKIPPED: ent_type=date (no Wikidata query)",
                mention_id, surface,
            )
            continue

        # ── Req 3b: PLACE mentions → conditional Wikidata query ───────
        if ent_type == "place":
            confidence = float(mention.get("confidence", 0.0))
            surface_low = surface.strip().lower()
            is_capitalized = surface.strip()[:1].isupper()
            is_known_place = surface_low in _KNOWN_PLACE_GAZETTEER
            is_blacklisted = surface_low in _EDITORIAL_BLACKLIST
            is_weak_lexical_place = surface_low in _WEAK_PLACE_LEXICAL_BLACKLIST
            context_window = _mention_context_window(base_text, start_off, end_off)
            has_place_context = _has_strong_place_context(surface, context_window)
            has_minimal_place_shape = len(surface_low) >= 4 and token_quality_score(surface_low) >= 0.45
            # Only query Wikidata if:
            #   (a) matches known-place gazetteer, OR
            #   (b) surface is capitalized and does not look like a generic lexical item, OR
            #   (c) confidence is high AND place context is strong
            place_linkable = (
                is_known_place
                or (
                    is_capitalized
                    and not is_blacklisted
                    and not is_weak_lexical_place
                    and has_minimal_place_shape
                )
                or (
                    confidence >= 0.82
                    and has_place_context
                    and not is_blacklisted
                    and not is_weak_lexical_place
                    and has_minimal_place_shape
                )
            )
            if not place_linkable:
                quality_skipped += 1
                reason_text = (
                    f"low_evidence_place: cap={is_capitalized} "
                    f"gazetteer={is_known_place} conf={confidence:.2f} "
                    f"blacklisted={is_blacklisted} lexical={is_weak_lexical_place} "
                    f"context={has_place_context} shape={has_minimal_place_shape}"
                )
                score_breakdown = _score_breakdown_for_mention(
                    mention=mention,
                    query_details=[],
                    name_likeness=0.0,
                    canonical_match=None,
                    selected=None,
                    top_candidates=[],
                    ocr_quality=ocr_quality,
                    status="unresolved_low_quality",
                    reason=reason_text,
                )
                _persist_mention_link(
                    run_id=run_id,
                    asset_ref=asset_ref,
                    mention=mention,
                    evidence_text=evidence_text,
                    evidence_span_id=evidence_span_id,
                    status="unresolved_low_quality",
                    reason=reason_text,
                    score_breakdown=score_breakdown,
                )
                mention_results.append({
                    "mention_id": mention_id,
                    "surface": surface,
                    "ent_type": ent_type,
                    "chunk_id": chunk_id,
                    "start_offset": start_off,
                    "end_offset": end_off,
                    "evidence_text": evidence_text,
                    "queries_attempted": [],
                    "query_details": [],
                    "status": "unresolved_low_quality",
                    "reason": reason_text,
                    "name_likeness": 0.0,
                    "canonical_match": None,
                    "selected": None,
                    "top_candidates": [],
                })
                log.info(
                    "mention %s surface=%r SKIPPED: low_evidence_place "
                    "(cap=%s gazetteer=%s conf=%.2f blacklisted=%s lexical=%s context=%s)",
                    mention_id, surface, is_capitalized, is_known_place,
                    confidence, is_blacklisted, is_weak_lexical_place, has_place_context,
                )
                continue

        # Name-likeness quality score
        name_likeness = _compute_name_likeness(surface, ent_type)
        if ent_type in ("person",) and name_likeness < _NAME_QUALITY_THRESHOLD:
            quality_skipped += 1
            reason_text = f"name_likeness={name_likeness:.3f} < {_NAME_QUALITY_THRESHOLD} (OCR garbage)"
            score_breakdown = _score_breakdown_for_mention(
                mention=mention,
                query_details=[],
                name_likeness=name_likeness,
                canonical_match=None,
                selected=None,
                top_candidates=[],
                ocr_quality=ocr_quality,
                status="unresolved_low_quality",
                reason=reason_text,
            )
            _persist_mention_link(
                run_id=run_id,
                asset_ref=asset_ref,
                mention=mention,
                evidence_text=evidence_text,
                evidence_span_id=evidence_span_id,
                status="unresolved_low_quality",
                reason=reason_text,
                score_breakdown=score_breakdown,
            )
            mention_results.append({
                "mention_id": mention_id,
                "surface": surface,
                "ent_type": ent_type,
                "chunk_id": chunk_id,
                "start_offset": start_off,
                "end_offset": end_off,
                "evidence_text": evidence_text,
                "queries_attempted": [],
                "query_details": [],
                "status": "unresolved_low_quality",
                "reason": reason_text,
                "name_likeness": name_likeness,
                "canonical_match": None,
                "selected": None,
                "top_candidates": [],
            })
            log.info(
                "mention %s surface=%r SKIPPED: name_likeness=%.3f < %.2f",
                mention_id, surface, name_likeness, _NAME_QUALITY_THRESHOLD,
            )
            continue

        # ── Canonical match check ─────────────────────────────────────
        canon_match = _check_canonical_match(surface)
        canon_match_info = None
        if canon_match:
            canonical_matched += 1
            canon_match_info = {
                "canon": canon_match["canon"],
                "token": canon_match["token"],
                "dist": canon_match["dist"],
            }
            log.info(
                "mention %s surface=%r CANONICAL MATCH: %s (dist=%d)",
                mention_id, surface, canon_match["canon"], canon_match["dist"],
            )

        # ── Query generation (progressive simplification) ─────────────
        queries = _build_query_variants(surface, method)
        if canon_match:
            canon_queries = canon_match["queries"]
            canon_set = {cq.lower() for cq in canon_queries}
            queries = [q for q in queries if q.lower() not in canon_set]
            for cq in reversed(canon_queries):
                queries.insert(0, cq)
        log.info(
            "mention %s surface=%r queries=%s ent_type=%s method=%s name_likeness=%.3f",
            mention_id, surface, queries, ent_type, method, name_likeness,
        )

        merged_candidates: dict[str, dict[str, Any]] = {}
        query_details: list[dict[str, Any]] = []
        canon_query_count = len(canon_match["queries"]) if canon_match else 0
        effective_ent_type = "person" if method == "rule:salvage_trigger" and ent_type == "person" else ent_type
        canonical_norm = canon_match["canon"].lower() if canon_match else ""

        for q_idx, query_str in enumerate(queries):
            if canon_match and q_idx >= canon_query_count and merged_candidates:
                log.debug(
                    "mention %s: canonical search produced %d candidates, skipping fallback query %r",
                    mention_id,
                    len(merged_candidates),
                    query_str,
                )
                continue

            if force_refresh:
                cache_status = "bypassed"
                wikidata_mode = "api" if api_calls_search < _MAX_SEARCH_CALLS_PER_RUN else "skip"
            else:
                _, cache_status = cache_check("wikidata", query_str, max_age_hours=6.0)
                if cache_status == "hit":
                    wikidata_mode = "cache"
                    cache_hits += 1
                elif api_calls_search < _MAX_SEARCH_CALLS_PER_RUN:
                    wikidata_mode = "api"
                else:
                    wikidata_mode = "skip"

            if wikidata_mode == "skip" and ent_type not in {"person", "work", "place"}:
                log.warning(
                    "mention %s: all authority lookups skipped for query %r due to caps/type",
                    mention_id,
                    query_str,
                )
                break

            source_results, search_calls, viaf_calls, geonames_calls = _search_all_sources_for_query(
                query_str,
                ent_type=effective_ent_type,
                top_k=top_k,
                wikidata_mode=wikidata_mode,
            )
            api_calls_search += search_calls
            api_calls_viaf += viaf_calls
            api_calls_geonames += geonames_calls

            raw_hits_by_source: dict[str, list[dict[str, Any]]] = {}
            for source_name, candidates in source_results.items():
                raw_hits_by_source[source_name] = [
                    {
                        "source": source_name,
                        "authority_id": _candidate_authority_id(candidate),
                        "qid": candidate.get("qid", ""),
                        "label": candidate.get("label", ""),
                        "description": candidate.get("description", ""),
                        "search_rank": idx + 1,
                    }
                    for idx, candidate in enumerate(candidates[:5])
                ]
                for candidate in candidates:
                    candidate_payload = dict(candidate)
                    candidate_payload["source"] = source_name
                    candidate_payload["authority_id"] = _candidate_authority_id(candidate_payload)
                    identity_key = _candidate_identity_key(candidate_payload)
                    if identity_key in merged_candidates:
                        merged_candidates[identity_key] = _merge_candidate(merged_candidates[identity_key], candidate_payload)
                    else:
                        merged_candidates[identity_key] = candidate_payload

            query_details.append(
                {
                    "query": query_str,
                    "cache_status": cache_status,
                    "wikidata_called": wikidata_mode == "api",
                    "raw_hits": raw_hits_by_source.get("wikidata", []),
                    "raw_hits_by_source": raw_hits_by_source,
                    "source_hit_counts": {source_name: len(candidates) for source_name, candidates in source_results.items()},
                }
            )

        candidates = list(merged_candidates.values())
        if ent_type in {"person", "work"} and candidates:
            wikidata_candidates = [
                candidate for candidate in candidates if _candidate_source(candidate) == "wikidata"
            ]
            filtered_wikidata = _prefilter_candidates(wikidata_candidates, surface, ent_type)
            allowed_keys = {_candidate_identity_key(candidate) for candidate in filtered_wikidata}
            candidates = [
                candidate
                for candidate in candidates
                if _candidate_source(candidate) != "wikidata" or _candidate_identity_key(candidate) in allowed_keys
            ]

        scored: list[dict[str, Any]] = []
        enrich_count = 0
        for candidate in candidates:
            source_name = _candidate_source(candidate)
            authority_id = _candidate_authority_id(candidate)
            description = str(candidate.get("description") or "")
            enriched_candidate = dict(candidate)

            if source_name == "wikidata" and candidate.get("qid") and enrich_count < _MAX_ENRICH_PER_MENTION:
                enrichment = enrich_wikidata_item(str(candidate.get("qid") or ""))
                api_calls_get += 1
                enrich_count += 1
                enriched_candidate = _merge_candidate(enriched_candidate, enrichment)
            else:
                enrichment = {}

            instance_of = list(enriched_candidate.get("instance_of_qids", []) or [])
            type_ok = is_type_compatible(
                effective_ent_type,
                instance_of,
                description=str(enriched_candidate.get("canonical_description") or description or ""),
            )
            if not type_ok:
                type_mismatch_count += 1

            desc_low = " ".join(
                part
                for part in [
                    str(enriched_candidate.get("description") or ""),
                    str(enriched_candidate.get("canonical_description") or ""),
                    str(enriched_candidate.get("parent_location") or ""),
                ]
                if part
            ).lower()
            domain_bonus = min(0.20, sum(0.05 for kw in _MEDIEVAL_DOMAIN_KEYWORDS if kw in desc_low))
            if source_name == "wikidata" and enriched_candidate.get("qid"):
                domain_bonus = min(0.22, domain_bonus + 0.02)

            score, score_parts = _score_candidate(
                mention=mention,
                candidate=enriched_candidate,
                context=context,
                canonical_norm=canonical_norm,
                domain_bonus=domain_bonus,
                type_ok=type_ok,
                resolved_tokens=resolved_tokens,
            )
            scored.append(
                {
                    **enriched_candidate,
                    "source": source_name,
                    "authority_id": authority_id,
                    "score": score,
                    "type_compatible": type_ok,
                    "score_breakdown": score_parts,
                    "instance_of_qids": instance_of,
                    "canonical_label": enriched_candidate.get("canonical_label", ""),
                    "canonical_description": enriched_candidate.get("canonical_description", ""),
                    "aliases": enriched_candidate.get("aliases", []),
                    "lat": enriched_candidate.get("lat"),
                    "lon": enriched_candidate.get("lon"),
                    "country_qids": enriched_candidate.get("country_qids", []),
                    "admin_qids": enriched_candidate.get("admin_qids", []),
                    "country_name": enriched_candidate.get("country_name", ""),
                    "admin1_name": enriched_candidate.get("admin1_name", ""),
                    "parent_location": enriched_candidate.get("parent_location", ""),
                }
            )

        if canonical_norm and scored:
            scored = rescore_with_canonical(scored, canonical_norm)

        result = disambiguate(scored, ocr_quality=ocr_quality)
        selected = result["selected"]
        status = result["status"]
        reason = result["reason"]

        for i, cand in enumerate(result["all"]):
            is_selected = selected is not None and cand is selected
            meta = {
                "source": _candidate_source(cand),
                "authority_id": _candidate_authority_id(cand),
                "qid": cand.get("qid", ""),
                "label": cand.get("label", ""),
                "description": cand.get("description", ""),
                "url": cand.get("url", ""),
                "viaf_id": cand.get("viaf_id", ""),
                "geonames_id": cand.get("geonames_id", ""),
                "type_compatible": cand.get("type_compatible", False),
                "instance_of_qids": cand.get("instance_of_qids", []),
                "score_breakdown": cand.get("score_breakdown", {}),
                "is_selected": is_selected,
                "link_status": status if is_selected else "",
                "rank": i + 1,
            }
            all_candidate_rows.append(
                {
                    "mention_id": mention_id,
                    "source": _candidate_source(cand),
                    "candidate": _candidate_authority_id(cand),
                    "score": cand.get("score", 0.0),
                    "meta_json": meta,
                }
            )

        top_candidates = [_top_candidate_payload(candidate) for candidate in result["all"][:3]]
        selected_payload = _selected_candidate_payload(selected)
        score_breakdown = _score_breakdown_for_mention(
            mention=mention,
            query_details=query_details,
            name_likeness=name_likeness,
            canonical_match=canon_match_info,
            selected=selected_payload,
            top_candidates=top_candidates,
            ocr_quality=ocr_quality,
            status=status,
            reason=reason,
        )

        if selected and _candidate_authority_id(selected):
            entity_record, entity_api_calls = _persist_selected_entity(
                selected=selected,
                surface=surface,
                ent_type=ent_type,
            )
            api_calls_get += entity_api_calls
            selected_payload = {
                **(selected_payload or {}),
                "entity_id": entity_record.get("entity_id"),
            }
            for token in normalize_for_search(
                " ".join(
                    part
                    for part in [
                        str(entity_record.get("canonical_label") or ""),
                        str(entity_record.get("country_name") or ""),
                        str(entity_record.get("admin1_name") or ""),
                    ]
                    if part
                )
            ).split():
                if len(token) >= 3:
                    resolved_tokens.add(token)

        _persist_mention_link(
            run_id=run_id,
            asset_ref=asset_ref,
            mention=mention,
            evidence_text=evidence_text,
            evidence_span_id=evidence_span_id,
            status=status,
            reason=reason,
            score_breakdown=score_breakdown,
            selected=selected,
        )

        mention_results.append({
            "mention_id": mention_id,
            "surface": surface,
            "ent_type": ent_type,
            "chunk_id": chunk_id,
            "start_offset": start_off,
            "end_offset": end_off,
            "evidence_text": evidence_text,
            "queries_attempted": queries,
            "query_details": query_details,
            "status": status,
            "reason": reason,
            "name_likeness": name_likeness,
            "canonical_match": canon_match_info,
            "selected": selected_payload,
            "top_candidates": top_candidates,
        })

    # ── 4. Persist candidates ─────────────────────────────────────────
    pipeline_db.insert_entity_candidates(all_candidate_rows)

    # ── 5. Build summary ──────────────────────────────────────────────
    elapsed_ms = round((time.monotonic() - t0) * 1000)

    # Counts
    type_counts = Counter(str(m.get("ent_type", "unknown")) for m in mentions)
    source_counts = Counter(str(row.get("source") or "unknown") for row in all_candidate_rows)
    linked_total = sum(1 for r in mention_results if r["status"] == "linked")
    unresolved_total = sum(1 for r in mention_results if _is_unresolved_status(r["status"]))
    ambiguous_total = sum(1 for r in mention_results if r["status"] == "ambiguous")
    skipped_total = sum(1 for r in mention_results if r["status"] == "skipped")

    log.info(
        "authority linking summary run_id=%s mentions=%d candidates=%d linked=%d unresolved=%d ambiguous=%d skipped=%d sources=%s api_calls_search=%d api_calls_viaf=%d api_calls_geonames=%d api_calls_get=%d took_ms=%d",
        run_id,
        len(mentions),
        len(all_candidate_rows),
        linked_total,
        unresolved_total,
        ambiguous_total,
        skipped_total,
        dict(source_counts),
        api_calls_search,
        api_calls_viaf,
        api_calls_geonames,
        api_calls_get,
        elapsed_ms,
    )

    return {
        "run_id": run_id,
        "asset_ref": asset_ref,
        "mentions_total": len(mentions),
        "type_counts": dict(type_counts),
        "candidates_total": len(all_candidate_rows),
        "source_counts": dict(source_counts),
        "linked_total": linked_total,
        "unresolved_total": unresolved_total,
        "ambiguous_total": ambiguous_total,
        "skipped_total": skipped_total,
        "quality_skipped": quality_skipped,
        "canonical_matched": canonical_matched,
        "type_mismatch_count": type_mismatch_count,
        "api_calls_search": api_calls_search,
        "api_calls_viaf": api_calls_viaf,
        "api_calls_geonames": api_calls_geonames,
        "api_calls_get": api_calls_get,
        "api_calls": api_calls_search + api_calls_viaf + api_calls_geonames + api_calls_get,
        "cache_hits": cache_hits,
        "took_ms": elapsed_ms,
        "ocr_quality": ocr_quality,
        "mention_results": mention_results,
        "_base_text": base_text,
    }


# ── Helpers ────────────────────────────────────────────────────────────


def _build_context(
    base_text: str,
    start: int,
    end: int,
    chunk_text: str,
    window: int = 200,
) -> str:
    """Build context string from surrounding text."""
    parts: list[str] = []
    if chunk_text:
        parts.append(chunk_text)
    if base_text:
        ctx_start = max(0, start - window)
        ctx_end = min(len(base_text), end + window)
        parts.append(base_text[ctx_start:ctx_end])
    return " ".join(parts)


def _clear_candidates_for_run(run_id: str) -> None:
    """Remove existing authority candidates for this run."""
    pipeline_db._init_db_if_needed()
    with pipeline_db._connect() as conn:
        conn.execute(
            """
            DELETE FROM entity_candidates
            WHERE mention_id IN (
                SELECT mention_id FROM entity_mentions WHERE run_id=?
            )
            """,
            (run_id,),
        )
        conn.commit()


def _empty_result(run_id: str, reason: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "asset_ref": "",
        "mentions_total": 0,
        "type_counts": {},
        "candidates_total": 0,
        "source_counts": {},
        "linked_total": 0,
        "unresolved_total": 0,
        "ambiguous_total": 0,
        "type_mismatch_count": 0,
        "api_calls_search": 0,
        "api_calls_viaf": 0,
        "api_calls_geonames": 0,
        "api_calls_get": 0,
        "api_calls": 0,
        "cache_hits": 0,
        "took_ms": 0,
        "mention_results": [],
        "_base_text": "",
    }


def _build_result_from_structured_rows(
    *,
    run_id: str,
    asset_ref: str,
    base_text: str,
    mentions: list[dict[str, Any]],
    structured_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows_by_mention = {str(row.get("mention_id") or ""): row for row in structured_rows}
    candidate_rows_by_mention: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidate_rows:
        candidate_rows_by_mention.setdefault(str(candidate.get("mention_id") or ""), []).append(candidate)

    type_counts: Counter[str] = Counter()
    mention_results: list[dict[str, Any]] = []
    linked_total = 0
    unresolved_total = 0
    ambiguous_total = 0
    skipped_total = 0
    source_counts: Counter[str] = Counter()

    for row in structured_rows:
        authority_source = str(row.get("authority_source") or "").strip()
        if authority_source:
            source_counts[authority_source] += 1
    for candidate in candidate_rows:
        source_counts[str(candidate.get("source") or "unknown")] += 1

    for mention in mentions:
        mid = str(mention.get("mention_id") or "")
        etype = str(mention.get("ent_type") or "unknown")
        type_counts[etype] += 1
        surface = str(mention.get("surface") or "")
        start_off = int(mention.get("start_offset", 0))
        end_off = int(mention.get("end_offset", 0))
        chunk_id = mention.get("chunk_id")
        evidence_text = base_text[start_off:end_off] if base_text and end_off <= len(base_text) else surface
        link_row = rows_by_mention.get(mid)
        status = str(link_row.get("link_status") or "unresolved") if link_row else "unresolved"
        reason = str(link_row.get("reason") or "no structured link row") if link_row else "no structured link row"

        if status == "linked":
            linked_total += 1
        elif status == "ambiguous":
            ambiguous_total += 1
        elif status == "skipped":
            skipped_total += 1
        else:
            unresolved_total += 1

        selected = None
        if link_row and str(link_row.get("entity_id") or "").strip():
            selected = {
                "source": str(link_row.get("authority_source") or ""),
                "authority_id": str(link_row.get("authority_id") or ""),
                "qid": str(link_row.get("wikidata_qid") or ""),
                "label": str(link_row.get("canonical_label") or ""),
                "description": str(link_row.get("description") or ""),
                "score": float(link_row.get("confidence") or 0.0),
                "viaf_id": str(link_row.get("viaf_id") or ""),
                "geonames_id": str(link_row.get("geonames_id") or ""),
                "type_compatible": link_row.get("type_compatible"),
            }

        top_candidates = []
        for candidate in candidate_rows_by_mention.get(mid, [])[:3]:
            meta = candidate.get("meta_json") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            top_candidates.append(
                {
                    "source": meta.get("source", candidate.get("source", "")),
                    "authority_id": meta.get("authority_id", candidate.get("candidate", "")),
                    "qid": meta.get("qid", candidate.get("candidate", "")),
                    "label": meta.get("label", ""),
                    "score": float(candidate.get("score", 0.0)),
                    "type_compatible": meta.get("type_compatible", True),
                }
            )

        mention_results.append(
            {
                "mention_id": mid,
                "surface": surface,
                "ent_type": etype,
                "chunk_id": chunk_id,
                "start_offset": start_off,
                "end_offset": end_off,
                "evidence_text": str(link_row.get("evidence_raw_text") or evidence_text) if link_row else evidence_text,
                "status": status,
                "reason": reason,
                "selected": selected,
                "top_candidates": top_candidates,
                "query_details": [],
                "queries_attempted": [surface] if surface else [],
                "canonical_match": None,
                "name_likeness": None,
            }
        )

    return {
        "run_id": run_id,
        "asset_ref": asset_ref,
        "mentions_total": len(mentions),
        "type_counts": dict(type_counts),
        "candidates_total": len(candidate_rows),
        "source_counts": dict(source_counts),
        "linked_total": linked_total,
        "unresolved_total": unresolved_total,
        "ambiguous_total": ambiguous_total,
        "skipped_total": skipped_total,
        "quality_skipped": skipped_total,
        "canonical_matched": 0,
        "type_mismatch_count": sum(
            1
            for candidate in candidate_rows
            if isinstance(candidate.get("meta_json"), dict) and candidate["meta_json"].get("type_compatible") is False
        ),
        "api_calls_search": 0,
        "api_calls_viaf": 0,
        "api_calls_geonames": 0,
        "api_calls_get": 0,
        "api_calls": 0,
        "cache_hits": 0,
        "took_ms": 0,
        "ocr_quality": text_quality_label(base_text) if base_text else "LOW",
        "mention_results": mention_results,
        "_base_text": base_text,
    }


def build_report_from_db(run_id: str) -> dict[str, Any]:
    """Reconstruct a linking report from persisted DB data.

    This does NOT re-run linking — it reads ``entity_mentions`` +
    ``entity_candidates`` and produces the same structure as
    ``run_authority_linking`` so :func:`build_linking_report` can format it.
    """
    run = pipeline_db.get_run(run_id)
    if run is None:
        raise ValueError(f"Run not found: {run_id}")

    asset_ref = str(run.get("asset_ref") or "")
    base_text = str(run.get("proofread_text") or run.get("ocr_text") or "")
    mentions = pipeline_db.list_entity_mentions(run_id)

    if not mentions:
        empty = _empty_result(run_id, "no mentions in DB")
        empty["asset_ref"] = asset_ref
        empty["_base_text"] = base_text
        return {"report": build_linking_report(empty), **empty}

    # Load all candidates grouped by mention_id
    pipeline_db._init_db_if_needed()
    with pipeline_db._connect() as conn:
        rows = conn.execute(
            """
            SELECT c.* FROM entity_candidates c
            JOIN entity_mentions m ON m.mention_id = c.mention_id
            WHERE m.run_id=?
            ORDER BY c.score DESC
            """,
            (run_id,),
        ).fetchall()
    cands_by_mention: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        mid = str(row["mention_id"])
        cand: dict[str, Any] = {k: row[k] for k in row.keys()}
        if cand.get("meta_json") and isinstance(cand["meta_json"], str):
            try:
                cand["meta_json"] = json.loads(cand["meta_json"])
            except Exception:
                pass
        cands_by_mention.setdefault(mid, []).append(cand)

    structured_rows = pipeline_db.list_mention_links_for_run(run_id)
    if structured_rows:
        candidate_rows = []
        for row in rows:
            candidate_row = {k: row[k] for k in row.keys()}
            meta = candidate_row.get("meta_json")
            if isinstance(meta, str):
                try:
                    candidate_row["meta_json"] = json.loads(meta)
                except Exception:
                    pass
            candidate_rows.append(candidate_row)
        result = _build_result_from_structured_rows(
            run_id=run_id,
            asset_ref=asset_ref,
            base_text=base_text,
            mentions=mentions,
            structured_rows=structured_rows,
            candidate_rows=candidate_rows,
        )
        result["report"] = build_linking_report(result)
        return result

    type_counts: Counter[str] = Counter()
    mention_results: list[dict[str, Any]] = []
    candidates_total = 0
    linked_total = 0
    unresolved_total = 0
    ambiguous_total = 0

    for m in mentions:
        mid = str(m["mention_id"])
        etype = str(m.get("ent_type") or "unknown")
        type_counts[etype] += 1
        surface = str(m.get("surface") or "")
        start_off = int(m.get("start_offset", 0))
        end_off = int(m.get("end_offset", 0))
        chunk_id = m.get("chunk_id")
        evidence_text = base_text[start_off:end_off] if base_text and end_off <= len(base_text) else surface

        cands = cands_by_mention.get(mid, [])
        candidates_total += len(cands)

        # Find selected candidate via meta_json.is_selected
        selected_cand = None
        status = "unresolved"
        reason = "no wikidata candidates"
        for c in cands:
            meta = c.get("meta_json") or {}
            if isinstance(meta, dict) and meta.get("is_selected"):
                selected_cand = c
                status = str(meta.get("link_status", "linked"))
                reason = f"score = {c.get('score', 0):.4f}"
                break

        if selected_cand is None and cands:
            # Check if ambiguous
            any_meta = (cands[0].get("meta_json") or {}) if cands else {}
            if isinstance(any_meta, dict) and any_meta.get("link_status") == "ambiguous":
                status = "ambiguous"
                reason = "marked ambiguous"
            elif cands:
                status = "unresolved"
                reason = "no candidate marked as selected"

        if status == "linked":
            linked_total += 1
        elif status == "ambiguous":
            ambiguous_total += 1
        else:
            unresolved_total += 1

        sel_meta = (selected_cand.get("meta_json") or {}) if selected_cand else {}
        if isinstance(sel_meta, str):
            try:
                sel_meta = json.loads(sel_meta)
            except Exception:
                sel_meta = {}

        mention_results.append({
            "mention_id": mid,
            "surface": surface,
            "ent_type": etype,
            "chunk_id": chunk_id,
            "start_offset": start_off,
            "end_offset": end_off,
            "evidence_text": evidence_text,
            "status": status,
            "reason": reason,
            "selected": {
                "qid": sel_meta.get("qid", ""),
                "label": sel_meta.get("label", ""),
                "description": sel_meta.get("description", ""),
                "score": float(selected_cand.get("score", 0)) if selected_cand else 0.0,
                "viaf_id": sel_meta.get("viaf_id", ""),
                "geonames_id": sel_meta.get("geonames_id", ""),
            } if selected_cand else None,
            "top_candidates": [
                {
                    "qid": (c.get("meta_json") or {}).get("qid", c.get("candidate", "")),
                    "label": (c.get("meta_json") or {}).get("label", ""),
                    "score": float(c.get("score", 0)),
                }
                for c in cands[:3]
            ],
        })

    result = {
        "run_id": run_id,
        "asset_ref": asset_ref,
        "mentions_total": len(mentions),
        "type_counts": dict(type_counts),
        "candidates_total": candidates_total,
        "source_counts": {"wikidata": sum(1 for c in rows if str(c["source"]) == "wikidata")},
        "linked_total": linked_total,
        "unresolved_total": unresolved_total,
        "ambiguous_total": ambiguous_total,
        "api_calls": 0,
        "api_calls_search": 0,
        "api_calls_viaf": 0,
        "api_calls_geonames": 0,
        "api_calls_get": 0,
        "cache_hits": 0,
        "took_ms": 0,
        "mention_results": mention_results,
        "_base_text": base_text,
    }
    result["report"] = build_linking_report(result)
    return result


# ── Report builder ─────────────────────────────────────────────────────


def build_linking_report_from_db(run_id: str) -> str:
    return str(build_report_from_db(run_id).get("report") or "")


def build_linking_report(result: dict[str, Any]) -> str:
    """Build a concise, scholar-facing authority summary."""
    L: list[str] = []
    run_id = result.get("run_id", "?")
    asset_ref = result.get("asset_ref", "?")
    mention_results = result.get("mention_results", [])

    L.append("=== AUTHORITY SUMMARY ===")
    L.append(f"run_id: {run_id}")
    L.append(f"asset_ref: {asset_ref}")
    L.append(
        "summary: "
        f"linked={result.get('linked_total', 0)}, "
        f"unresolved={result.get('unresolved_total', 0)}, "
        f"ambiguous={result.get('ambiguous_total', 0)}, "
        f"mentions={result.get('mentions_total', 0)}"
    )
    source_counts = result.get("source_counts", {}) or {}
    if source_counts:
        L.append("sources: " + ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items()) if v))
    L.append("")

    linked = [r for r in mention_results if r.get("status") == "linked"]
    top_linked = sorted(linked, key=lambda r: (r.get("selected") or {}).get("score", 0), reverse=True)[:6]
    L.append(f"=== LINKED ENTITIES ({len(top_linked)}) ===")
    if not top_linked:
        L.append("- none")
    else:
        for r in top_linked:
            sel = r.get("selected") or {}
            ids = [
                f"{sel.get('source', 'authority')}:{sel.get('authority_id') or sel.get('qid', '')}",
                f"wikidata={sel.get('qid', '')}" if sel.get("qid") else "",
                f"viaf={sel.get('viaf_id', '')}" if sel.get("viaf_id") else "",
                f"geonames={sel.get('geonames_id', '')}" if sel.get("geonames_id") else "",
            ]
            ids = [item for item in ids if item]
            L.append(
                f"- {r.get('surface', '')} -> {sel.get('label', '?')} "
                f"[{r.get('ent_type', '?')}] | confidence={float(sel.get('score', 0.0)):.2f}"
            )
            L.append(f"  source_ids: {', '.join(ids)}")
            L.append(
                f"  evidence: chunk={r.get('chunk_id', '?')} offsets={r.get('start_offset', 0)}-{r.get('end_offset', 0)} "
                f'text="{str(r.get("evidence_text", ""))[:120]}"'
            )
    L.append("")

    failures = [
        r for r in mention_results
        if _is_unresolved_status(r.get("status", "")) or r.get("status") == "ambiguous"
    ]
    top_failures = failures[:6]
    L.append(f"=== UNRESOLVED OR AMBIGUOUS ({len(top_failures)}) ===")
    if not top_failures:
        L.append("- none")
    else:
        for r in top_failures:
            display_type = _display_probable_type(
                str(r.get("ent_type") or ""),
                str(r.get("reason") or ""),
                str(r.get("surface") or ""),
            )
            L.append(
                f"- {r.get('surface', '')} [{display_type or '?'}] | "
                f"status={r.get('status', '?')} | reason={_summarize_authority_reason(str(r.get('reason', '')), str(r.get('surface', '')), display_type or str(r.get('ent_type', '')))}"
            )
            top_candidate = (r.get("top_candidates") or [{}])[0]
            if top_candidate and any(top_candidate.values()):
                candidate_id = top_candidate.get("authority_id") or top_candidate.get("qid", "")
                L.append(
                    f"  top_candidate: {top_candidate.get('source', 'authority')}:{candidate_id} "
                    f"| {top_candidate.get('label', '?')} | score={float(top_candidate.get('score', 0.0)):.2f}"
                )
            L.append(
                f"  evidence: chunk={r.get('chunk_id', '?')} offsets={r.get('start_offset', 0)}-{r.get('end_offset', 0)} "
                f'text="{str(r.get("evidence_text", ""))[:120]}"'
            )
    return "\n".join(L)
