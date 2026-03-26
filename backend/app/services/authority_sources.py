"""External authority source helpers for VIAF and GeoNames.

These helpers keep direct source lookups separate from the main
authority-linking orchestrator so the resolver can query multiple
authorities in parallel.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import quote_plus

from app.config import settings

log = logging.getLogger("archai.authority_sources")

_USER_AGENT = "Archai-OCR-Pipeline/1.0 (research; mailto:archai@example.com)"


def _http_json(url: str, params: dict[str, Any], *, timeout: int = 10) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    query = "&".join(
        f"{quote_plus(str(key))}={quote_plus(str(value))}"
        for key, value in params.items()
        if value is not None and str(value) != ""
    )
    request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        log.debug("Authority source HTTP error for %s: %s", url, exc)
        return {}
    except Exception as exc:  # noqa: BLE001
        log.debug("Authority source request failed for %s: %s", url, exc)
        return {}


def _fetch_viaf_record(viaf_id: str) -> dict[str, Any]:
    viaf_id = str(viaf_id or "").strip()
    if not viaf_id:
        return {}
    return _http_json(f"https://viaf.org/viaf/{viaf_id}/viaf.json", {}, timeout=10)


def search_viaf(query: str, *, k: int = 5, ent_type: str = "") -> list[dict[str, Any]]:
    if ent_type not in {"person", "work"}:
        return []
    payload = _http_json(
        "https://viaf.org/viaf/AutoSuggest",
        {"query": query},
        timeout=10,
    )
    rows = payload.get("result") or []
    if not isinstance(rows, list):
        return []

    out: list[dict[str, Any]] = []
    for row in rows[:k]:
        if not isinstance(row, dict):
            continue
        viaf_id = str(row.get("viafid") or "").strip()
        label = str(row.get("term") or "").strip()
        if not viaf_id or not label:
            continue
        record = _fetch_viaf_record(viaf_id)
        aliases: list[dict[str, str]] = []
        seen: set[str] = set()
        main_headings = record.get("mainHeadings", {}).get("data", [])
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

        name_type = str(record.get("nameType") or row.get("nametype") or "").strip()
        titles = record.get("titles")
        description_parts = [part for part in [name_type, label] if part]
        out.append(
            {
                "source": "viaf",
                "authority_id": viaf_id,
                "qid": "",
                "viaf_id": viaf_id,
                "geonames_id": "",
                "label": label,
                "description": " | ".join(description_parts) or "VIAF authority record",
                "url": f"https://viaf.org/viaf/{viaf_id}",
                "aliases": aliases,
                "instance_of_qids": [],
                "canonical_label": label,
                "canonical_description": "VIAF authority record",
                "lat": None,
                "lon": None,
                "country_qids": [],
                "admin_qids": [],
                "country_name": "",
                "admin1_name": "",
                "parent_location": "",
                "titles": titles,
                "name_type": name_type,
                "source_confidence": 0.92,
            }
        )
    return out


def search_geonames(query: str, *, k: int = 5, ent_type: str = "") -> list[dict[str, Any]]:
    if ent_type != "place":
        return []
    username = str(settings.geonames_username or "").strip()
    if not username:
        return []
    base_url = str(settings.geonames_base_url or "http://api.geonames.org").rstrip("/")
    payload = _http_json(
        f"{base_url}/searchJSON",
        {
            "q": query,
            "maxRows": max(1, min(int(k), 10)),
            "style": "FULL",
            "username": username,
        },
        timeout=int(settings.geonames_timeout_seconds or 10),
    )
    rows = payload.get("geonames") or []
    if not isinstance(rows, list):
        return []

    out: list[dict[str, Any]] = []
    for row in rows[:k]:
        if not isinstance(row, dict):
            continue
        geoname_id = str(row.get("geonameId") or "").strip()
        label = str(row.get("toponymName") or row.get("name") or "").strip()
        if not geoname_id or not label:
            continue
        country_name = str(row.get("countryName") or "").strip()
        admin1_name = str(row.get("adminName1") or "").strip()
        fcode_name = str(row.get("fcodeName") or row.get("fclName") or "place").strip()
        parent_location = " > ".join(part for part in (admin1_name, country_name) if part)
        out.append(
            {
                "source": "geonames",
                "authority_id": geoname_id,
                "qid": "",
                "viaf_id": "",
                "geonames_id": geoname_id,
                "label": label,
                "description": " | ".join(part for part in (fcode_name, parent_location) if part) or "GeoNames place",
                "url": f"https://www.geonames.org/{geoname_id}",
                "aliases": [{"lang": "", "value": label}],
                "instance_of_qids": [],
                "canonical_label": label,
                "canonical_description": fcode_name or "GeoNames place",
                "lat": row.get("lat"),
                "lon": row.get("lng"),
                "country_qids": [],
                "admin_qids": [],
                "country_name": country_name,
                "admin1_name": admin1_name,
                "parent_location": parent_location,
                "feature_code": str(row.get("fcode") or "").strip(),
                "source_confidence": 0.95,
            }
        )
    return out
