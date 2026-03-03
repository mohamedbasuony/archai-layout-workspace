"""Wikidata entity search client with on-disk cache.

Queries the Wikidata ``wbsearchentities`` API for candidate entities,
optionally enriching with VIAF (P214) and GeoNames (P1566) IDs.

The on-disk cache avoids redundant API calls across pipeline re-runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

log = logging.getLogger("archai.wikidata")

# ── Configuration ──────────────────────────────────────────────────────

_WBSEARCH_URL = "https://www.wikidata.org/w/api.php"
_WBGETENTITIES_URL = "https://www.wikidata.org/w/api.php"
_USER_AGENT = "Archai-OCR-Pipeline/1.0 (research; mailto:archai@example.com)"
_DEFAULT_K = 5
_REQUEST_DELAY_S = 0.25          # polite delay between API calls

# ── Normalisation ──────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalise_surface(text: str) -> str:
    """Lowercase, strip diacritics, remove punctuation, collapse whitespace."""
    text = text.strip().lower()
    # Strip diacritics
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


# ── On-disk cache (SQLite) ─────────────────────────────────────────────

_CACHE_LOCK = threading.Lock()
_CACHE_READY = False


def _cache_path() -> Path:
    raw = os.getenv("ARCHAI_WIKIDATA_CACHE", "").strip()
    if raw:
        return Path(raw).expanduser()
    # Default alongside the pipeline DB
    app_dir = Path(__file__).resolve().parents[1]
    return app_dir / ".data" / "wikidata_cache.sqlite"


def _cache_conn() -> sqlite3.Connection:
    p = _cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_cache_if_needed() -> None:
    global _CACHE_READY
    if _CACHE_READY:
        return
    with _CACHE_LOCK:
        if _CACHE_READY:
            return
        with _cache_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS wikidata_cache (
                    cache_key TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    query TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            # Purge stale empty-result rows so they never block searches
            try:
                conn.execute(
                    "DELETE FROM wikidata_cache WHERE result_json IN ('[]', '')"
                )
            except Exception:
                pass
            conn.commit()
        _CACHE_READY = True


def _cache_key(source: str, query: str, *, lang: str = "", etype: str = "") -> str:
    """Build a cache key incorporating source, query, language, and entity type.

    Including *lang* and *etype* prevents cross-contamination when the
    same surface is searched with different parameters.
    """
    norm = normalise_surface(query)
    raw = f"{source}:{norm}"
    if lang:
        raw += f":lang={lang}"
    if etype:
        raw += f":etype={etype}"
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_get(
    source: str,
    query: str,
    *,
    max_age_hours: float | None = None,
) -> list[dict[str, Any]] | None:
    """Return cached candidates or ``None``.

    If *max_age_hours* is set **and** the cached result is an empty list
    ("no candidates"), entries older than *max_age_hours* are ignored so
    that the caller retries the API.  Non-empty results are always
    returned regardless of age.
    """
    _init_cache_if_needed()
    key = _cache_key(source, query)
    with _cache_conn() as conn:
        row = conn.execute(
            "SELECT result_json, created_at FROM wikidata_cache WHERE cache_key=?",
            (key,),
        ).fetchone()
    if row is None:
        return None
    try:
        data = json.loads(row["result_json"])
    except Exception:
        return None

    # Expire stale negatives
    if max_age_hours is not None and isinstance(data, list) and len(data) == 0:
        from datetime import datetime, timezone, timedelta
        try:
            created = datetime.fromisoformat(row["created_at"])
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - created > timedelta(hours=max_age_hours):
                log.debug("Ignoring stale empty cache for %s (age > %sh)", query, max_age_hours)
                return None
        except Exception:
            pass  # malformed timestamp → treat as valid
    return data


def cache_check(
    source: str,
    query: str,
    *,
    max_age_hours: float | None = None,
) -> tuple[list[dict[str, Any]] | None, str]:
    """Check cache and return ``(data, status)``.

    *status* is one of:

    - ``"miss"`` — no cache entry exists
    - ``"hit"`` — non-empty cached results (always valid)
    - ``"empty-hit"`` — empty cached results within *max_age_hours* TTL
    - ``"expired"`` — empty cached results that exceeded the TTL
    """
    _init_cache_if_needed()
    key = _cache_key(source, query)
    with _cache_conn() as conn:
        row = conn.execute(
            "SELECT result_json, created_at FROM wikidata_cache WHERE cache_key=?",
            (key,),
        ).fetchone()
    if row is None:
        return None, "miss"
    try:
        data = json.loads(row["result_json"])
    except Exception:
        return None, "miss"
    if not isinstance(data, list):
        return None, "miss"

    # Non-empty results are always valid
    if len(data) > 0:
        return data, "hit"

    # Empty results — check TTL
    if max_age_hours is not None:
        from datetime import datetime, timezone, timedelta
        try:
            created = datetime.fromisoformat(row["created_at"])
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - created > timedelta(hours=max_age_hours):
                log.debug("cache expired for %s (empty result, age > %sh)", query, max_age_hours)
                return None, "expired"
        except Exception:
            pass
    return data, "empty-hit"


def cache_put(source: str, query: str, results: list[dict[str, Any]]) -> None:
    """Store candidates in cache.

    **Empty results are never cached** (option A) so that negative hits
    cannot suppress future Wikidata searches.
    """
    if not results:
        log.debug("cache_put: skipping empty results for %s/%s", source, query)
        return
    _init_cache_if_needed()
    key = _cache_key(source, query)
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    blob = json.dumps(results, ensure_ascii=False)
    with _cache_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO wikidata_cache (cache_key, source, query, result_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (key, source, normalise_surface(query), blob, ts),
        )
        conn.commit()


# ── HTTP helpers ───────────────────────────────────────────────────────

def _http_get(url: str, params: dict[str, str]) -> dict[str, Any]:
    """Simple synchronous HTTP GET returning JSON.

    Uses ``urllib`` to avoid adding a dependency on ``requests``/``httpx``
    at runtime.
    """
    import urllib.request
    import urllib.error

    qs = "&".join(f"{k}={quote_plus(v)}" for k, v in params.items())
    full_url = f"{url}?{qs}"
    req = urllib.request.Request(full_url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        log.warning("Wikidata HTTP error %s for %s", exc.code, full_url)
        return {}
    except Exception as exc:
        log.warning("Wikidata request failed for %s: %s", full_url, exc)
        return {}


_last_request_ts: float = 0.0
_rate_lock = threading.Lock()


def _rate_limit() -> None:
    """Enforce polite delay between API requests."""
    global _last_request_ts
    with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_request_ts
        if elapsed < _REQUEST_DELAY_S:
            time.sleep(_REQUEST_DELAY_S - elapsed)
        _last_request_ts = time.monotonic()


# ── Wikidata search ───────────────────────────────────────────────────

def search_wikidata(
    surface: str,
    *,
    language: str = "fr",
    k: int = _DEFAULT_K,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Search Wikidata for entity candidates matching *surface*.

    Returns up to *k* candidates, each with keys:
    ``qid``, ``label``, ``description``, ``url``.

    Results are cached on disk.  Pass ``force_refresh=True`` to bypass
    the cache and always hit the API.

    **v3 fixes:**
    - Applies ``normalize_for_search`` to the query for better matching.
    - Cache key includes language to avoid cross-contamination.
    - Empty-hit cache entries are treated as misses (never block API).
    """
    if not force_refresh:
        cached = cache_get("wikidata", surface, max_age_hours=6.0)
        if cached is not None:
            return cached[:k]

    # Normalize the search query for better Wikidata matching
    from app.services.text_normalization import normalize_unicode, ocr_confusion_fixes
    search_query = normalize_unicode(surface)
    search_query = ocr_confusion_fixes(search_query)
    # Keep diacritics for Wikidata (stripping them can hurt French searches)
    search_query = search_query.strip()
    if not search_query:
        search_query = surface

    _rate_limit()
    params = {
        "action": "wbsearchentities",
        "search": search_query,
        "language": language,
        "format": "json",
        "limit": str(k),
        "type": "item",
    }
    data = _http_get(_WBSEARCH_URL, params)
    results: list[dict[str, Any]] = []
    for item in data.get("search", []):
        results.append({
            "qid": item.get("id", ""),
            "label": item.get("label", ""),
            "description": item.get("description", ""),
            "url": item.get("url") or f"https://www.wikidata.org/wiki/{item.get('id', '')}",
        })
    cache_put("wikidata", surface, results)
    return results[:k]


# ── Enrichment: VIAF + GeoNames from Wikidata ─────────────────────────

def enrich_wikidata_item(qid: str) -> dict[str, Any]:
    """Fetch VIAF (P214) and GeoNames (P1566) IDs for a Wikidata item.

    Also retrieves P31 (instance of) for type-checking.

    Returns dict with keys: ``viaf_id``, ``geonames_id``,
    ``instance_of_qids`` (list of Q-IDs for P31 values).
    """
    cache_key_str = f"enrich:{qid}"
    cached = cache_get("wikidata_enrich", cache_key_str)
    if cached is not None:
        return cached[0] if cached else {}

    _rate_limit()
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "props": "claims",
        "format": "json",
    }
    data = _http_get(_WBGETENTITIES_URL, params)
    entity = data.get("entities", {}).get(qid, {})
    claims = entity.get("claims", {})

    viaf_id = ""
    p214 = claims.get("P214", [])
    if p214:
        viaf_id = str(
            p214[0]
            .get("mainsnak", {})
            .get("datavalue", {})
            .get("value", "")
        )

    geonames_id = ""
    p1566 = claims.get("P1566", [])
    if p1566:
        geonames_id = str(
            p1566[0]
            .get("mainsnak", {})
            .get("datavalue", {})
            .get("value", "")
        )

    # P31 — instance of
    instance_of_qids: list[str] = []
    for claim in claims.get("P31", []):
        ms = claim.get("mainsnak", {})
        dv = ms.get("datavalue", {})
        val = dv.get("value", {})
        if isinstance(val, dict) and val.get("id"):
            instance_of_qids.append(str(val["id"]))

    result = {
        "viaf_id": viaf_id,
        "geonames_id": geonames_id,
        "instance_of_qids": instance_of_qids,
    }
    cache_put("wikidata_enrich", cache_key_str, [result])
    return result


# ── Type-compatibility mapping ─────────────────────────────────────────

# Maps our entity types to Wikidata P31 Q-IDs that are compatible
_ETYPE_TO_COMPATIBLE_QIDS: dict[str, set[str]] = {
    "person": {
        "Q5",           # human
        "Q15632617",    # fictional human
        "Q95074",       # fictional character
        "Q3658341",     # literary character
        "Q15773317",    # legendary character
        "Q15773347",    # mythological character
        "Q4271324",     # mythical or legendary character
        "Q21070568",    # fictional entity
        "Q14073567",    # mythical character
    },
    "place": {
        "Q515",         # city
        "Q486972",      # human settlement
        "Q6256",        # country
        "Q35657",       # state
        "Q3024240",     # historical country
        "Q82794",       # geographic region
        "Q23442",       # island
        "Q46831",       # mountain range
        "Q8502",        # mountain
        "Q4022",        # river
        "Q618123",      # geographical feature
        "Q2221906",     # geographic location
        "Q1549591",     # big city
        "Q3957",        # town
        "Q532",         # village
    },
    "org": {
        "Q43229",       # organization
        "Q4830453",     # business
        "Q3918",        # university
        "Q9174",        # religion
        "Q1530022",     # religious order
    },
    "work": {
        "Q7725634",     # literary work
        "Q571",         # book
        "Q47461344",    # written work
        "Q11424",       # film
        "Q5398426",     # TV series
    },
    "title": {
        "Q7725634",     # literary work
        "Q571",         # book
        "Q47461344",    # written work
    },
    "event": {
        "Q1190554",     # event
        "Q178561",      # battle
        "Q198",         # war
    },
}

# Inverse: Q-IDs → compatible entity types
_QID_TO_ETYPES: dict[str, set[str]] = {}
for _etype, _qids in _ETYPE_TO_COMPATIBLE_QIDS.items():
    for _qid in _qids:
        _QID_TO_ETYPES.setdefault(_qid, set()).add(_etype)


def is_type_compatible(
    ent_type: str,
    instance_of_qids: list[str],
    *,
    description: str = "",
) -> bool:
    """Check if Wikidata P31 values are compatible with our entity type.

    **Precision-first (v2):** if no P31 values are known we return
    ``False`` so that un-typed Wikidata items are never auto-linked.

    **Medieval-aware (v4):** for ``ent_type="person"`` we also:
    - Accept candidates whose P31 includes fictional/legendary character
      QIDs (Q95074, Q3658341, Q15773317, etc.)
    - Accept candidates whose description contains medieval/Arthurian
      domain keywords (even if P31 is missing or unusual).
    - Reject candidates that are given-name / family-name entries
      (Q202444, Q101352, Q12308941, Q11879590, Q4167410).
    """
    ent_lower = ent_type.lower()
    compatible = _ETYPE_TO_COMPATIBLE_QIDS.get(ent_lower)
    if compatible is None:
        return True   # unknown *our* type → don't penalise

    qid_set = set(instance_of_qids) if instance_of_qids else set()

    # ── Name-entity exclusion (person only) ───────────────────────────
    if ent_lower == "person" and qid_set & _NAME_ENTITY_QIDS:
        return False  # given name / family name / disambiguation → reject

    if not instance_of_qids:
        # No P31 data — fall through to description-keyword check
        # for person entities only; other types remain strict.
        if ent_lower != "person":
            return False
    else:
        # Standard P31 check
        if compatible & qid_set:
            return True

    # ── Description-keyword fallback (person only) ────────────────────
    if ent_lower == "person" and description:
        desc_low = description.lower()
        for kw in _PERSON_DESCRIPTION_ACCEPT_KEYWORDS:
            if kw in desc_low:
                # Make sure it's not also a name-entity via description
                if not any(nk in desc_low for nk in _NAME_DESCRIPTION_REJECT_KEYWORDS):
                    return True

    # If we reach here with no P31 data → reject
    if not instance_of_qids:
        return False

    return bool(compatible & qid_set)


# ── Name-entity P31 Q-IDs to reject for person linking ────────────────
_NAME_ENTITY_QIDS: set[str] = {
    "Q202444",      # given name
    "Q101352",      # family name
    "Q12308941",    # male given name
    "Q11879590",    # female given name
    "Q4167410",     # Wikimedia disambiguation page
    "Q66480858",    # surname
}

# ── Description keywords that indicate a person-compatible entity ─────
_PERSON_DESCRIPTION_ACCEPT_KEYWORDS: tuple[str, ...] = (
    "arthurian",
    "legendary",
    "fictional character",
    "literary character",
    "character",
    "knight of the round table",
    "chevalier de la table ronde",
    "mythological",
    "mythical",
    "round table",
    "table ronde",
    "grail",
    "graal",
    "camelot",
)

# ── Description keywords that indicate a name-entity (not a person) ──
_NAME_DESCRIPTION_REJECT_KEYWORDS: tuple[str, ...] = (
    "given name",
    "family name",
    "surname",
    "prénom",
    "nom de famille",
)
