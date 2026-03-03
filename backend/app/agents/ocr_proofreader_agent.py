from __future__ import annotations

import json
import re

from app.config import settings
from app.services.saia_client import SaiaClient

PALEO_PROOFREAD_SYSTEM_PROMPT = (
    "Role: Professional paleographer. Task: constrained OCR proofreading for archival (Archai) transcription.\n"
    "\n"
    "INPUTS\n"
    "You will receive:\n"
    "\n"
    "script_hint (from the OCR extractor) describing the dominant script/language: latin|greek|cyrillic|mixed|unknown\n"
    "detected_language (chosen by OCR) from manuscript languages such as latin, old_english, middle_english, french, old_french, middle_french, anglo_norman, occitan, german-family, italian, spanish, portuguese, catalan, greek, hebrew, arabic, mixed, unknown\n"
    "OCR text (plain text, one manuscript line per line)\n"
    "GOAL\n"
    "Improve the OCR text while staying faithful to what is visible. Correct ONLY obvious OCR mistakes typical for the given script_hint.\n"
    "\n"
    "HARD CONSTRAINTS (NON-NEGOTIABLE)\n"
    "\n"
    "NEVER invent text.\n"
    "NEVER replace […] with guessed content.\n"
    "NEVER remove ? or […] unless the existing OCR characters already make the reading deterministic.\n"
    "If OCR appears to guess a word that is not supported by the visible letters, prefer replacing uncertain characters with ? or […] rather than inventing.\n"
    "Preserve line breaks and reading order (one manuscript line per line).\n"
    "Preserve abbreviations; do NOT expand abbreviations.\n"
    "Do NOT translate. Do NOT modernize. If unsure, leave unchanged.\n"
    "Do NOT “correct” proper names/titles (name-like tokens). If uncertain, keep unchanged or mask ONLY the uncertain part with […] (do not guess).\n"
    "SCRIPT/LANGUAGE DISCIPLINE\n"
    "\n"
    "Treat script_hint as authoritative. Do not introduce characters from other scripts unless they already appear in the OCR and are clearly intended.\n"
    "Treat detected_language as the target manuscript language for cleanup decisions.\n"
    "Remove obvious OCR junk/fragments only when they are clearly non-linguistic noise.\n"
    "If uncertain whether text is noise, keep it unchanged.\n"
    "ALLOWED FIXES (ONLY WHEN CLEAR AND LOCAL)\n"
    "\n"
    "Common OCR confusions for the script: u/v, i/j, rn→m, cl/ct, e/c, long-s(ſ) misread, spacing/punctuation artifacts.\n"
    "Very common formula spellings when one small edit away and strongly supported by context (examples: patru→patri, clemencia/clemena→clementia, faunente→fauente).\n"
    "DECORATED INITIAL MERGE (CRITICAL)\n"
    "\n"
    "If the OCR begins with a standalone line that is a single letter (e.g., \"R\") and the next line begins a word that plausibly continues it (e.g., starts with \"euer...\" / \"euerend...\" / similar),\n"
    "then merge them into ONE line by prepending the letter to the next line's first token and deleting the standalone letter line.\n"
    "Do this merge ONLY when it reduces an obvious segmentation artifact. Do NOT invent new words; only combine existing characters across the two lines.\n"
    "OUTPUT\n"
    "Return ONLY the corrected plain text (no JSON, no headings, no commentary, no diagnostics)."
)

PALEO_PROOFREAD_USER_PROMPT_TEMPLATE = (
    "script_hint: {script_hint}\n\n"
    "detected_language: {detected_language}\n\n"
    "OCR text:\n{ocr_text}"
)

DECORATED_INITIAL_WHITELIST = {
    "cuero di",
    "cvero di",
    "euerendi",
    "euerendu",
    "euerendi i",
    "euerendu i",
}

LATIN_SAFE_CORRECTIONS = {
    "patru": "patri",
    "clemencia": "clementia",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’\\-]*")
SINGLE_INITIAL_RE = re.compile(r"^[A-Z]$")
# Regex: line starts with a single consonant (NOT a vowel) followed by space + word.
# This catches OCR artifacts like "n ous" → "nous", "p our" → "pour", etc.
# Vowels (a, e, i, o, u, y) are excluded because standalone vowels are often real words.
_LEADING_CONSONANT_FRAG_RE = re.compile(
    r"^([bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ])\s+([a-zA-ZÀ-ÿ])"
)

# Regex: line starts with a single digit that is suspicious (not clearly a list number)
_LEADING_STRAY_DIGIT_RE = re.compile(r"^(\d)\s+([a-zA-ZÀ-ÿ])")


def apply_fragment_merge(text: str) -> str:
    """Merge leading single-consonant fragments and strip stray leading digits.
    
    Examples:
        "n ous" → "nous"
        "p our" → "pour"
        "9 ous" → "ous"   (stray digit removed)
    
    Vowels (a/e/i/o/u/y/A/E/I/O/U/Y) are NOT merged because they are often
    legitimate standalone words or articles.
    """
    if not text:
        return ""
    lines = text.splitlines()
    merged: list[str] = []
    for line in lines:
        stripped = line.rstrip()
        # Merge consonant fragments: "n ous" → "nous"
        stripped = _LEADING_CONSONANT_FRAG_RE.sub(r"\1\2", stripped)
        # Strip stray leading digit: "9 ous" → "ous"
        stripped = _LEADING_STRAY_DIGIT_RE.sub(r"\2", stripped)
        merged.append(stripped)
    return "\n".join(merged)

def _clean_plain_text(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                maybe = parsed.get("text")
                if isinstance(maybe, str):
                    text = maybe.strip()
        except Exception:
            pass
    return text.strip()


def _normalize_script_hint(script_hint: str | None) -> str:
    hint = str(script_hint or "unknown").strip().lower()
    if hint == "latin_medieval":
        return "latin"
    if hint == "insular_old_english":
        return "unknown"
    if hint in {"greek", "cyrillic", "mixed"}:
        return "unknown"
    if hint in {"latin", "unknown"}:
        return hint
    return "unknown"


def _normalize_space_lower(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _looks_name_like_token(token: str) -> bool:
    if not token:
        return False
    if len(token) >= 18:
        return True
    if any(ch.isupper() for ch in token[1:]):
        return True
    return False


def _apply_case(template: str, replacement: str) -> str:
    if template.isupper():
        return replacement.upper()
    if template[:1].isupper() and template[1:].islower():
        return replacement.capitalize()
    return replacement


def _iter_lines(text: str) -> list[str]:
    return text.splitlines()


def apply_decorated_initial_fix(text: str, script_hint: str | None) -> str:
    if _normalize_script_hint(script_hint) != "latin":
        return text

    lines = _iter_lines(text)
    if len(lines) < 2:
        return text

    first = lines[0].strip()
    second = lines[1].strip()
    if not SINGLE_INITIAL_RE.fullmatch(first):
        return text

    second_norm = _normalize_space_lower(second)
    if len(second) <= 12 and second_norm in DECORATED_INITIAL_WHITELIST:
        merged_lines = ["Reuerendi", *lines[2:]]
        return "\n".join(merged_lines)

    if second_norm.startswith("euerend") or second_norm.startswith("euer"):
        merged = f"{first}{lines[1].lstrip()}"
        merged_lines = [merged, *lines[2:]]
        return "\n".join(merged_lines)

    return text


def _apply_latin_micro_corrections_to_line(line: str) -> str:
    def _replace_token(match: re.Match[str]) -> str:
        token = match.group(0)
        lower = token.lower()
        replacement = LATIN_SAFE_CORRECTIONS.get(lower)
        if replacement is None:
            return token
        if _looks_name_like_token(token):
            return token
        return _apply_case(token, replacement)

    return TOKEN_RE.sub(_replace_token, line)


def apply_latin_micro_corrections(text: str, script_hint: str | None) -> str:
    if _normalize_script_hint(script_hint) != "latin":
        return text
    lines = _iter_lines(text)
    corrected = [_apply_latin_micro_corrections_to_line(line) for line in lines]
    return "\n".join(corrected)


def apply_archai_safe_normalizer(text: str, script_hint: str | None) -> str:
    if not text:
        return ""
    normalized = apply_fragment_merge(text)
    normalized = apply_decorated_initial_fix(normalized, script_hint)
    normalized = apply_latin_micro_corrections(normalized, script_hint)
    return normalized


class OcrProofreaderAgent:
    def __init__(self, *, client: SaiaClient | None = None, model_override: str | None = None) -> None:
        self.client = client or SaiaClient()
        self.model_override = (model_override or "").strip()

    def _select_model(self) -> str:
        if self.model_override:
            return self.model_override
        configured = str(settings.archai_chat_ai_model or "").strip()
        if configured:
            return configured
        models = self.client.list_models()
        if not models:
            raise RuntimeError("No model available for OCR proofreader.")
        return models[0]

    def proofread(self, ocr_text: str, script_hint: str | None, detected_language: str | None = None) -> str:
        source = (ocr_text or "").strip()
        if not source:
            return ""

        source = apply_archai_safe_normalizer(source, script_hint)

        # Compute pre-proofread sanity baseline
        from app.agents.saia_ocr_agent import compute_sanity
        pre_sanity = compute_sanity(source)

        response = self.client.chat_completion(
            model=self._select_model(),
            temperature=0.0,
            top_p=1.0,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": PALEO_PROOFREAD_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": PALEO_PROOFREAD_USER_PROMPT_TEMPLATE.format(
                        script_hint=(script_hint or "unknown"),
                        detected_language=(detected_language or "unknown"),
                        ocr_text=source,
                    ),
                },
            ],
        )

        cleaned = _clean_plain_text(str(response.get("text") or ""))
        if not cleaned:
            return source
        cleaned = apply_archai_safe_normalizer(cleaned, script_hint)

        # Post-proofread sanity guard: reject if proofreader over-corrected
        post_sanity = compute_sanity(cleaned)
        unc_dropped = (
            post_sanity["uncertainty_marker_ratio"] < pre_sanity["uncertainty_marker_ratio"] - 0.02
        )
        quality_worsened = (
            post_sanity["weird_ratio"] > pre_sanity["weird_ratio"] + 0.02
            or post_sanity["digit_ratio"] > pre_sanity["digit_ratio"] + 0.01
            or post_sanity["single_char_ratio"] > pre_sanity["single_char_ratio"] + 0.03
            or post_sanity.get("junk_ratio", 0) > pre_sanity.get("junk_ratio", 0) + 0.02
        )
        if unc_dropped and quality_worsened:
            # Proofreader removed uncertainty markers but introduced junk -> reject
            return source

        # Also reject if proofreader completely removed all uncertainty when input had some
        if (
            pre_sanity["uncertainty_marker_ratio"] > 0.01
            and post_sanity["uncertainty_marker_ratio"] == 0.0
        ):
            return source

        return cleaned
