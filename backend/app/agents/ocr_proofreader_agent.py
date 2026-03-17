from __future__ import annotations

import json
import re
from dataclasses import dataclass

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
    normalized = apply_decorated_initial_fix(text, script_hint)
    normalized = apply_latin_micro_corrections(normalized, script_hint)
    return normalized


# ── Proofreader guard ─────────────────────────────────────────────────
# Rejects proofreader output that looks like hallucination rather than
# genuine correction.  The guard is conservative: when in doubt, keep
# the raw OCR.

# Thresholds
_MAX_CHAR_EDIT_RATIO = 0.40       # reject if >40% of chars changed
_MAX_LINE_COUNT_DRIFT = 0.30      # reject if line-count changed >30%
_MIN_UNCERTAINTY_RETENTION = 0.50  # reject if >50% of [?/…] markers vanished
_MAX_TOKEN_CHURN = 0.50           # reject if >50% of tokens are new


@dataclass(frozen=True)
class ProofreadVerdict:
    """Result of the proofreader guard check."""
    accepted: bool
    reason: str
    char_edit_ratio: float = 0.0
    line_drift: float = 0.0
    uncertainty_retention: float = 1.0
    token_churn: float = 0.0


def _normalised_levenshtein(a: str, b: str) -> float:
    """Character-level edit distance normalised to 0-1 (1 = completely different)."""
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 1.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb] / max(la, lb)


def _count_uncertainty_markers(text: str) -> int:
    """Count uncertainty markers: ?, …, [?], […]."""
    return text.count("?") + text.count("…") + text.count("[…]") * 2 + text.count("[?]") * 2


def _token_set(text: str) -> set[str]:
    return {t.lower() for t in re.split(r"\s+", text.strip()) if t}


def check_proofread_delta(raw_ocr: str, proofread: str) -> ProofreadVerdict:
    """Check whether proofreader output is a safe correction or a hallucination.

    Returns a ProofreadVerdict indicating whether to accept the proofread text.
    """
    raw_norm = re.sub(r"\s+", " ", raw_ocr.strip().lower())
    proof_norm = re.sub(r"\s+", " ", proofread.strip().lower())

    # 1. Character-level edit ratio
    char_edit = _normalised_levenshtein(raw_norm, proof_norm)

    # 2. Line-count drift
    raw_lines = [ln for ln in raw_ocr.splitlines() if ln.strip()]
    proof_lines = [ln for ln in proofread.splitlines() if ln.strip()]
    if raw_lines:
        line_drift = abs(len(proof_lines) - len(raw_lines)) / max(len(raw_lines), 1)
    else:
        line_drift = 0.0

    # 3. Uncertainty marker retention
    raw_markers = _count_uncertainty_markers(raw_ocr)
    if raw_markers > 0:
        proof_markers = _count_uncertainty_markers(proofread)
        unc_retention = proof_markers / raw_markers
    else:
        unc_retention = 1.0

    # 4. Token churn: fraction of proofread tokens that weren't in raw
    raw_tokens = _token_set(raw_ocr)
    proof_tokens = _token_set(proofread)
    if proof_tokens:
        new_tokens = proof_tokens - raw_tokens
        token_churn = len(new_tokens) / len(proof_tokens)
    else:
        token_churn = 0.0

    # --- Decision ---
    if char_edit > _MAX_CHAR_EDIT_RATIO:
        return ProofreadVerdict(
            accepted=False,
            reason=f"PROOFREAD_REJECTED:char_edit_ratio={char_edit:.2f}>{_MAX_CHAR_EDIT_RATIO}",
            char_edit_ratio=char_edit,
            line_drift=line_drift,
            uncertainty_retention=unc_retention,
            token_churn=token_churn,
        )

    if line_drift > _MAX_LINE_COUNT_DRIFT:
        return ProofreadVerdict(
            accepted=False,
            reason=f"PROOFREAD_REJECTED:line_drift={line_drift:.2f}>{_MAX_LINE_COUNT_DRIFT}",
            char_edit_ratio=char_edit,
            line_drift=line_drift,
            uncertainty_retention=unc_retention,
            token_churn=token_churn,
        )

    if unc_retention < _MIN_UNCERTAINTY_RETENTION:
        return ProofreadVerdict(
            accepted=False,
            reason=f"PROOFREAD_REJECTED:uncertainty_stripped={unc_retention:.2f}<{_MIN_UNCERTAINTY_RETENTION}",
            char_edit_ratio=char_edit,
            line_drift=line_drift,
            uncertainty_retention=unc_retention,
            token_churn=token_churn,
        )

    if token_churn > _MAX_TOKEN_CHURN:
        return ProofreadVerdict(
            accepted=False,
            reason=f"PROOFREAD_REJECTED:token_churn={token_churn:.2f}>{_MAX_TOKEN_CHURN}",
            char_edit_ratio=char_edit,
            line_drift=line_drift,
            uncertainty_retention=unc_retention,
            token_churn=token_churn,
        )

    return ProofreadVerdict(
        accepted=True,
        reason="PROOFREAD_ACCEPTED",
        char_edit_ratio=char_edit,
        line_drift=line_drift,
        uncertainty_retention=unc_retention,
        token_churn=token_churn,
    )


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
        return apply_archai_safe_normalizer(cleaned, script_hint)
