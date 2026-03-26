from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests


DEFAULT_IMAGE_PATH = Path("/Users/mobasuony/Downloads/e-codices_fmb-cb-0001_001r_max.jpg")
DEFAULT_BASE_URL = "http://127.0.0.1:8000/api"
DEFAULT_METADATA = {
    "language": "old french",
    "year": "1290",
    "script_family": "Gothic French",
    "document_type": "manuscript",
    "place_or_origin": "unknown",
    "notes": "thesis validation page",
}
REQUIRED_SEGMENTATION_LABELS = {"Main script black", "Embellished", "Page Number", "Column"}
BAD_TRANSLATION_PHRASES = {
    "in the age of the child stretched out",
    "to spin ruin that comes",
    "where the fountains of the noble people",
    "the foreign people of villainy",
}
LETTER_ONLY_RE = re.compile(r"^(?:[A-Z]|UNKNOWN)$")
MULTI_LETTER_RE = re.compile(r"^Region\s+\d+:\s+(?:[A-Z]|UNKNOWN)$")
VERIFICATION_BLOCK_RE = re.compile(
    r"\[Verification\]\s*"
    r"Assessment:\s*.+\n"
    r"Verified answer:\s*.+\n"
    r"Notes:\s*.+\n"
    r"Citations checked:\s*.+\n"
    r"Verifier model:\s*.+",
    re.DOTALL,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the real thesis smoke pipeline on one fixed manuscript page.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--image-path", default=str(DEFAULT_IMAGE_PATH))
    parser.add_argument("--language", default=DEFAULT_METADATA["language"])
    parser.add_argument("--year", default=DEFAULT_METADATA["year"])
    parser.add_argument("--script-family", default=DEFAULT_METADATA["script_family"])
    parser.add_argument("--document-type", default=DEFAULT_METADATA["document_type"])
    parser.add_argument("--origin", default=DEFAULT_METADATA["place_or_origin"])
    parser.add_argument("--notes", default=DEFAULT_METADATA["notes"])
    parser.add_argument("--output-json", default="", help="Optional path to write full smoke results as JSON.")
    return parser


def _b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _api_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _translation_prompt(source_text: str, metadata: dict[str, str]) -> str:
    source_language = str(metadata.get("language") or "unknown").strip() or "unknown"
    year = str(metadata.get("year") or "").strip()
    script_family = str(metadata.get("script_family") or "").strip()
    place_or_origin = str(metadata.get("place_or_origin") or "").strip()
    document_type = str(metadata.get("document_type") or "").strip()
    notes = str(metadata.get("notes") or "").strip()
    return "\n".join(
        part
        for part in [
            f"Translate the extracted manuscript passage from {source_language} into fluent, faithful English.",
            "This is a translation request, not an OCR request.",
            "Use the extracted transcript below as the only source text.",
            f"Source language: {source_language}.",
            "Target language: English.",
            year and f"The manuscript year is {year}. Use this only as a weak historical-language hint.",
            script_family and f"The script family is {script_family}. Use this only as a weak reading hint.",
            place_or_origin and f"The manuscript origin is {place_or_origin}. Use this only as a weak dialect hint.",
            document_type and f"The document type is {document_type}. Use this only as a weak genre/context hint.",
            notes and f"Additional notes: {notes}. Use this only as weak context and never to invent text.",
            "Treat the transcript as the authoritative source passage to interpret and render into English.",
            (
                "Treat spelling variation, abbreviation, and likely OCR distortions as expected features of Old French and resolve them contextually when the intended reading is reasonably clear."
                if "old french" in source_language.lower()
                else "Use sentence-level and passage-level context to resolve obvious orthographic or OCR-like distortions where the intended sense is reasonably clear."
            ),
            "Translate at the level of clauses, sentences, and the whole passage, not as a word-by-word gloss.",
            "Produce the best coherent English rendering that the passage supports.",
            "Prefer fluent English syntax over literal token-by-token paraphrase whenever the context makes the intended sense reasonably clear.",
            "Keep the translation roughly proportional to the source passage; do not expand a damaged or repetitive passage into a longer narrative than the source supports.",
            "Silently normalize obvious OCR-like distortions internally when the likely source reading is clear from context.",
            "Do not turn an unclear token into a confident person, place, or plot detail unless the passage clearly supports that reading.",
            "If a clause remains too corrupt to interpret confidently, mark only that local span with [unclear] instead of inventing connective narrative or repeated moral commentary.",
            "Preserve uncertainty only where it is genuinely unavoidable after contextual interpretation.",
            "If a word or phrase remains unresolved, mark only that local span with [unclear] or [token?].",
            "Do not describe the page, layout, decoration, image, or OCR process.",
            "Do not add a translator's note, note, explanation, or editorial commentary.",
            "Return only the English translation.",
            "",
            "OCR-extracted source text:",
            source_text,
        ]
        if part
    )


def _post_json(session: requests.Session, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    response = session.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _post_segmentation(session: requests.Session, base_url: str, image_path: Path) -> dict[str, Any]:
    with image_path.open("rb") as handle:
        response = session.post(
            _api_url(base_url, "/predict/single"),
            files={"image": (image_path.name, handle, "image/jpeg")},
            data={"confidence": "0.25", "iou": "0.3"},
            timeout=300,
        )
    response.raise_for_status()
    return response.json()


def _label_regions(coco: dict[str, Any], label_name: str) -> list[dict[str, Any]]:
    categories = {int(item["id"]): str(item["name"]) for item in list(coco.get("categories") or []) if isinstance(item, dict) and "id" in item and "name" in item}
    regions: list[dict[str, Any]] = []
    for annotation in list(coco.get("annotations") or []):
        if not isinstance(annotation, dict):
            continue
        if categories.get(int(annotation.get("category_id", -1))) != label_name:
            continue
        bbox = list(annotation.get("bbox") or [])
        if len(bbox) < 4:
            continue
        x, y, w, h = [float(value) for value in bbox[:4]]
        segmentation = annotation.get("segmentation")
        polygons: list[list[float]] = []
        if isinstance(segmentation, list):
            if segmentation and isinstance(segmentation[0], (int, float)):
                polygons = [[float(value) for value in segmentation]]
            else:
                polygons = [
                    [float(value) for value in poly]
                    for poly in segmentation
                    if isinstance(poly, list) and len(poly) >= 6
                ]
        regions.append(
            {
                "region_id": str(annotation.get("id") or ""),
                "bbox_xyxy": [x, y, x + w, y + h],
                "polygons": polygons,
            }
        )
    regions.sort(key=lambda row: (row["bbox_xyxy"][1], row["bbox_xyxy"][0]))
    return regions


def _evaluation_row(name: str, passed: bool, summary: str, *, snippet: str = "", details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "stage": name,
        "passed": bool(passed),
        "summary": summary,
        "snippet": snippet.strip(),
        "details": details or {},
    }


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _translation_pass(text: str, *, source_text: str) -> tuple[bool, str]:
    clean = _clean_text(text)
    lower = clean.lower()
    if not clean:
        return False, "empty translation"
    if any(phrase in lower for phrase in BAD_TRANSLATION_PHRASES):
        return False, "translation still contains previously observed absurd literal artifacts"
    if lower.startswith(("this page ", "the page ", "the manuscript ", "the image ")):
        return False, "translation looks like a page/image description"
    if "ocr" in lower and "translation" in lower:
        return False, "translation contains pipeline/meta commentary"
    if "note:" in lower or "translator's note" in lower:
        return False, "translation still includes editorial note/commentary"
    if source_text and len(clean) > max(int(len(source_text) * 2.2), 4200):
        return False, "translation appears over-expanded relative to the source passage"
    return True, "clean text-only English translation returned"


def _letter_answer_pass(text: str, expected_count: int | None = None) -> tuple[bool, str]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return False, "empty letter-identification response"
    if len(lines) == 1 and LETTER_ONLY_RE.fullmatch(lines[0]):
        return True, "single letter-only answer returned"
    if all(MULTI_LETTER_RE.fullmatch(line) for line in lines):
        if expected_count is not None and len(lines) < expected_count:
            return False, f"returned {len(lines)} letter lines for {expected_count} regions"
        return True, "deterministic per-region letter answers returned"
    return False, "response contains prose instead of letter-only output"


def _presentation_pass(text: str) -> tuple[bool, str]:
    clean = str(text or "").strip()
    lower = clean.lower()
    if not clean:
        return False, "empty presentation view"
    if len(clean) > 3500:
        return False, "presentation view is too long"
    if "linked entities" not in lower or "unresolved mentions" not in lower:
        return False, "presentation view is missing core entity/unresolved sections"
    if any(token in lower for token in ("moral allegory", "didactic function", "likely poetic", "legal fragment")):
        return False, "presentation view still includes speculative interpretation"
    return True, "concise evidence-first presentation view returned"


def _debug_pass(text: str) -> tuple[bool, str]:
    clean = str(text or "").strip()
    if not clean:
        return False, "empty debug view"
    match = re.search(r"```json\s*(\{.*\})\s*```", clean, re.DOTALL)
    if not match:
        return False, "debug view is not a consistent JSON-in-markdown artifact"
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return False, "debug JSON block could not be parsed"
    required = {"query", "run_id", "summary", "authority_summary"}
    if not required.issubset(payload.keys()):
        return False, "debug view is missing required structured keys"
    return True, "structured debug artifact returned"


def _verification_pass(text: str, verification: dict[str, Any] | None) -> tuple[bool, str]:
    body = str(text or "")
    if body.count("[Verification]") != 1:
        return False, "verification block is missing or duplicated"
    if not VERIFICATION_BLOCK_RE.search(body):
        return False, "verification block is not canonical"
    if any(token in body for token in ("draft answer retained", "<think>", "```json", "{\"assessment\"")):
        return False, "verification leaked malformed or internal content"
    if not isinstance(verification, dict):
        return False, "verification payload missing"
    return True, "canonical verification block returned"


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    image_path = Path(args.image_path).expanduser().resolve()
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    metadata = {
        "language": str(args.language),
        "year": str(args.year),
        "script_family": str(args.script_family),
        "document_type": str(args.document_type),
        "place_or_origin": str(args.origin),
        "notes": str(args.notes),
    }
    image_b64 = _b64_file(image_path)
    session = requests.Session()

    results: dict[str, Any] = {"fixture": {"image_path": str(image_path), "metadata": metadata}, "stages": {}}

    segmentation_payload = _post_segmentation(session, args.base_url, image_path)
    coco = dict(segmentation_payload.get("coco_json") or {})
    labels = sorted({str(item.get("name")) for item in list(coco.get("categories") or []) if isinstance(item, dict) and item.get("name")})
    missing_labels = sorted(REQUIRED_SEGMENTATION_LABELS - set(labels))
    results["stages"]["segmentation"] = _evaluation_row(
        "segmentation",
        not missing_labels,
        "required labels present" if not missing_labels else f"missing labels: {', '.join(missing_labels)}",
        details={"labels": labels, "stats": segmentation_payload.get("stats") or {}},
    )

    extraction_payload = {
        "document_id": "thesis-smoke-document",
        "page_id": image_path.name,
        "image_id": image_path.name,
        "image_b64": image_b64,
        "language_hint": metadata["language"],
        "script_hint_seed": metadata["script_family"],
        "apply_proofread": True,
        "location_suggestions": [],
        "regions": [],
        "metadata": metadata,
    }
    extraction = _post_json(session, _api_url(args.base_url, "/ocr/extract_full_page"), extraction_payload, timeout=1800)
    transcript = str(extraction.get("text") or "").strip()
    run_id = str(extraction.get("run_id") or "").strip()
    authority_report = str(extraction.get("authority_report") or "").strip()
    results["stages"]["extraction"] = _evaluation_row(
        "extraction",
        bool(transcript and run_id and int(extraction.get("chunks_count") or 0) > 0),
        "non-empty transcript and persisted chunk pipeline returned" if transcript and run_id else "missing transcript or run_id",
        snippet=transcript[:220],
        details={
            "status": extraction.get("status"),
            "run_id": run_id,
            "chunks_count": extraction.get("chunks_count"),
            "mentions_count": extraction.get("mentions_count"),
            "model_used": extraction.get("model_used"),
        },
    )

    shared_context = {
        "ocr_run_id": run_id,
        "transcript": transcript,
        "authority_report": authority_report,
        "document_language": metadata["language"],
        "document_year": metadata["year"],
        "place_or_origin": metadata["place_or_origin"],
        "script_family": metadata["script_family"],
        "document_type": metadata["document_type"],
        "document_notes": metadata["notes"],
    }

    translation_response = _post_json(
        session,
        _api_url(args.base_url, "/chat/completions"),
        {
            "messages": [{"role": "user", "content": _translation_prompt(transcript, metadata)}],
            "stream": False,
            "context": {**shared_context, "chat_stage": "translation"},
        },
        timeout=900,
    )
    translation_text = str(translation_response.get("text") or "").strip()
    translation_ok, translation_summary = _translation_pass(translation_text, source_text=transcript)
    results["stages"]["translation"] = _evaluation_row(
        "translation",
        translation_ok
        and str(((translation_response.get("inspection") or {}).get("input_source_summary") or {}).get("source_type") or "") == "extracted_transcript",
        translation_summary,
        snippet=translation_text[:300],
        details={
            "model_used": (translation_response.get("stage_metadata") or {}).get("model_used"),
            "stage_metadata": translation_response.get("stage_metadata"),
            "inspection": translation_response.get("inspection"),
        },
    )

    rag_presentation = _post_json(
        session,
        _api_url(args.base_url, "/chat/completions"),
        {"messages": [{"role": "user", "content": "print the RAG"}], "stream": False, "context": {"ocr_run_id": run_id}},
        timeout=120,
    )
    rag_presentation_text = str(rag_presentation.get("text") or "")
    rag_presentation_ok, rag_presentation_summary = _presentation_pass(rag_presentation_text)
    results["stages"]["rag_presentation"] = _evaluation_row(
        "rag_presentation",
        rag_presentation_ok,
        rag_presentation_summary,
        snippet=rag_presentation_text[:260],
        details={"stage_metadata": rag_presentation.get("stage_metadata"), "inspection": rag_presentation.get("inspection")},
    )

    rag_debug = _post_json(
        session,
        _api_url(args.base_url, "/chat/completions"),
        {"messages": [{"role": "user", "content": "print the RAG debug"}], "stream": False, "context": {"ocr_run_id": run_id}},
        timeout=120,
    )
    rag_debug_text = str(rag_debug.get("text") or "")
    rag_debug_ok, rag_debug_summary = _debug_pass(rag_debug_text)
    results["stages"]["rag_debug"] = _evaluation_row(
        "rag_debug",
        rag_debug_ok,
        rag_debug_summary,
        snippet=rag_debug_text[:260],
        details={"stage_metadata": rag_debug.get("stage_metadata"), "inspection": rag_debug.get("inspection")},
    )

    authority_answer = _post_json(
        session,
        _api_url(args.base_url, "/chat/completions"),
        {
            "messages": [{"role": "user", "content": "what are the authority links and entities you extracted"}],
            "stream": False,
            "context": {**shared_context, "chat_stage": "entity_qa"},
        },
        timeout=900,
    )
    authority_text = str(authority_answer.get("text") or "")
    authority_lower = authority_text.lower()
    authority_report_lower = authority_report.lower()
    authority_ok = all(token not in authority_lower and token not in authority_report_lower for token in ("enfant [place]", "vilanie [place]"))
    results["stages"]["authority_extraction_linking"] = _evaluation_row(
        "authority_extraction_linking",
        authority_ok,
        "concise authority answer without weak lexical place typing" if authority_ok else "weak lexical item still surfaced as place",
        snippet=authority_text[:320],
        details={
            "authority_report": authority_report,
            "stage_metadata": authority_answer.get("stage_metadata"),
            "inspection": authority_answer.get("inspection"),
        },
    )

    embellished_regions = _label_regions(coco, "Embellished")
    page_number_regions = _label_regions(coco, "Page Number")

    embellished_single = _post_json(
        session,
        _api_url(args.base_url, "/chat/label-analysis"),
        {
            "question": "what is the embellished letter",
            "label_name": "Embellished",
            "image_b64": image_b64,
            "filename": image_path.name,
            "regions": embellished_regions,
        },
        timeout=180,
    )
    embellished_single_text = str(embellished_single.get("text") or "").strip()
    embellished_single_ok, embellished_single_summary = _letter_answer_pass(embellished_single_text)
    results["stages"]["embellished_letter_singular"] = _evaluation_row(
        "embellished_letter_singular",
        embellished_single_ok,
        embellished_single_summary,
        snippet=embellished_single_text,
        details={"analysis_mode": embellished_single.get("analysis_mode"), "stage_metadata": embellished_single.get("stage_metadata")},
    )

    embellished_plural = _post_json(
        session,
        _api_url(args.base_url, "/chat/label-analysis"),
        {
            "question": "what are the embellished letters",
            "label_name": "Embellished",
            "image_b64": image_b64,
            "filename": image_path.name,
            "regions": embellished_regions,
        },
        timeout=180,
    )
    embellished_plural_text = str(embellished_plural.get("text") or "").strip()
    embellished_plural_ok, embellished_plural_summary = _letter_answer_pass(embellished_plural_text, expected_count=max(1, len(embellished_regions)))
    results["stages"]["embellished_letter_plural"] = _evaluation_row(
        "embellished_letter_plural",
        embellished_plural_ok,
        embellished_plural_summary,
        snippet=embellished_plural_text,
        details={"analysis_mode": embellished_plural.get("analysis_mode"), "stage_metadata": embellished_plural.get("stage_metadata")},
    )

    page_number_answer = _post_json(
        session,
        _api_url(args.base_url, "/chat/label-analysis"),
        {
            "question": "what is the page number",
            "label_name": "Page Number",
            "image_b64": image_b64,
            "filename": image_path.name,
            "regions": page_number_regions,
        },
        timeout=180,
    )
    page_number_text = str(page_number_answer.get("text") or "").strip()
    page_number_lower = page_number_text.lower()
    page_number_ok = (
        bool(page_number_regions)
        and bool(page_number_text)
        and "no page number detected" not in page_number_lower
        and "strict diplomatic transcription task" not in page_number_lower
        and "preserve reading order exactly" not in page_number_lower
    )
    results["stages"]["page_number_qa"] = _evaluation_row(
        "page_number_qa",
        page_number_ok,
        "page number crop answered through label QA" if page_number_ok else "page-number response was missing or ungrounded",
        snippet=page_number_text,
        details={"analysis_mode": page_number_answer.get("analysis_mode"), "ocr_text": page_number_answer.get("ocr_text"), "stage_metadata": page_number_answer.get("stage_metadata")},
    )

    verification_ok, verification_summary = _verification_pass(authority_text, authority_answer.get("verification"))
    results["stages"]["verification"] = _evaluation_row(
        "verification",
        verification_ok,
        verification_summary,
        snippet=authority_text[-400:],
        details={"verification": authority_answer.get("verification")},
    )

    results["all_passed"] = all(bool(stage.get("passed")) for stage in results["stages"].values())
    return results


def _print_report(results: dict[str, Any]) -> None:
    print("Thesis Smoke Report")
    print("===================")
    for key, stage in results.get("stages", {}).items():
        status = "PASS" if stage.get("passed") else "FAIL"
        print(f"- {key}: {status} | {stage.get('summary', '')}")
        snippet = str(stage.get("snippet") or "").strip()
        if snippet:
            print(f"  snippet: {snippet[:240]}")
    print("")
    print(f"Overall: {'PASS' if results.get('all_passed') else 'FAIL'}")


def main() -> int:
    args = _parser().parse_args()
    results = run_smoke(args)
    _print_report(results)
    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Full results written to {out_path}")
    return 0 if results.get("all_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
