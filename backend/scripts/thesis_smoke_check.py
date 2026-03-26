from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _bootstrap_backend_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_bootstrap_backend_path()

from app.agents.label_analysis_agent import LabelAnalysisAgent  # noqa: E402
from app.agents.paleography_verification_agent import PaleographyVerificationAgent  # noqa: E402
from app.services.chat_ai import build_rag_evidence_for_debug, create_chat_completion  # noqa: E402


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_regions(path: Path) -> list[dict[str, Any]]:
    raw = _load_json(path)
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        regions = raw.get("regions")
        if isinstance(regions, list):
            return [item for item in regions if isinstance(item, dict)]
    raise SystemExit("regions JSON must be a list of regions or an object with a 'regions' list.")


def _b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a lightweight thesis smoke check against one already-extracted page/run.",
    )
    parser.add_argument("--run-id", required=True, help="Existing OCR run_id")
    parser.add_argument("--transcript-file", required=True, help="Path to extracted transcript text file")
    parser.add_argument("--page-image-path", required=True, help="Path to the original page image")
    parser.add_argument("--regions-json-file", required=True, help="Path to segmentation regions JSON for one label")
    parser.add_argument("--label-name", default="Embellished", help="Segmentation label to test")
    parser.add_argument(
        "--translation-request",
        default="Translate the extracted text into English.",
        help="Translation prompt for the transcript",
    )
    parser.add_argument(
        "--entity-question",
        default="Which linked entities are present on this page?",
        help="Authority/entity QA prompt",
    )
    parser.add_argument(
        "--label-question",
        default="Which letter is this embellished initial?",
        help="Label-analysis question",
    )
    parser.add_argument(
        "--verification-question",
        default="Which linked entities are present on this page?",
        help="Verification question",
    )
    parser.add_argument("--document-language", default="", help="Uploaded manuscript language")
    parser.add_argument("--document-year", default="", help="Uploaded manuscript year")
    parser.add_argument("--place-or-origin", default="", help="Uploaded manuscript origin/place")
    parser.add_argument("--script-family", default="", help="Uploaded manuscript script family")
    parser.add_argument("--document-type", default="", help="Uploaded manuscript type")
    parser.add_argument("--document-notes", default="", help="Uploaded manuscript notes")
    return parser


def main() -> int:
    args = _parser().parse_args()
    transcript_path = Path(args.transcript_file).expanduser().resolve()
    page_image_path = Path(args.page_image_path).expanduser().resolve()
    regions_path = Path(args.regions_json_file).expanduser().resolve()

    transcript = _load_text(transcript_path)
    regions = _load_regions(regions_path)
    page_image_b64 = _b64_file(page_image_path)

    shared_context = {
        "ocr_run_id": args.run_id,
        "transcript": transcript,
        "document_language": args.document_language,
        "document_year": args.document_year,
        "place_or_origin": args.place_or_origin,
        "script_family": args.script_family,
        "document_type": args.document_type,
        "document_notes": args.document_notes,
    }

    translation = create_chat_completion(
        messages=[{"role": "user", "content": args.translation_request}],
        context={**shared_context, "chat_stage": "translation"},
        stream=False,
    )

    entity_answer = create_chat_completion(
        messages=[{"role": "user", "content": args.entity_question}],
        context={**shared_context, "chat_stage": "entity_qa"},
        stream=False,
    )

    rag_presentation = create_chat_completion(
        messages=[{"role": "user", "content": "print the RAG"}],
        context={"ocr_run_id": args.run_id},
        stream=False,
    )
    rag_debug = create_chat_completion(
        messages=[{"role": "user", "content": "print the RAG debug"}],
        context={"ocr_run_id": args.run_id},
        stream=False,
    )

    label_result = LabelAnalysisAgent().run(
        SimpleNamespace(
            question=args.label_question,
            label_name=args.label_name,
            image_b64=page_image_b64,
            filename=page_image_path.name,
            regions=regions,
        )
    )

    rag_debug_payload = build_rag_evidence_for_debug(args.entity_question, args.run_id)
    verification = PaleographyVerificationAgent().run(
        SimpleNamespace(
            question=args.verification_question,
            draft_answer=str(entity_answer.get("text") or "").strip(),
            transcript=transcript,
            authority_report="",
            evidence_text=str(rag_debug_payload.get("evidence_text") or ""),
            ocr_run_id=args.run_id,
        )
    )

    summary = {
        "translation": translation,
        "entity_qa": entity_answer,
        "rag_presentation": rag_presentation,
        "rag_debug": rag_debug,
        "label_analysis": {
            "status": label_result.status,
            "text": label_result.text,
            "label_name": label_result.label_name,
            "analysis_mode": label_result.analysis_mode,
            "model_used": label_result.model_used,
            "stage_metadata": label_result.stage_metadata,
            "inspection": label_result.inspection,
        },
        "verification": {
            "assessment": verification.assessment,
            "corrected_answer": verification.corrected_answer,
            "notes": verification.notes,
            "citations_checked": verification.citations_checked,
            "model_used": verification.model_used,
            "stage_metadata": verification.stage_metadata,
            "inspection": verification.inspection,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
