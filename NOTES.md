# ArchAI Migration Notes (Kraken Removal -> Document + Chat Workspace)

## Phase 0 reconnaissance

- Backend framework: FastAPI (`backend/app/main.py`), routers under `backend/app/routers`.
- Existing API routes:
  - `health`, `classes`, `predict`, `download`, `analytics`, `extract`.
- Existing extraction stack:
  - Route: `backend/app/routers/extract.py`
  - Services: `backend/app/services/kraken_extraction.py`, `backend/app/services/extract_jobs.py`, `backend/app/services/kraken_model_manager.py`, `backend/app/services/model_config.py`, `backend/app/services/saia_extraction.py`.
- Config system:
  - Pydantic settings in `backend/app/config.py` + `.env` support.
- Frontend framework:
  - Next.js app router (`frontend/src/app`).
  - Current analysis workspace route: `/analyze` with left/canvas/right panels.
  - Extraction UI/state heavily tied to Kraken in `frontend/src/lib/atoms/*`, `frontend/src/components/workspace/*`, `frontend/src/lib/api/extract.ts`, `frontend/src/lib/api/modelSetup.ts`, `frontend/src/lib/types/extract.ts`, `frontend/src/lib/types/modelSetup.ts`.
- Document image/page storage/serving (current):
  - Backend task files in `backend/.tasks/<task_id>` via `backend/app/services/file_manager.py`.
  - Frontend currently also uses browser object URLs for local previews.

## Planned deletions (Kraken and extraction subsystem)

- Backend files:
  - `backend/app/routers/extract.py`
  - `backend/app/services/kraken_extraction.py`
  - `backend/app/services/extract_jobs.py`
  - `backend/app/services/kraken_model_manager.py`
  - `backend/app/services/model_config.py`
  - `backend/app/services/saia_extraction.py` (legacy extraction path)
  - Kraken-only tests in `backend/tests/test_adaptive_selection.py`, `backend/tests/test_extraction_quality.py`, `backend/tests/test_saia_extraction.py`
- Frontend files/modules (Kraken/extraction-specific):
  - `frontend/src/lib/api/extract.ts`
  - `frontend/src/lib/api/modelSetup.ts`
  - `frontend/src/lib/types/extract.ts`
  - `frontend/src/lib/types/modelSetup.ts`
  - Kraken-bound workspace atoms/components under `frontend/src/lib/atoms/*`, `frontend/src/components/workspace/*`, and `frontend/src/components/results/KrakenTextModal.tsx`
  - Old analysis routes (`/analyze`, `/single`, `/batch`) replaced by new workspace route

## Dependency cleanup

- Remove backend dependency: `kraken` from `backend/pyproject.toml`.
- Keep `openai` for chat proxy to GWDG OpenAI-compatible API.

## New modules to add

- Backend:
  - `backend/app/services/chat_ai.py` (proxy/service layer to GWDG API)
  - `backend/app/routers/chat.py` (`GET /api/chat/models`, `POST /api/chat/completions`)
  - `backend/tests/test_chat_api.py` (mocked API calls / env usage)
- Frontend:
  - Route: `/workspace` and `/workspace/[documentId]`
  - `frontend/src/components/workspace/DocumentChatWorkspace.tsx`
  - `frontend/src/lib/api/chat.ts`
  - `frontend/src/lib/workspace/types.ts` (document/page/chat types)

## Target architecture after migration

- Main UX: document sidebar + central chat panel.
- Chat goes through backend proxy only; API keys remain server-side via env vars.
- Request context includes document/page metadata and optional page image payload for vision-capable models.
