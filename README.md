# ArchAI Workspace

ArchAI now uses a **Document + Chat** workflow.

## What this app does

- Upload document page images in the workspace
- Navigate pages in a left document sidebar
- Chat with a selected model in the center panel
- Optionally attach the current page image for vision-capable models
- Proxy all model calls through backend endpoints (`/api/chat/*`)

Legacy OCR/HTR extraction flows were removed.

## Architecture

- Backend: FastAPI (`backend/app/main.py`)
- Frontend: Next.js app router (`frontend/src/app`)
- Chat provider: GWDG Chat AI OpenAI-compatible API (`https://chat-ai.academiccloud.de/v1`)

## Required environment variables

Set these in `backend/.env.local` (or `backend/.env`) or your shell:

```bash
CHAT_AI_API_KEY=...
CHAT_AI_BASE_URL=https://chat-ai.academiccloud.de/v1
SAIA_API_KEY=...
SAIA_BASE_URL=https://chat-ai.academiccloud.de/v1
SAIA_TIMEOUT_SECONDS=120
SAIA_MODELS_CACHE_TTL_SECONDS=300
SAIA_OCR_MODEL_PREFS=qwen3-vl-30b-a3b-instruct,internvl3.5-30b-a3b,mistral-large-3-675b-instruct-2512,gemma-3-27b-it
SAIA_OCR_TEMPERATURE=0
SAIA_OCR_MAX_TOKENS=8192
SAIA_MODEL_PROBE=0
ARCHAI_CHAT_AI_MODEL=meta-llama-3.1-8b-instruct
```

Do not commit API keys.

## Run locally

### 1) Backend

```bash
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 2) Frontend

```bash
cd frontend
npm run dev -- --hostname 127.0.0.1 --port 3001
```

Open: `http://127.0.0.1:3001/workspace`

## API endpoints

- `GET /api/chat/models`
  - Lists available model IDs from GWDG
- `POST /api/chat/completions`
  - Accepts chat messages, optional context, and optional streaming
  - Supports OpenAI-style multimodal message blocks for vision use cases
- `POST /api/ocr/saia`
  - Full-page SAIA OCR endpoint (server-side only, no browser key exposure)
  - Sends the current full page image directly to SAIA vision chat models
  - Returns `status`, `model_used`, `fallbacks`, `warnings`, `lines`, `text`, `script_hint`, `confidence`
- `POST /api/evidence/spans`
  - Stores OCR spans (text + coords + model/prompt/crop hash provenance)

## Workspace behavior

- Route: `/workspace` and `/workspace/[documentId]`
- Left panel:
  - document picker
  - current page preview
  - segmentation overlay toggle
  - page navigation + thumbnails
- Middle panel:
  - message history
  - model selector
  - include-page-image toggle
  - streaming assistant responses
  - `Extract (SAIA OCR)` action with popup, copy, insert-into-chat
