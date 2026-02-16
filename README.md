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

Set these in `backend/.env` or your shell:

```bash
ARCHAI_CHAT_AI_API_KEY=...
ARCHAI_CHAT_AI_BASE_URL=https://chat-ai.academiccloud.de/v1
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

## Workspace behavior

- Route: `/workspace` and `/workspace/[documentId]`
- Left panel:
  - document picker
  - current page preview
  - page navigation + thumbnails
- Center panel:
  - message history
  - model selector
  - include-page-image toggle
  - streaming assistant responses
