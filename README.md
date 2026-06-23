# ArchAI Layout Workspace

ArchAI is a local-first workspace for medieval manuscript page analysis. It pairs
a FastAPI backend with a Next.js document workspace for segmentation, OCR,
grounded chat, translation, and entity analysis.

## Capabilities

- Upload and navigate multi-page manuscript images
- Run layout segmentation and inspect the overlay
- Extract full-page text through the configured GLM OCR runtime
- Crop and analyze labeled manuscript regions
- Chat over the current document, including translation and entity questions
- Persist OCR traces, evidence spans, quality signals, and authority-linking data

## Architecture

- `backend/app/main.py`: ASGI entry point
- `backend/app/core/`: application factory and model-pool lifecycle
- `backend/app/api/`: public router registration
- `backend/app/routers/`: HTTP endpoints
- `backend/app/services/`: OCR, RAG, evidence, and model integrations
- `backend/app/agents/`: OCR and analysis orchestration
- `frontend/src/components/workspace/`: workspace UI orchestration
- `frontend/src/features/workspace/`: workspace constants, browser file helpers,
  command intent parsing, and segmentation utilities
- `frontend/src/lib/`: API clients and shared workspace types

## Configuration

Copy the backend environment template and provide the services you intend to use:

```bash
cd backend
cp .env.example .env
```

The Chat AI and SAIA keys are optional until chat or vision analysis is used. OCR
requires a reachable GLM/Ollama runtime by default. Never commit API keys or
local model weights.

Important variables include:

```bash
CHAT_AI_API_KEY=...
SAIA_API_KEY=...
GLMOCR_OLLAMA_HOST=http://localhost:11434
GLMOCR_OLLAMA_MODEL=glm-ocr:latest
```

## Run Locally

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev -- --hostname 127.0.0.1 --port 3001
```

Open [http://127.0.0.1:3001/workspace](http://127.0.0.1:3001/workspace).

## Validation

```bash
cd backend
python -m pytest -q

cd ../frontend
npm run lint
npm run build
```

## API Highlights

- `GET /api/health`: service status
- `GET /api/chat/models`: available chat models
- `POST /api/chat/completions`: chat and streaming responses
- `POST /api/predict`: page segmentation
- `POST /api/ocr/extract_full_page`: full-page OCR with persisted analysis
- `POST /api/ocr/page_with_trace`: segmented OCR trace pipeline
- `GET /api/ocr/trace/{run_id}`: persisted trace snapshot
