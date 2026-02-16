from fastapi import APIRouter

from app.services import inference

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": inference._model_pool is not None,
    }
