import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # CORS
    cors_origins: str = (
        "http://localhost:3000,"
        "http://127.0.0.1:3000,"
        "http://localhost:3001,"
        "http://127.0.0.1:3001"
    )

    # Model paths
    model_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

    # Analytics
    analytics_db_path: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "analytics.db"
    )
    analytics_username: str = "admin"
    analytics_password: str = "layout2024"

    # JWT
    jwt_secret: str = "change-this-to-a-random-string"
    jwt_expiry_minutes: int = 60

    # Processing
    max_pool_workers: int = 3
    task_ttl_minutes: int = 60

    # Chat AI (GWDG OpenAI-compatible)
    chat_ai_api_key: str = ""
    chat_ai_base_url: str = "https://chat-ai.academiccloud.de/v1"
    archai_chat_ai_api_key: str = ""
    archai_chat_ai_base_url: str = "https://chat-ai.academiccloud.de/v1"
    archai_chat_ai_model: str = "qwen3-30b-a3b-instruct-2507"
    chat_rag_model: str = "qwen3-30b-a3b-instruct-2507"
    translation_model: str = "llama-3.3-70b-instruct"

    # SAIA OCR Agent configuration
    saia_api_key: str = ""
    saia_base_url: str = "https://chat-ai.academiccloud.de/v1"
    saia_timeout_seconds: int = 120
    saia_models_cache_ttl_seconds: int = 300
    saia_label_analysis_model: str = "qwen3-vl-30b-a3b-instruct"
    label_visual_model: str = "qwen3-vl-30b-a3b-instruct"
    label_visual_fallback_model: str = "internvl3.5-30b-a3b"
    saia_ocr_model_preferences: str = ""
    saia_ocr_model_prefs: str = ""
    saia_ocr_models: str = ""
    saia_ocr_temperature: float = 0.0
    saia_ocr_max_tokens: int = 4096
    ocr_crop_upscale: int = 2
    ocr_max_pixels_per_tile: int = 160000000
    ocr_max_long_edge: int = 12000
    ocr_image_size_retry_limit: int = 2
    ocr_image_retry_shrink: float = 0.82

    # Backward-compatible SAIA key names (legacy config)
    archai_saia_api_key: str = ""

    # RAG / ChromaDB
    chroma_persist_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), ".data", "chroma"
    )
    rag_collection_name: str = "archai_chunks"
    rag_entity_collection_name: str = "archai_entities"
    rag_embedding_model: str = "multilingual-e5-large-instruct"
    rag_top_k: int = 5
    rag_entity_top_k: int = 4
    rag_auto_index: bool = True

    # Authority source expansion
    geonames_base_url: str = "http://api.geonames.org"
    geonames_username: str = "demo"
    geonames_timeout_seconds: int = 10

    # Paleography verification
    paleography_verification_enabled: bool = True
    paleography_verification_model: str = "qwen3-235b-a22b"
    paleography_verification_temperature: float = 0.0
    paleography_verification_max_tokens: int = 900

    # OCR backend routing
    ocr_backend_default: str = "auto"

    # Kraken recognition
    kraken_device: str = "cpu"
    kraken_line_padding: int = 16
    kraken_models_dir: str = "weights/kraken_models"
    kraken_default_recognition_model_path: str = "weights/kraken_recognition.mlmodel"
    kraken_mccatmus_model_path: str = "weights/kraken_models/mccatmus.mlmodel"
    kraken_catmus_model_path: str = "weights/kraken_models/catmus_medieval.mlmodel"
    kraken_cremma_medieval_model_path: str = "weights/kraken_models/cremma_medieval.mlmodel"
    kraken_cremma_lat_model_path: str = "weights/kraken_models/cremma_medieval_lat.mlmodel"

    # Calamari recognition
    calamari_models_dir: str = "archai/vendor/layout/backend/weights/calamari_models"
    calamari_default_model_dir: str = "archai/vendor/layout/backend/weights/calamari_models/calamari_models_v1/gt4histocr"
    calamari_gt4histocr_model_dir: str = "archai/vendor/layout/backend/weights/calamari_models/calamari_models_v1/gt4histocr"
    calamari_historical_french_model_dir: str = "archai/vendor/layout/backend/weights/calamari_models/calamari_models_v1/historical_french"
    calamari_antiqua_historical_model_dir: str = "archai/vendor/layout/backend/weights/calamari_models/calamari_models_v1/antiqua_historical"
    calamari_fraktur_historical_model_dir: str = "archai/vendor/layout/backend/weights/calamari_models/calamari_models_v1/fraktur_historical"
    calamari_gothic_model_dir: str = "archai/vendor/layout/backend/weights/calamari_models/calamari_models_v1/gt4histocr"
    calamari_bastard_model_dir: str = "archai/vendor/layout/backend/weights/calamari_models/gt4histocr_tmp/deep3_htr-bastard"
    calamari_english_model_dir: str = "archai/vendor/layout/backend/weights/calamari_models/calamari_models_v1/uw3-modern-english"

    # GLM-OCR (self-hosted mode)
    glmocr_device: str = "0"
    glmocr_ollama_host: str = "http://localhost:11434"
    glmocr_ollama_model: str = "glm-ocr:latest"
    glmocr_ollama_timeout_seconds: int = 300
    glmocr_ollama_temperature: float = 0.0
    glmocr_ollama_retries_per_variant: int = 2
    glmocr_max_payload_bytes: int = 2000000

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def emanuskript_model_path(self) -> str:
        return os.path.join(self.model_dir, "best_emanuskript_segmentation.pt")

    @property
    def catmus_model_path(self) -> str:
        return os.path.join(self.model_dir, "best_catmus.pt")

    @property
    def zone_model_path(self) -> str:
        return os.path.join(self.model_dir, "best_zone_detection.pt")

    model_config = {"env_file": (".env", ".env.local"), "extra": "ignore"}


settings = Settings()
