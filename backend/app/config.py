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
    archai_chat_ai_model: str = "internvl3.5-30b-a3b"

    # SAIA OCR Agent configuration
    saia_api_key: str = ""
    saia_base_url: str = "https://chat-ai.academiccloud.de/v1"
    saia_timeout_seconds: int = 120
    saia_models_cache_ttl_seconds: int = 300
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
