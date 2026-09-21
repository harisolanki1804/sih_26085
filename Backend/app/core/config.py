import os
from typing import List
from pydantic_settings import BaseSettings
from app.core.timeline_config import EVENT_CODE


class Settings(BaseSettings):
    PROJECT_NAME: str = "VARUNA Early Warning & Multi-Hazard Decision Platform"
    API_V1_STR: str = "/api/v1"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "AI-driven Multi-Hazard Early Warning System for Urban Flash Floods, "
        "featuring explainable risk scoring, DEM-based inundation heuristics, and historical event replay."
    )

    # Base Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")

    # Database Configuration (Defaults to SQLite for local development, switchable via env)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(os.path.dirname(BASE_DIR), 'varuna.db')}"
    )

    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "*"
    ]

    # Pilot Area Configurations
    DEFAULT_REGION_CODE: str = "IN-MH-BOM-01"
    DEFAULT_EVENT_CODE: str = EVENT_CODE

    # MOSDAC Satellite Data API (username/password only, no API key)
    MOSDAC_USERNAME: str = os.getenv("MOSDAC_USERNAME", "")
    MOSDAC_PASSWORD: str = os.getenv("MOSDAC_PASSWORD", "")

    # Offline GenAI (Ollama) — leave empty to use the built-in rule-based chatbot
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "phi3:mini")

    # Alert Webhook Dispatch — leave empty to disable
    ALERT_WEBHOOK_URL: str = os.getenv("ALERT_WEBHOOK_URL", "")
    ALERT_WEBHOOK_SECRET: str = os.getenv("ALERT_WEBHOOK_SECRET", "")

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
