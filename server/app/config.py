"""Central configuration — loaded from .env at startup."""

from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Resolve .env from repo root
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    # --- LLM ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # --- ElevenLabs ---
    eleven_api_key: str = ""
    eleven_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # default Rachel
    eleven_stt_model: str = "scribe_v1"

    # --- Google OAuth (shared: Gmail, Calendar, etc.) ---
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/auth/google/callback"

    # --- Microsoft Graph (Outlook) ---
    ms_tenant_id: str = ""
    ms_client_id: str = ""
    ms_client_secret: str = ""
    ms_redirect_uri: str = "http://localhost:8000/auth/outlook/callback"

    # --- Slack ---
    slack_bot_token: str = ""

    # --- Speaker verification ---
    speaker_verify_threshold: float = 0.65

    # --- Server ---
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    database_path: str = "data/assistant.db"
    audit_log_path: str = "data/audit.log"

    # --- Agentic ---
    max_agentic_steps: int = 5

    # --- Device control ---
    device_controller_timeout: float = 5.0

    # --- MQTT ---
    mqtt_broker_host: str = "localhost"
    mqtt_broker_port: int = 1883
    mqtt_username: str = ""
    mqtt_password: str = ""
    mqtt_topic_prefix: str = "nova"

    # --- Home Assistant ---
    ha_url: str = ""       # e.g. "http://homeassistant.local:8123"
    ha_token: str = ""     # Long-lived access token

    # --- Vision ---
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    presence_monitoring_enabled: bool = False
    presence_check_interval: int = 10
    vision_context_enabled: bool = True
    vision_context_capture_interval: float = 0.1       # Raw frame grab rate (persistent camera)
    vision_context_change_check_interval: float = 0.5  # OpenCV analysis rate (decimated)
    vision_context_describe_interval: float = 2.0      # LLM describe cadence (only on change)
    vision_context_buffer_seconds: float = 60.0
    vision_context_skip_static: bool = True             # Skip LLM describe when scene is static
    vision_context_local: bool = True                   # Use local SmolVLM instead of Groq API

    # --- Integrations ---
    openweathermap_api_key: str = ""
    spotify_access_token: str = ""

    # --- Personality ---
    assistant_name: str = "NOVA"

    model_config = {"env_file": str(_ENV_PATH), "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
