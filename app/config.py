"""Runtime configuration, read once from environment variables."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Model
    model_id: str = os.environ.get("MODEL_ID", "openbmb/VoxCPM2")
    # Pinned Hugging Face commit so an upstream change can never silently alter what we serve.
    model_revision: str = os.environ.get("MODEL_REVISION", "32279effe8c19989596f05d353d1447f51d9e915")
    device: str = os.environ.get("DEVICE", "auto")  # auto | cuda | cuda:N | mps | cpu
    load_denoiser: bool = _env_bool("LOAD_DENOISER", False)

    # Security / limits
    api_key: str = os.environ.get("API_KEY", "").strip()
    generated_api_key: bool = False
    rate_limit_per_minute: int = _env_int("RATE_LIMIT_PER_MINUTE", 30)
    max_text_chars: int = _env_int("MAX_TEXT_CHARS", 1000)
    max_upload_bytes: int = _env_int("MAX_UPLOAD_MB", 10) * 1024 * 1024
    min_reference_seconds: float = _env_float("MIN_REFERENCE_SECONDS", 1.0)
    max_reference_seconds: float = _env_float("MAX_REFERENCE_SECONDS", 30.0)
    generate_timeout_seconds: float = _env_float("GENERATE_TIMEOUT_SECONDS", 120.0)
    cors_origins: list[str] = field(
        default_factory=lambda: [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
    )

    # Paths
    tmp_dir: str = os.environ.get("TMP_DIR", "/tmp/tts")


def load_settings() -> Settings:
    settings = Settings()
    if not settings.api_key:
        # No key configured: generate one for this process so the service is never open by accident.
        settings = Settings(api_key=secrets.token_urlsafe(24), generated_api_key=True)
    os.makedirs(settings.tmp_dir, exist_ok=True)
    return settings


settings = load_settings()
