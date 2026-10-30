"""FastAPI service serving the VoxCPM2 text-to-speech model.

Endpoints
---------
GET  /health            liveness + model status (no auth)
GET  /ready             200 once the model is loaded, 503 before (used by Docker healthcheck)
POST /v1/tts            JSON  -> WAV   plain TTS, optional voice description
POST /v1/tts/clone      form  -> WAV   voice cloning from an uploaded reference clip
POST /v1/audio/speech   JSON  -> WAV   OpenAI-compatible alias of /v1/tts
GET  /                  tiny web UI
GET  /docs              Swagger UI
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from . import audio
from .config import settings
from .engine import EngineBusy, EngineNotReady, TTSEngine
from .security import enforce_rate_limit, require_api_key

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tts.api")
for _noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

STATIC_DIR = Path(__file__).parent / "static"
MAX_DESCRIPTION_CHARS = 300


# --------------------------------------------------------------------------- schemas
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=settings.max_text_chars, description="Text to speak.")
    voice_description: Optional[str] = Field(
        None,
        max_length=MAX_DESCRIPTION_CHARS,
        description="Optional voice design, e.g. 'A deep, calm male narrator'. Omit for the default voice.",
        examples=["A warm, gentle female voice"],
    )
    cfg_value: float = Field(2.0, ge=1.0, le=4.0, description="Guidance scale. 2.0 is the recommended default.")
    inference_timesteps: int = Field(10, ge=4, le=30, description="Diffusion steps. More = slower, slightly better.")
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1, description="Fix for reproducible output.")


class OpenAISpeechRequest(BaseModel):
    """Subset of OpenAI's /v1/audio/speech body so existing clients work unchanged."""

    model_config = ConfigDict(protected_namespaces=())

    model: str = Field("voxcpm2", description="Ignored; kept for compatibility.")
    input: str = Field(..., min_length=1, max_length=settings.max_text_chars)
    voice: str = Field("default", description="'default' or a free-text voice description.")
    response_format: str = Field("wav", description="Only 'wav' is supported.")
    speed: Optional[float] = Field(None, description="Ignored; kept for compatibility.")


# --------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = getattr(app.state, "engine", None)
    if engine is None:
        engine = TTSEngine(
            model_id=settings.model_id,
            revision=settings.model_revision,
            device_preference=settings.device,
            load_denoiser=settings.load_denoiser,
            queue_timeout=settings.generate_timeout_seconds,
        )
        app.state.engine = engine
    if not engine.ready and engine.load_error is None:
        # Load in the background so /health answers immediately while weights download.
        threading.Thread(target=engine.load, name="model-loader", daemon=True).start()
    if settings.generated_api_key:
        log.warning("API_KEY was not set. Generated a key for this run: %s", settings.api_key)
        log.warning("Set API_KEY in your environment (or .env) to use a stable key.")
    yield


app = FastAPI(
    title="VoxCPM2 TTS Service",
    version="1.0.0",
    description=(
        "Text-to-speech powered by [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2). "
        "Generate speech from text, design a voice with a short description, or clone a voice from a "
        "reference clip. All protected endpoints need an API key (`X-API-Key` header or `Authorization: Bearer`)."
    ),
    lifespan=lifespan,
)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "X-API-Key", "Content-Type"],
    )


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    log.info("%s %s -> %s (%.0f ms) rid=%s", request.method, request.url.path, response.status_code, elapsed_ms, request_id)
    return response


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "-")
    log.exception("Unhandled error rid=%s", rid)
    # Never leak stack traces or internal paths to clients.
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "request_id": rid})


# --------------------------------------------------------------------------- helpers
def _engine(request: Request) -> TTSEngine:
    return request.app.state.engine


async def _synthesize_response(engine: TTSEngine, **kwargs) -> Response:
    t0 = time.perf_counter()
    try:
        wav = await run_in_threadpool(engine.synthesize, **kwargs)
    except EngineNotReady as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Model is not ready: {exc}. Check /health and retry.",
            headers={"Retry-After": "15"},
        ) from None
    except EngineBusy:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Server is busy, please retry.", headers={"Retry-After": "5"}
        ) from None
    except ValueError as exc:  # input rejected by the model itself
        raise HTTPException(422, str(exc)) from None

    seconds = len(wav) / engine.sample_rate
    gen_ms = (time.perf_counter() - t0) * 1000
    body = audio.encode_wav(wav, engine.sample_rate)
    return Response(
        content=body,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'inline; filename="speech.wav"',
            "X-AI-Generated": "true",
            "X-Audio-Seconds": f"{seconds:.2f}",
            "X-Sample-Rate": str(engine.sample_rate),
            "X-Generation-Ms": f"{gen_ms:.0f}",
        },
    )


protected = [Depends(enforce_rate_limit), Depends(require_api_key)]


# --------------------------------------------------------------------------- routes
@app.get("/", include_in_schema=False)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health", tags=["ops"])
async def health(request: Request) -> dict:
    engine = _engine(request)
    return {
        "status": "ok",
        "model": engine.model_id,
        "revision": engine.revision,
        "model_loaded": engine.ready,
        "load_error": engine.load_error,
        "device": engine.device,
        "sample_rate": engine.sample_rate,
        "load_seconds": engine.load_seconds,
        "limits": {
            "max_text_chars": settings.max_text_chars,
            "max_upload_mb": settings.max_upload_bytes // (1024 * 1024),
            "max_reference_seconds": settings.max_reference_seconds,
            "rate_limit_per_minute": settings.rate_limit_per_minute,
        },
    }


@app.get("/ready", tags=["ops"])
async def ready(request: Request) -> JSONResponse:
    engine = _engine(request)
    if engine.ready:
        return JSONResponse({"ready": True, "device": engine.device})
    return JSONResponse({"ready": False, "load_error": engine.load_error}, status_code=503)


@app.post(
    "/v1/tts",
    tags=["tts"],
    dependencies=protected,
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}, "description": "16-bit PCM WAV, mono, 48 kHz"}},
    summary="Generate speech (default voice or designed voice)",
)
async def tts(body: TTSRequest, request: Request) -> Response:
    return await _synthesize_response(
        _engine(request),
        text=body.text,
        voice_description=body.voice_description,
        cfg_value=body.cfg_value,
        inference_timesteps=body.inference_timesteps,
        seed=body.seed,
    )


@app.post(
    "/v1/tts/clone",
    tags=["tts"],
    dependencies=protected,
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}, "description": "16-bit PCM WAV, mono, 48 kHz"}},
    summary="Clone a voice from a reference clip",
)
async def tts_clone(
    request: Request,
    text: str = Form(..., min_length=1, max_length=settings.max_text_chars, description="Text to speak."),
    reference_audio: Optional[UploadFile] = File(
        None, description="1-30 s clip of the voice to clone (WAV/FLAC/OGG/MP3). Omit for the default voice."
    ),
    prompt_text: Optional[str] = Form(
        None, max_length=settings.max_text_chars, description="Exact transcript of the clip. Optional, improves similarity."
    ),
    style: Optional[str] = Form(
        None, max_length=MAX_DESCRIPTION_CHARS, description="Optional style instruction, e.g. 'cheerful, slightly faster'."
    ),
    consent: bool = Form(False, description="Must be true: you confirm you have the speaker's permission to clone their voice."),
    cfg_value: float = Form(2.0, ge=1.0, le=4.0),
    inference_timesteps: int = Form(10, ge=4, le=30),
    seed: Optional[int] = Form(None, ge=0, le=2**31 - 1),
) -> Response:
    engine = _engine(request)
    has_reference = reference_audio is not None and (reference_audio.filename or reference_audio.size)
    if not has_reference:
        # No clip: behave exactly like /v1/tts, treating 'style' as the voice description.
        return await _synthesize_response(
            engine, text=text, voice_description=style, cfg_value=cfg_value, inference_timesteps=inference_timesteps, seed=seed
        )
    if not consent:
        raise HTTPException(
            422,
            "Set consent=true to confirm you have permission to clone this voice.",
        )

    ref_path, _ = await audio.save_reference_upload(
        reference_audio,
        max_bytes=settings.max_upload_bytes,
        min_seconds=settings.min_reference_seconds,
        max_seconds=settings.max_reference_seconds,
        tmp_dir=settings.tmp_dir,
    )
    try:
        return await _synthesize_response(
            engine,
            text=text,
            voice_description=style,
            reference_wav_path=ref_path,
            prompt_text=prompt_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            seed=seed,
        )
    finally:
        try:
            os.remove(ref_path)  # reference clips are never retained
        except OSError:
            pass


@app.post(
    "/v1/audio/speech",
    tags=["compat"],
    dependencies=protected,
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}}},
    summary="OpenAI-compatible speech endpoint",
)
async def openai_speech(body: OpenAISpeechRequest, request: Request) -> Response:
    if body.response_format.lower() != "wav":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only response_format='wav' is supported.")
    voice = body.voice.strip()
    description = None if voice.lower() in {"", "default", "alloy", "echo", "fable", "onyx", "nova", "shimmer"} else voice
    if description and len(description) > MAX_DESCRIPTION_CHARS:
        raise HTTPException(422, f"voice description longer than {MAX_DESCRIPTION_CHARS} chars.")
    return await _synthesize_response(_engine(request), text=body.input, voice_description=description)
