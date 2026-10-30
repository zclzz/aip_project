"""Upload validation and WAV encoding."""
from __future__ import annotations

import io
import os
import uuid

import numpy as np
import soundfile as sf
from fastapi import HTTPException, UploadFile, status

AI_LABEL = "AI-generated speech (VoxCPM2). Not a recording of a real person."


async def save_reference_upload(
    upload: UploadFile, *, max_bytes: int, min_seconds: float, max_seconds: float, tmp_dir: str
) -> tuple[str, float]:
    """Read an uploaded clip with a hard size cap, prove it decodes, and re-encode it as a clean mono WAV.

    The user-supplied filename and bytes never touch disk directly: only our own re-encoded file does.
    Returns (path, duration_seconds).
    """
    buf = io.BytesIO()
    size = 0
    while chunk := await upload.read(1024 * 1024):
        size += len(chunk)
        if size > max_bytes:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"reference_audio exceeds the {max_bytes // (1024 * 1024)} MB limit.",
            )
        buf.write(chunk)
    if size == 0:
        raise HTTPException(422, "reference_audio is empty.")

    buf.seek(0)
    try:
        data, sr = sf.read(buf, dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001 - any decode failure means "not audio we accept"
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "reference_audio must be a decodable WAV, FLAC, OGG or MP3 file.",
        ) from None

    duration = data.shape[0] / float(sr)
    if duration < min_seconds:
        raise HTTPException(422, f"reference_audio is shorter than {min_seconds:g}s.")
    if duration > max_seconds:
        raise HTTPException(422, f"reference_audio is longer than {max_seconds:g}s.")

    mono = data.mean(axis=1)
    path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.wav")
    sf.write(path, mono, sr, subtype="PCM_16")
    return path, duration


def encode_wav(wav: np.ndarray, sample_rate: int) -> bytes:
    """16-bit PCM WAV with an embedded comment marking the audio as AI-generated."""
    buf = io.BytesIO()
    with sf.SoundFile(buf, mode="w", samplerate=sample_rate, channels=1, format="WAV", subtype="PCM_16") as f:
        f.comment = AI_LABEL
        f.software = "voxcpm2-tts-service"
        f.write(np.clip(wav, -1.0, 1.0))
    return buf.getvalue()
