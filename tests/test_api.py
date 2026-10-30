import io

import numpy as np
import soundfile as sf

AUTH = {"X-API-Key": "test-key"}


def wav_bytes(seconds=2.0, sr=16000):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    buf = io.BytesIO()
    sf.write(buf, 0.2 * np.sin(2 * np.pi * 220 * t), sr, format="WAV")
    return buf.getvalue()


def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_ready(client):
    assert client.get("/ready").status_code == 200


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "VoxCPM2" in r.text


def test_missing_key_rejected(client):
    r = client.post("/v1/tts", json={"text": "hi"})
    assert r.status_code == 401


def test_wrong_key_rejected(client):
    r = client.post("/v1/tts", json={"text": "hi"}, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_bearer_key_accepted(client):
    r = client.post("/v1/tts", json={"text": "hi"}, headers={"Authorization": "Bearer test-key"})
    assert r.status_code == 200


def test_tts_returns_labelled_wav(client, engine):
    r = client.post("/v1/tts", json={"text": "Hello there", "voice_description": "calm narrator", "seed": 7}, headers=AUTH)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.headers["x-ai-generated"] == "true"
    assert r.content[:4] == b"RIFF"
    data, sr = sf.read(io.BytesIO(r.content))
    assert sr == 48000 and len(data) == 24000
    assert b"AI-generated" in r.content  # comment chunk embedded in the file
    assert engine.calls[-1]["voice_description"] == "calm narrator"
    assert engine.calls[-1]["seed"] == 7


def test_text_too_long(client):
    r = client.post("/v1/tts", json={"text": "x" * 201}, headers=AUTH)
    assert r.status_code == 422


def test_bad_params_rejected(client):
    r = client.post("/v1/tts", json={"text": "hi", "inference_timesteps": 999}, headers=AUTH)
    assert r.status_code == 422


def test_clone_without_reference_uses_default_voice(client, engine):
    r = client.post("/v1/tts/clone", data={"text": "hi", "style": "cheerful"}, headers=AUTH)
    assert r.status_code == 200
    assert "reference_wav_path" not in engine.calls[-1]
    assert engine.calls[-1]["voice_description"] == "cheerful"


def test_clone_requires_consent(client):
    r = client.post(
        "/v1/tts/clone",
        data={"text": "hi"},
        files={"reference_audio": ("ref.wav", wav_bytes(), "audio/wav")},
        headers=AUTH,
    )
    assert r.status_code == 422
    assert "consent" in r.json()["detail"]


def test_clone_rejects_non_audio(client):
    r = client.post(
        "/v1/tts/clone",
        data={"text": "hi", "consent": "true"},
        files={"reference_audio": ("evil.wav", b"<html>not audio</html>", "audio/wav")},
        headers=AUTH,
    )
    assert r.status_code == 415


def test_clone_rejects_too_short_clip(client):
    r = client.post(
        "/v1/tts/clone",
        data={"text": "hi", "consent": "true"},
        files={"reference_audio": ("ref.wav", wav_bytes(seconds=0.3), "audio/wav")},
        headers=AUTH,
    )
    assert r.status_code == 422


def test_clone_with_reference_and_transcript(client, engine):
    r = client.post(
        "/v1/tts/clone",
        data={"text": "hi", "consent": "true", "prompt_text": "what the clip says"},
        files={"reference_audio": ("ref.wav", wav_bytes(), "audio/wav")},
        headers=AUTH,
    )
    assert r.status_code == 200
    call = engine.calls[-1]
    assert call["reference_wav_path"].endswith(".wav")
    assert call["prompt_text"] == "what the clip says"
    import os

    assert not os.path.exists(call["reference_wav_path"])  # cleaned up after the request


def test_openai_compat(client, engine):
    r = client.post("/v1/audio/speech", json={"model": "tts-1", "input": "hi", "voice": "a soft whisper"}, headers=AUTH)
    assert r.status_code == 200 and r.content[:4] == b"RIFF"
    assert engine.calls[-1]["voice_description"] == "a soft whisper"
    r = client.post("/v1/audio/speech", json={"input": "hi", "response_format": "mp3"}, headers=AUTH)
    assert r.status_code == 400


def test_rate_limit(client):
    codes = [client.post("/v1/tts", json={"text": "hi"}, headers=AUTH).status_code for _ in range(6)]
    assert codes[:5] == [200] * 5 and codes[5] == 429


def test_not_ready_returns_503(client, engine):
    engine.ready = False
    r = client.post("/v1/tts", json={"text": "hi"}, headers=AUTH)
    assert r.status_code == 503
