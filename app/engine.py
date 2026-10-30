"""Thin wrapper around the VoxCPM2 model: device selection, loading, and serialized generation."""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

log = logging.getLogger("tts.engine")


class EngineNotReady(Exception):
    """Model has not finished loading (or failed to load)."""


class EngineBusy(Exception):
    """Could not get the GPU within the queue timeout."""


def pick_device(preference: str) -> str:
    """Resolve the DEVICE setting: GPU first, CPU if there is none."""
    import torch

    pref = (preference or "auto").strip().lower()
    if pref in {"", "auto"}:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if pref.startswith("cuda") and not torch.cuda.is_available():
        log.warning("DEVICE=%s requested but CUDA is not available; falling back to CPU", pref)
        return "cpu"
    return pref


class TTSEngine:
    def __init__(
        self,
        model_id: str,
        revision: str,
        device_preference: str = "auto",
        load_denoiser: bool = False,
        queue_timeout: float = 120.0,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device_preference = device_preference
        self.load_denoiser = load_denoiser
        self.queue_timeout = queue_timeout

        self.model = None
        self.device: Optional[str] = None
        self.sample_rate: Optional[int] = None
        self.model_path: Optional[str] = None
        self.load_error: Optional[str] = None
        self.load_seconds: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Download (if needed) and load the model. Safe to run in a background thread."""
        t0 = time.time()
        try:
            from huggingface_hub import snapshot_download
            from voxcpm import VoxCPM

            log.info("Fetching %s@%s", self.model_id, self.revision)
            self.model_path = snapshot_download(self.model_id, revision=self.revision)
            self.device = pick_device(self.device_preference)
            log.info("Loading model on %s", self.device)
            model = VoxCPM(
                voxcpm_model_path=self.model_path,
                zipenhancer_model_path=None if not self.load_denoiser else "iic/speech_zipenhancer_ans_multiloss_16k_base",
                enable_denoiser=self.load_denoiser,
                # torch.compile needs a C compiler and a ~60 s warm-up at every start; measured no faster here.
                optimize=False,
                device=self.device,
            )
            self.sample_rate = int(model.tts_model.sample_rate)
            self.model = model
            self.load_seconds = round(time.time() - t0, 1)
            log.info("Model ready on %s in %ss (sample rate %s Hz)", self.device, self.load_seconds, self.sample_rate)
        except Exception as exc:  # noqa: BLE001 - we want to surface any load failure via /health
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.exception("Model failed to load")

    def synthesize(
        self,
        text: str,
        *,
        voice_description: Optional[str] = None,
        reference_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Return a mono float32 waveform at ``self.sample_rate``. Blocking; call from a worker thread."""
        if not self.ready:
            raise EngineNotReady(self.load_error or "model is still loading")

        full_text = text.strip()
        if voice_description and voice_description.strip():
            # VoxCPM2 reads a leading parenthetical as a voice description (or style instruction when cloning).
            full_text = f"({voice_description.strip()}) {full_text}"

        kwargs = dict(text=full_text, cfg_value=cfg_value, inference_timesteps=inference_timesteps)
        if reference_wav_path:
            kwargs["reference_wav_path"] = reference_wav_path
            if prompt_text and prompt_text.strip():
                # "Ultimate" cloning: reference clip + its transcript for highest similarity.
                kwargs["prompt_wav_path"] = reference_wav_path
                kwargs["prompt_text"] = prompt_text.strip()

        if not self._lock.acquire(timeout=self.queue_timeout):
            raise EngineBusy("generation queue timed out")
        try:
            if seed is not None:
                import torch

                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            wav = self.model.generate(**kwargs)
        finally:
            self._lock.release()
        return np.asarray(wav, dtype=np.float32).reshape(-1)
