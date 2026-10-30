import os

# Configure before the app module is imported.
os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "5")
os.environ.setdefault("MAX_TEXT_CHARS", "200")
os.environ.setdefault("TMP_DIR", os.path.join(os.path.dirname(__file__), ".tmp"))

import numpy as np
import pytest
from fastapi.testclient import TestClient


class FakeEngine:
    """Stands in for the real model so tests run in milliseconds without a GPU."""

    model_id = "fake/model"
    revision = "deadbeef"
    device = "cpu"
    sample_rate = 48000
    load_error = None
    load_seconds = 0.0
    ready = True

    def __init__(self):
        self.calls = []

    def synthesize(self, text, **kwargs):
        from app.engine import EngineNotReady

        if not self.ready:
            raise EngineNotReady("still loading")
        self.calls.append({"text": text, **kwargs})
        t = np.linspace(0, 0.5, int(self.sample_rate * 0.5), endpoint=False)
        return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture()
def engine():
    return FakeEngine()


@pytest.fixture()
def client(engine):
    from app.main import app
    from app.security import rate_limiter

    rate_limiter._hits.clear()
    app.state.engine = engine
    with TestClient(app) as c:
        yield c
