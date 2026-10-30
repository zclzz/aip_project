# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Model weights live on a volume so they are downloaded once, not baked into the image.
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    # Writable scratch locations (the root filesystem is read-only when run via compose).
    HOME=/tmp/home \
    XDG_CACHE_HOME=/tmp/cache \
    TORCHINDUCTOR_CACHE_DIR=/tmp/cache/inductor \
    TRITON_CACHE_DIR=/tmp/cache/triton \
    MPLCONFIGDIR=/tmp/cache/mpl \
    NUMBA_CACHE_DIR=/tmp/cache/numba \
    TMP_DIR=/tmp/tts

# ffmpeg: torchaudio/torchcodec decode reference clips through it. libsndfile: soundfile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged runtime user.
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home --home-dir /tmp/home app

WORKDIR /srv

# PyTorch CUDA 12.8 wheels: run on any NVIDIA driver >= 525, and fall back to CPU where there is no GPU.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.11.0 torchaudio==2.11.0 torchcodec==0.11.1
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

RUN mkdir -p /models && chown -R app:app /models /srv
USER app

EXPOSE 8000
# Long start period: the first boot downloads ~4.7 GB of weights.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header", "--no-access-log"]
