# VoxCPM2 TTS Service

A dockerised [FastAPI](https://fastapi.tiangolo.com/) service that serves [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)

It does three things through one small API:

| Mode | What you send | What you get |
|---|---|---|
| **Generate** | text | speech in the model's default voice |
| **Design** | text + a short description ("a deep, calm male narrator") | speech in a voice matching the description |
| **Clone** | text + a 1–30 s reference clip (+ optional transcript) | speech in the reference speaker's voice |


## Quick start

```bash
git clone <this repo> && cd <this repo>
cp .env.example .env            # optional: set API_KEY; otherwise one is generated and printed in the logs

# GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
# CPU only (slow, but works anywhere)
docker compose up --build -d

docker compose logs -f tts 
```

When the log says `Model ready on cuda`, open:

- **http://localhost:8010/** - web page: type text, pick a mode, press play
- **http://localhost:8010/docs** - interactive Swagger UI

First start takes a few minutes (weight download). Later starts take about 30 s.

## Credits

Model: [OpenBMB VoxCPM2](https://github.com/OpenBMB/VoxCPM) (Apache-2.0). 