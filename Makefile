# Convenience targets. Plain `docker compose` commands are documented in the README.
COMPOSE := docker compose -f docker-compose.yml
HAS_GPU := $(shell command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && echo yes)
ifeq ($(HAS_GPU),yes)
  COMPOSE += -f docker-compose.gpu.yml
endif

.PHONY: help up up-cpu down logs build test smoke

help:
	@echo "make up       build + start (GPU if nvidia-smi works, else CPU)"
	@echo "make up-cpu   build + start on CPU only"
	@echo "make down     stop and remove the container"
	@echo "make logs     follow the service logs (shows the generated API key)"
	@echo "make test     run the unit tests locally (no model needed)"
	@echo "make smoke    hit the running service with curl (needs API_KEY env var)"

up:
	$(COMPOSE) up --build -d
	@echo "Service starting on http://localhost:$${PORT:-8010}  (follow with: make logs)"

up-cpu:
	docker compose -f docker-compose.yml up --build -d

down:
	docker compose -f docker-compose.yml down

logs:
	docker compose -f docker-compose.yml logs -f tts

build:
	docker compose -f docker-compose.yml build

test:
	python -m pytest -q

smoke:
	@test -n "$$API_KEY" || (echo "set API_KEY first, e.g. export API_KEY=..." && exit 1)
	curl -sf http://localhost:$${PORT:-8010}/health | python -m json.tool
	curl -sf -X POST http://localhost:$${PORT:-8010}/v1/tts -H "X-API-Key: $$API_KEY" -H "Content-Type: application/json" \
	  -d '{"text":"The service is up and speaking.","voice_description":"A clear, friendly voice"}' -o smoke.wav \
	  && echo "wrote smoke.wav ($$(stat -c %s smoke.wav) bytes)"
