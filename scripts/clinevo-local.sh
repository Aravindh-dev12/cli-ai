#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cmd="${1:-up}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "clinevo: missing required command: $1" >&2
    exit 1
  }
}

wait_for() {
  local name="$1"
  local url="$2"
  local attempts="${3:-120}"
  for _ in $(seq 1 "$attempts"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "ready: $name"
      return 0
    fi
    sleep 2
  done
  echo "clinevo: timed out waiting for $name at $url" >&2
  docker compose ps || true
  docker compose logs --tail=120 ai-service backend || true
  return 1
}

case "$cmd" in
  up)
    require_cmd docker
    require_cmd curl
    if [[ ! -f .env ]]; then
      cp .env.example .env
      echo "created .env from .env.example"
    fi

    docker compose up --build -d

    wait_for "AI service" "http://localhost:8000/health" 120
    wait_for "backend" "http://localhost:8080/actuator/health" 180

    echo
    echo "Clinevo local runtime is up:"
    echo "  UI:              http://localhost:4200"
    echo "  API:             http://localhost:8080"
    echo "  Agent status:    http://localhost:8080/api/agent/status"
    echo "  AI health:       http://localhost:8000/health"
    echo "  Decision health: http://localhost:8000/decision/health"
    echo
    echo "Warming the configured decision backend (first run may download Laya weights):"
    curl -fsS -X POST http://localhost:8000/decision       -H 'Content-Type: application/json'       -d '{"state":{"source_type":"LOCAL_WARMUP","body":"Synthetic warmup only. Classify this local runtime health check."}}'       | python -m json.tool
    ;;
  down)
    require_cmd docker
    docker compose down
    ;;
  doctor)
    require_cmd curl
    wait_for "AI service" "http://localhost:8000/health" 3
    wait_for "backend" "http://localhost:8080/actuator/health" 3
    curl -fsS http://localhost:8000/decision/health | python -m json.tool
    curl -fsS http://localhost:8080/api/agent/status | python -m json.tool
    ;;
  logs)
    require_cmd docker
    docker compose logs -f --tail=200
    ;;
  status)
    require_cmd docker
    docker compose ps
    ;;
  *)
    echo "usage: bash scripts/clinevo-local.sh {up|down|doctor|logs|status}" >&2
    exit 2
    ;;
esac
