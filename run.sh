#!/usr/bin/env bash
# Convenience launcher. Loads .env if present, then starts the API.
set -e
cd "$(dirname "$0")"
if [ -f .env ]; then
  set -a; source .env; set +a
fi
python3 serve_api.py --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
