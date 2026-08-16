#!/bin/sh
set -e

# Runs as root so this can fix ownership on a freshly-attached platform
# volume (Railway/Render/etc. mount it root-owned regardless of what was
# chown'd at build time) before dropping to the non-root appuser for the
# actual app process.
mkdir -p /app/data
chown -R appuser:appuser /app/data

exec su appuser -s /bin/sh -c 'exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"'
