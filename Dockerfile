FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN adduser --disabled-password --gecos "" appuser

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

RUN mkdir -p /app/data && chown -R appuser:appuser /app/data

USER appuser

EXPOSE 8000

# Shell form (not exec/JSON-array form) so $PORT actually gets substituted —
# platforms like Railway/Heroku/Render assign their own port at runtime and
# route traffic to it; a hardcoded --port 8000 works locally via
# docker-compose (which maps host:container explicitly) but leaves the app
# unreachable on a platform that expects it to bind to *their* port.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
