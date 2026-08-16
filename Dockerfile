FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN adduser --disabled-password --gecos "" appuser

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh && mkdir -p /app/data && chown -R appuser:appuser /app/data

EXPOSE 8000

# Stays root at container start (not USER appuser) — the entrypoint needs
# root to fix ownership on a freshly-attached platform volume before it
# drops to appuser itself. $PORT substitution (platforms like
# Railway/Heroku/Render assign their own port at runtime) happens inside the
# entrypoint script too, for the same reason a hardcoded exec-form CMD can't
# do shell substitution.
ENTRYPOINT ["/docker-entrypoint.sh"]
