from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # Server-held key — no accounts, so a single Anthropic key funds every
    # request; access is gated by multi-use codes instead (see code_store.py).
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-5"
    anthropic_max_retries: int = 1

    admin_secret: str | None = None
    codes_db_path: str = "/app/data/codes.db"
    # /api/prompts validates a code but doesn't spend a use (so users can
    # regenerate while reviewing) — cap the free regenerations per code so
    # that path can't rack up unlimited billed calls on its own.
    max_prompt_calls_per_code: int = 10

    prompts_per_persona: int = 3
    persona_count: int = 8
    persona_concurrency: int = 4
    job_ttl_seconds: int = 1800

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
