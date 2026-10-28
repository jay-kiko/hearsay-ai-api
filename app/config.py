from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # Server-held key — no accounts, so a single Anthropic key funds every
    # request; access is gated by multi-use codes instead (see code_store.py).
    anthropic_api_key: str
    # Three model tiers, chosen per step by how much the output shapes what a
    # client sees (see CLAUDE.md "Model tiers"):
    #   quality  — generative steps whose wording/content *is* the result
    #              (personas, prompts, categories, insights). Run rarely per
    #              job, so the cost is worth it.
    #   standard — the buyer-persona "answer" itself (pipeline.py, the actual
    #              signal being measured, fans out per prompt) plus judgement
    #              calls that decide what the report shows (detect research,
    #              source classification, brand detection).
    #   fast     — mechanical work only (sentiment classification, relevance
    #              filtering, running the source search's given queries).
    anthropic_quality_model: str = "claude-opus-5-5"
    anthropic_model: str = "claude-sonnet-5"
    anthropic_fast_model: str = "claude-haiku-4-5-20251001"
    # Retries for a failed call (rate limits have their own, longer budget in
    # anthropic_client). The SDK's own retries are off, so this is the total.
    anthropic_max_retries: int = 2
    # Per call. Web-search calls are the slowest, measured at up to ~100s.
    anthropic_timeout_seconds: float = 180.0
    # Calls in flight at once across every job in the process.
    anthropic_max_concurrency: int = 16

    admin_secret: str | None = None
    codes_db_path: str = "/app/data/codes.db"
    # /api/prompts validates a code but doesn't spend a use (so users can
    # regenerate while reviewing) — cap the free regenerations per code so
    # that path can't rack up unlimited billed calls on its own.
    max_prompt_calls_per_code: int = 10
    # Failed access-code / admin-secret lookups allowed per window before
    # further lookups are refused (see app/rate_limit.py).
    code_failures_per_ip: int = 20
    code_failures_global: int = 1000
    code_failure_window_seconds: int = 600

    prompts_per_persona: int = 3
    persona_count: int = 4
    persona_concurrency: int = 4
    job_ttl_seconds: int = 1800

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
