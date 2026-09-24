from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # LLM Config
    primary_model: str = "llama3:8b"
    fallback_model: str = "llama3.2:3b"

    # Langsmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "prodcution_api"

    # Applicaton
    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
