from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    search_endpoint: str
    search_api_key: str
    search_index_name: str = "chunks-index"
    search_semantic_config: str = "chunks-semantic-config"
    search_vector_field: str = "contentVector"

    openai_endpoint: str
    openai_api_key: str
    openai_api_version: str
    chat_deployment: str = "gpt-5-mini"
    embedding_deployment: str = "text-embedding-3-large"


@lru_cache
def get_settings() -> Settings:
    return Settings()