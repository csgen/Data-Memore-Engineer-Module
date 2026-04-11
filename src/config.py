"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Neo4j Aura
    neo4j_uri: str
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # ChromaDB Cloud
    chroma_host: str = "api.trychroma.com"
    chroma_port: int = 443
    chroma_api_key: str = ""
    chroma_tenant: str = ""
    chroma_database: str = ""

    # OpenAI
    openai_api_key: str

    # Tavily Search API
    tavily_api_key: str = ""

    # Telegram scraper API (hosted on Google Cloud)
    telegram_scraper_api_url: str = ""
    telegram_scraper_api_key: str = ""

    # Model settings
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o"

settings = Settings()
