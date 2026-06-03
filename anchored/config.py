"""Centralized configuration, loaded from environment / .env.

All knobs that affect a baseline (chunk size, model, k) live here so they can be
recorded alongside metrics in BASELINE.md.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Elasticsearch ---
    es_url: str = "http://elasticsearch:9200"
    es_index: str = "anchored_cuad"

    # --- Embeddings ---
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # --- Generation (optional; retrieval baseline needs no LLM) ---
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None

    # --- Retrieval / chunking ---
    top_k: int = 10
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- Paths ---
    data_dir: str = "data"
    traces_dir: str = "traces"


settings = Settings()
