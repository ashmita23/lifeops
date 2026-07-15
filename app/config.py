"""Centralized runtime configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # --- LLM gateway / model routing (see app/llm_gateway.py) ---
    # The cheap "local" tier is a quantized open model served by Ollama; the
    # cloud tier is the strong model. Routing sends simple/synthesis calls to
    # local and real tool-calling to cloud, with local->cloud fallback.
    cloud_model: str = os.getenv("CLOUD_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    local_model: str = os.getenv("LOCAL_MODEL", "ollama/llama3.1:8b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    local_timeout_seconds: float = float(os.getenv("LOCAL_TIMEOUT_SECONDS", "20"))
    # Feature flag: when False, every call uses the cloud model (routing off).
    # This is the minimal canary/rollback lever - flip without a redeploy.
    model_routing_enabled: bool = os.getenv("MODEL_ROUTING_ENABLED", "false").lower() == "true"

    # Response cache + per-session guardrails (see app/budget.py).
    llm_cache_enabled: bool = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"
    session_cost_cap_usd: float = float(os.getenv("SESSION_COST_CAP_USD", "1.00"))
    session_rate_limit_per_min: int = int(os.getenv("SESSION_RATE_LIMIT_PER_MIN", "60"))

    # --- RAG over journal entries (see app/journal_index.py, app/embeddings.py) ---
    # Retrieval uses Pinecone (managed serverless vector DB). All RAG features
    # no-op gracefully when PINECONE_API_KEY is unset (see rag_enabled), so the
    # app and tests run fine without it.
    pinecone_api_key: str | None = os.getenv("PINECONE_API_KEY") or None
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "lifeops-journal")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    database_path: str = os.getenv("DATABASE_PATH", "lifeops.db")
    default_timezone: str = os.getenv("DEFAULT_TIMEZONE", "America/Chicago")
    google_oauth_credentials_path: str | None = os.getenv("GOOGLE_OAUTH_CREDENTIALS_PATH") or None
    # Optional override for where the Calendar MCP server reads/writes its
    # cached OAuth token - see app/mcp_client.py for why this matters on
    # ephemeral hosts.
    google_calendar_mcp_token_path: str | None = os.getenv("GOOGLE_CALENDAR_MCP_TOKEN_PATH") or None

    # Shared-password gate for the publicly reachable Gradio UI (see
    # frontend/gradio_app.py). Both must be set for auth to actually apply -
    # deliberately no silent bypass if only one is configured.
    gradio_auth_user: str | None = os.getenv("GRADIO_AUTH_USER") or None
    gradio_auth_pass: str | None = os.getenv("GRADIO_AUTH_PASS") or None

    @property
    def gradio_auth(self) -> tuple[str, str] | None:
        if self.gradio_auth_user and self.gradio_auth_pass:
            return (self.gradio_auth_user, self.gradio_auth_pass)
        return None

    @property
    def rag_enabled(self) -> bool:
        """RAG (journal retrieval) is available only when Pinecone is
        configured and a real embedding model can be called. Everything RAG
        checks this and degrades to a no-op otherwise."""
        return bool(self.pinecone_api_key) and not self.demo_mode

    # langfuse's SDK auto-reads LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY/LANGFUSE_HOST
    # from the environment itself; these are only kept here so we can log
    # whether tracing is actually configured, and so app.trace_export can
    # call the Langfuse REST API directly with the same credentials.
    langfuse_public_key: str | None = os.getenv("LANGFUSE_PUBLIC_KEY") or None
    langfuse_secret_key: str | None = os.getenv("LANGFUSE_SECRET_KEY") or None
    langfuse_host: str | None = os.getenv("LANGFUSE_HOST") or None

    @property
    def demo_mode(self) -> bool:
        """True when no LLM API key is configured, so we fall back to the
        deterministic local parser instead of calling a real model."""
        return not self.openai_api_key

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


settings = Settings()
