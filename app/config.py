"""Centralized runtime configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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
