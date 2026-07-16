"""Deployment entry point (Railway / Hugging Face Spaces).

Named space_app.py (not app.py) deliberately - this project's backend
package is itself named `app` (app/agent.py, app/config.py, etc.), and
`import app` always resolves to that package over a same-named app.py
file at the repo root. A literal app.py here would be silently
unreachable via import, even though `python app.py` might appear to work
as a script - avoiding the name entirely sidesteps the ambiguity.

Serves the FastAPI app from app/web.py, which wraps the Gradio UI with
per-user "Sign in with Google" (falling back to single-user no-login when
the Web OAuth client isn't configured). $PORT is assigned by the host.
"""

import os

import uvicorn

from app.web import create_app

# Module-level so `uvicorn space_app:app` also works, not just `python space_app.py`.
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("SERVER_NAME", "0.0.0.0"),
        port=int(os.environ.get("PORT", "7860")),
    )
