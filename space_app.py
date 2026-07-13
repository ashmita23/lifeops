"""Hugging Face Spaces entry point.

Named space_app.py (not app.py) deliberately - this project's backend
package is itself named `app` (app/agent.py, app/config.py, etc.), and
`import app` always resolves to that package over a same-named app.py
file at the repo root. A literal app.py here would be silently
unreachable via import, even though `python app.py` might appear to work
as a script - avoiding the name entirely sidesteps the ambiguity.
The Space's app_file setting in README.md's frontmatter points here.
"""

from frontend.gradio_app import main

if __name__ == "__main__":
    main()
