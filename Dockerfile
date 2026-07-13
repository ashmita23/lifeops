FROM python:3.11-slim

# Node.js/npx is required to spawn the Google Calendar MCP server
# (@cocal/google-calendar-mcp, see app/mcp_client.py). Without it, calendar
# integration just falls back to the local mock calendar automatically -
# but installing it here means it actually works when configured.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway assigns a dynamic port via $PORT; space_app.py/gradio_app.py read
# it (defaulting to 7860 for local runs where PORT isn't set).
ENV PORT=7860
EXPOSE 7860

# The Gradio chat UI is the primary, actually-used interface (see README) -
# this is what gets deployed. The FastAPI backend (app/main.py) stays
# available for local `uvicorn app.main:app` use but isn't run here.
CMD ["python", "space_app.py"]
