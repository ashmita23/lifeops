"""Prevents the test suite from sending real telemetry to whatever Langfuse
project happens to be configured in .env. This runs at collection time,
before any test module imports app.config (and therefore before
load_dotenv() runs) - load_dotenv() only fills in env vars that aren't
already set, so overwriting these here means the real .env values never
take effect during tests. Auth then fails harmlessly (same as when no keys
are configured at all), instead of writing fake zero-cost traces into a
real Langfuse project every test run.
"""

import os

os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-test-disabled"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-test-disabled"
