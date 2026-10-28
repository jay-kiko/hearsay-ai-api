import os

# Settings requires a key at import time; tests never call the real API.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
