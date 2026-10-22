"""config.py - central settings for VoiceFlow Agent.

Secrets (API keys, webhook URL) are NEVER hard-coded. They are read from:
  1. environment variables / a local .env file (local development), or
  2. Streamlit secrets (Streamlit Community Cloud).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Keep CrewAI quiet and private: no telemetry.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"


def get_secret(name, default=None):
    """Return a secret from the environment first, then from Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        # No secrets file / not running under Streamlit: fall through.
        pass
    return default


def _int(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# ----- Secrets (read at call time so Streamlit secrets are available) -----
def groq_api_key():
    return get_secret("GROQ_API_KEY")


def hf_token():
    return get_secret("HF_TOKEN")


def n8n_webhook_url():
    return get_secret("N8N_WEBHOOK_URL")


def default_recipient_email():
    return get_secret("DEFAULT_RECIPIENT_EMAIL", "")


# ----- Models -----
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# "hf_api" (default, lightweight) or "local" (needs: pip install sentence-transformers)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "hf_api")
HF_EMBED_URL = os.getenv(
    "HF_EMBED_URL",
    "https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction",
)
EMBED_BATCH = _int("EMBED_BATCH", 16)

# ----- RAG -----
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db")))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "voiceflow_kb")
# Sizes are in tokens (approximated as 1 token ~ 4 characters).
# NOTE: all-MiniLM-L6-v2 only reads ~256 tokens per chunk, so for best retrieval
# quality you may lower CHUNK_TOKENS to ~200 (see README, "Tuning").
CHUNK_TOKENS = _int("CHUNK_TOKENS", 800)
CHUNK_OVERLAP_TOKENS = _int("CHUNK_OVERLAP_TOKENS", 100)
TOP_K = _int("TOP_K", 4)

# ----- Agents -----
DEFAULT_PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v2")
MAX_REVISIONS = _int("MAX_REVISIONS", 2)  # Writer <-> Reviewer loops
LLM_RETRIES = _int("LLM_RETRIES", 3)  # retries on rate limits / transient errors
WEB_MAX_RESULTS = _int("WEB_MAX_RESULTS", 5)


def available_prompt_versions():
    """Folders like prompts/v1, prompts/v2."""
    if not PROMPTS_DIR.exists():
        return []
    return sorted(p.name for p in PROMPTS_DIR.iterdir() if p.is_dir() and p.name.startswith("v"))
