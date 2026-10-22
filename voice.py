"""voice.py - speech-to-text with Groq Whisper, plus yes/no understanding.

The browser records audio (Streamlit's st.audio_input); we send the bytes to Groq's
hosted whisper-large-v3 model and get text back.
"""
from __future__ import annotations

import re

import config


class VoiceError(Exception):
    """Raised when transcription fails (missing key, network, API error...)."""


def transcribe(audio_bytes, filename="speech.wav", language=None):
    """Convert recorded audio (bytes) to text. `language` is an optional ISO code like 'en' or 'ur'."""
    api_key = config.groq_api_key()
    if not api_key:
        raise VoiceError("GROQ_API_KEY is not set. Add it to .env (or Streamlit secrets).")
    if not audio_bytes:
        raise VoiceError("No audio received. Please record again.")
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        kwargs = {"file": (filename, audio_bytes), "model": config.WHISPER_MODEL, "temperature": 0.0}
        if language:
            kwargs["language"] = language
        response = client.audio.transcriptions.create(**kwargs)
        text = getattr(response, "text", None) or str(response)
        return text.strip()
    except VoiceError:
        raise
    except Exception as exc:  # network, rate limit, bad audio, ...
        raise VoiceError(f"Speech-to-text failed: {exc}") from exc


# ---- understanding a spoken "yes / no" (used for the send-to-Google approval) ----
_NO = re.compile(
    r"\b(no|nope|nah|don'?t|do not|cancel|stop|never ?mind|not now|negative|abort)\b", re.I
)
_YES = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok(ay)?|please do|go ahead|send( it| this| that)?|do it|confirm(ed)?|approved?)\b",
    re.I,
)


def interpret_confirmation(text):
    """Return 'yes', 'no' or 'unclear'.

    Deliberately conservative: any negative word wins, so "no, don't send it" is 'no'
    and we never send by accident.
    """
    if not text or not text.strip():
        return "unclear"
    if _NO.search(text):
        return "no"
    if _YES.search(text):
        return "yes"
    return "unclear"
