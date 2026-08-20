"""
Standalone Speech-to-Text (STT) pipeline backed by the Groq Whisper API.

Phase 1 scope: microphone audio -> transcription text ONLY.

This module is kept intentionally independent from the RAG / chat pipeline so
that it can be connected later without rewriting it. The single public entry
point is:

    transcribe_audio(audio_file) -> str

Configuration follows the existing project convention:
    - `.env` file in the project root (loaded via python-dotenv)
    - `GROQ_API_KEY` environment variable (no hardcoded keys)
    - identical to how app.py / agents/chat_agent.py read the key

This module contains NO Streamlit imports on purpose — it is UI-agnostic and
usable from any caller (a Streamlit app, a script, or the future chat pipeline).
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from groq import Groq

# Load .env from the project root, same as app.py and agents/chat_agent.py.
load_dotenv()

logger = logging.getLogger(__name__)

# ── Model Configuration ──────────────────────────────────────────────────────

STT_MODEL = "whisper-large-v3-turbo"

# Audio types the Groq /audio/transcriptions endpoint accepts.
_ACCEPTED_MIME_TYPES = frozenset({
    "audio/wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/flac",
    "audio/ogg",
    "audio/mp4",
    "audio/webm",
})


# ── Client Factory ───────────────────────────────────────────────────────────

_client: Groq | None = None


def _get_groq_client() -> Groq:
    """
    Lazily create and cache the Groq client.

    Reuses the project's existing GROQ_API_KEY environment variable (loaded
    from `.env`). Raises a friendly RuntimeError when the key is missing or the
    client cannot be initialized.
    """
    global _client

    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Add it to the '.env' file in the project root and restart the app."
            )
        try:
            _client = Groq(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Could not initialize the Groq client: {e}") from e

    return _client


# ── Audio Extraction ─────────────────────────────────────────────────────────

def _extract_audio_bytes(audio_file) -> tuple[bytes, str, str]:
    """
    Return ``(audio bytes, filename hint, mime type)`` from an audio input.

    Supports Streamlit's ``st.audio_input`` UploadedFile (via ``getvalue()``)
    as well as any file-like object that supports ``read()``.

    Raises ValueError for missing/empty/unreadable audio.
    """
    if audio_file is None:
        raise ValueError("No audio recorded. Please record or upload audio first.")

    name = str(getattr(audio_file, "name", None) or "recording.wav")
    mime = str(getattr(audio_file, "type", None) or "audio/wav")

    if hasattr(audio_file, "getvalue"):
        raw = audio_file.getvalue()
    elif hasattr(audio_file, "read"):
        try:
            raw = audio_file.read()
        except Exception as e:
            raise ValueError(f"Could not read the audio file: {e}") from e
    else:
        raise ValueError(
            "Unsupported audio input: expected an uploaded/recorded audio file "
            "(e.g. the object returned by st.audio_input)."
        )

    if not raw:
        raise ValueError("The audio is empty. Please record something and try again.")

    # Normalize unknown-but-audio-like mime types; Groq can still infer the format.
    if mime not in _ACCEPTED_MIME_TYPES:
        logger.warning("Unrecognized audio mime type %r — falling back to audio/wav.", mime)
        mime = "audio/wav"

    return bytes(raw), name, mime


# ── Transcription ────────────────────────────────────────────────────────────

def transcribe_audio(audio_file) -> str:
    """
    Convert recorded microphone audio into text via Groq Whisper.

    Accepts the audio object returned by ``st.audio_input()`` (an UploadedFile)
    or any file-like object exposing bytes. Returns only the transcription text.

    Parameters
    ----------
    audio_file:
        The recorded/uploaded audio (e.g. from Streamlit's audio_input widget).

    Returns
    -------
    str
        The transcription text only.

    Raises
    ------
    ValueError
        If no audio, empty audio, or an unreadable/unrecognized input is given.
    RuntimeError
        If the Groq API call fails (missing key, network error, or API error).
    """
    data, name, mime = _extract_audio_bytes(audio_file)

    client = _get_groq_client()

    logger.info(
        "Transcribing %r (%d bytes, %s) with model %s",
        name, len(data), mime, STT_MODEL,
    )

    try:
        transcription = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=(name, data, mime),
            response_format="json",
        )
        text = str(transcription.text).strip()
    except RuntimeError:
        raise
    except Exception as e:
        logger.exception("Groq Whisper transcription failed for %r", name)
        raise RuntimeError(f"Transcription failed: {e}") from e

    if not text:
        raise RuntimeError("Groq returned an empty transcription. Please record again and retry.")

    return text