"""
backend.py
----------
Core AI + data-processing logic for "The Silent Co-Driver".

Responsibilities:
1. Speech-to-text transcription of radio audio (Whisper, via HF `transformers` pipeline)
2. Speech emotion recognition from raw audio (Wav2Vec2, via HF `transformers` pipeline)
3. Lap-time CSV parsing
4. Aligning a detected "mood moment" with the lap(s) it overlaps, so the dashboard
   can show whether stress correlates with slower lap times.

All heavy models are loaded once and cached (Streamlit's @st.cache_resource is used
in app.py; this module stays framework-agnostic so it can be unit tested on its own).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Model identifiers (Hugging Face Hub)
# ---------------------------------------------------------------------------
ASR_MODEL_ID = "openai/whisper-tiny"          # fast, good enough for radio chatter
SER_MODEL_ID = "Dpngtm/wav2vec2-emotion-recognition"  # acoustic emotion classifier

# Map raw model labels -> the 3 labels our dashboard cares about, with an emoji/color
EMOTION_DISPLAY_MAP = {
    "angry":    {"label": "Stressed", "emoji": "🔴", "color": "#e74c3c"},
    "fear":     {"label": "Stressed", "emoji": "🔴", "color": "#e74c3c"},
    "disgust":  {"label": "Stressed", "emoji": "🔴", "color": "#e74c3c"},
    "sad":      {"label": "Tired",    "emoji": "🟡", "color": "#f39c12"},
    "neutral":  {"label": "Calm",     "emoji": "🟢", "color": "#2ecc71"},
    "calm":     {"label": "Calm",     "emoji": "🟢", "color": "#2ecc71"},
    "happy":    {"label": "Calm",     "emoji": "🟢", "color": "#2ecc71"},
    "surprise": {"label": "Stressed", "emoji": "🔴", "color": "#e74c3c"},
}

DEFAULT_DISPLAY = {"label": "Unknown", "emoji": "⚪", "color": "#95a5a6"}


@dataclass
class AnalysisResult:
    transcript: str
    raw_emotion: str
    display_label: str
    emoji: str
    color: str
    confidence: float


# ---------------------------------------------------------------------------
# Model loading (kept as plain functions so app.py can wrap them with
# st.cache_resource — avoids reloading multi-hundred-MB models on every rerun)
# ---------------------------------------------------------------------------
def load_asr_pipeline():
    """Loads the Whisper speech-to-text pipeline from the HF Hub."""
    from transformers import pipeline  # lazy import: keeps CSV/unit-test paths light
    return pipeline("automatic-speech-recognition", model=ASR_MODEL_ID)


def load_ser_pipeline():
    """Loads the Wav2Vec2 speech-emotion-recognition pipeline from the HF Hub."""
    from transformers import pipeline  # lazy import: keeps CSV/unit-test paths light
    return pipeline("audio-classification", model=SER_MODEL_ID)


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------
def transcribe_audio(asr_pipe, audio_path: str) -> str:
    """Runs Whisper on the given audio file path and returns plain text."""
    result = asr_pipe(audio_path)
    return result["text"].strip()


def detect_emotion(ser_pipe, audio_path: str) -> tuple[str, float]:
    """
    Runs the Wav2Vec2 SER model and returns (top_label, confidence_score).
    """
    predictions = ser_pipe(audio_path)
    # pipeline returns a list of {"label": ..., "score": ...} sorted by score desc
    top = predictions[0]
    return top["label"].lower(), float(top["score"])


def analyze_clip(asr_pipe, ser_pipe, audio_path: str) -> AnalysisResult:
    """Runs both models on one audio file and packages a UI-ready result."""
    transcript = transcribe_audio(asr_pipe, audio_path)
    raw_label, score = detect_emotion(ser_pipe, audio_path)
    display = EMOTION_DISPLAY_MAP.get(raw_label, DEFAULT_DISPLAY)

    return AnalysisResult(
        transcript=transcript,
        raw_emotion=raw_label,
        display_label=display["label"],
        emoji=display["emoji"],
        color=display["color"],
        confidence=round(score * 100, 1),
    )


# ---------------------------------------------------------------------------
# Lap data handling
# ---------------------------------------------------------------------------
REQUIRED_LAP_COLUMNS = {"lap", "lap_time_sec", "timestamp_sec"}


def load_lap_data(csv_file) -> pd.DataFrame:
    """
    Parses a lap-time CSV. Expected columns:
        lap            -> lap number (int)
        lap_time_sec   -> lap time in seconds (float)
        timestamp_sec  -> seconds into the session when that lap finished (float)
                           (used to align with the audio clip's timing)

    Accepts a file path or a file-like object (Streamlit's UploadedFile works for both).
    """
    df = pd.read_csv(csv_file)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_LAP_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {sorted(missing)}. "
            f"Expected columns: {sorted(REQUIRED_LAP_COLUMNS)}"
        )

    df = df.sort_values("lap").reset_index(drop=True)
    return df


def nearest_lap_for_timestamp(lap_df: pd.DataFrame, radio_timestamp_sec: float) -> Optional[int]:
    """
    Given the moment (in session-seconds) the radio message was spoken, find the
    lap whose completion time is closest to it — this is the lap the driver's
    mood most likely affected.
    """
    if lap_df.empty:
        return None
    idx = (lap_df["timestamp_sec"] - radio_timestamp_sec).abs().idxmin()
    return int(lap_df.loc[idx, "lap"])
