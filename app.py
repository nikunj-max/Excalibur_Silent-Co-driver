"""
app.py (Gradio version)
------------------------
The Silent Co-Driver — Race Engineer Dashboard
A Gradio app that lets an engineer upload a driver radio clip + a lap-time CSV,
then shows:
  - the transcript (Whisper)
  - the detected mood (Wav2Vec2 speech-emotion-recognition)
  - a lap-time chart with the affected lap highlighted

Run locally:
    python app.py

Deploy: push this repo to a Hugging Face Space with SDK = Gradio.
"""

import tempfile
import os

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

from backend import (
    load_asr_pipeline,
    load_ser_pipeline,
    analyze_clip,
    load_lap_data,
    nearest_lap_for_timestamp,
)

# ---------------------------------------------------------------------------
# Load models once at startup (Gradio has no per-run rerun model like Streamlit,
# so we just load them at import time and reuse across requests)
# ---------------------------------------------------------------------------
print("Loading Whisper (ASR)...")
ASR_PIPE = load_asr_pipeline()
print("Loading Wav2Vec2 (SER)...")
SER_PIPE = load_ser_pipeline()
print("Models ready.")

SAMPLE_AUDIO = {
    "Calm clip": "sample_data/calm_clip.wav",
    "Stressed clip": "sample_data/stressed_clip.wav",
    "Tired clip": "sample_data/tired_clip.wav",
}
SAMPLE_CSV = "sample_data/lap_times.csv"


def build_lap_chart(lap_df: pd.DataFrame, affected_lap: int | None):
    """Builds a matplotlib line chart of lap times, highlighting the affected lap."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(lap_df["lap"], lap_df["lap_time_sec"], marker="o", color="#4da3ff", linewidth=2)

    if affected_lap is not None and affected_lap in lap_df["lap"].values:
        row = lap_df[lap_df["lap"] == affected_lap].iloc[0]
        ax.scatter([row["lap"]], [row["lap_time_sec"]], color="#e74c3c", s=140, zorder=5,
                   label=f"Radio call (Lap {affected_lap})")
        ax.legend()

    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap time (s)")
    ax.set_title("Lap times")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def analyze(audio_path, csv_path, sample_choice, radio_timestamp):
    """
    Main callback. Falls back to bundled sample data if the user hasn't
    uploaded their own audio / CSV.
    """
    # Resolve audio source
    if audio_path is None:
        if not sample_choice:
            return "⚠️ Please upload an audio clip or pick a sample clip.", "", None, None
        audio_path = SAMPLE_AUDIO[sample_choice]

    # Resolve lap CSV source
    if csv_path is None:
        csv_path = SAMPLE_CSV

    try:
        lap_df = load_lap_data(csv_path)
    except Exception as e:
        return f"⚠️ Could not read lap CSV: {e}", "", None, None

    result = analyze_clip(ASR_PIPE, SER_PIPE, audio_path)

    mood_text = f"{result.emoji} **{result.display_label}**  \nConfidence: {result.confidence}% (raw label: `{result.raw_emotion}`)"

    affected_lap = nearest_lap_for_timestamp(lap_df, radio_timestamp or 0)
    note = ""
    if affected_lap is not None:
        row = lap_df[lap_df["lap"] == affected_lap].iloc[0]
        note = (
            f"\n\n📍 This call lines up closest with **Lap {affected_lap}** "
            f"(**{row['lap_time_sec']:.2f}s**)."
        )

    fig = build_lap_chart(lap_df, affected_lap)

    return result.transcript or "(no speech detected)", mood_text + note, fig, lap_df


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="The Silent Co-Driver", theme=gr.themes.Soft(primary_hue="red")) as demo:
    gr.Markdown(
        "# 🏎️ The Silent Co-Driver\n"
        "Reading driver stress from radio calls — so nobody has to choose between "
        "watching the data and listening to the driver."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload your clip and lap data")
            audio_input = gr.Audio(label="Radio audio clip", type="filepath")
            csv_input = gr.File(label="Lap-time CSV", file_types=[".csv"])
            sample_dropdown = gr.Dropdown(
                choices=list(SAMPLE_AUDIO.keys()),
                label="...or pick a bundled sample clip instead",
                value=None,
            )
            timestamp_input = gr.Number(
                label="Radio message timestamp (sec into session)",
                value=0,
                precision=0,
            )
            analyze_btn = gr.Button("🔎 Analyze radio call", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 2. What was said, and how it was said")
            transcript_output = gr.Textbox(label="Transcript", lines=3)
            mood_output = gr.Markdown(label="Detected mood")

    gr.Markdown("### 3. Does mood line up with lap performance?")
    chart_output = gr.Plot(label="Lap times")
    lap_table_output = gr.Dataframe(label="Lap data", interactive=False)

    with gr.Accordion("Expected lap CSV format", open=False):
        gr.Markdown(
            "```\n"
            "lap,lap_time_sec,timestamp_sec\n"
            "1,92.4,95\n"
            "2,91.8,190\n"
            "3,95.1,286\n"
            "```\n"
            "`timestamp_sec` is how many seconds into the session that lap finished — "
            "it lines the radio call up with the right lap."
        )

    analyze_btn.click(
        fn=analyze,
        inputs=[audio_input, csv_input, sample_dropdown, timestamp_input],
        outputs=[transcript_output, mood_output, chart_output, lap_table_output],
    )

if __name__ == "__main__":
    demo.launch()
