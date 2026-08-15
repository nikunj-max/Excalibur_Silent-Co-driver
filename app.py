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

Deploy: containerized with the included Dockerfile and deployed on Railway.
Railway injects a PORT environment variable at runtime, which this app reads.
"""

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

# ---------------------------------------------------------------------------
# Racing HUD theme — dark cockpit at night, red/amber floodlights, glass panels
# ---------------------------------------------------------------------------
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.red,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Rajdhani"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#0b0d12",
    body_background_fill_dark="#0b0d12",
    block_background_fill="rgba(22, 25, 34, 0.72)",
    block_border_color="rgba(230, 57, 70, 0.35)",
    block_border_width="1px",
    block_radius="18px",
    block_label_text_color="#ff8a5c",
    block_title_text_color="#f5f5f5",
    body_text_color="#e8e8e8",
    button_primary_background_fill="linear-gradient(90deg, #e63946 0%, #ff8500 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #ff4d5e 0%, #ffa03a 100%)",
    button_primary_text_color="#0b0d12",
    shadow_drop_lg="0 10px 40px rgba(230, 57, 70, 0.3)",
    input_background_fill="rgba(11, 13, 18, 0.6)",
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Orbitron:wght@600;800&display=swap');

/* ---------- Animated night-track background ---------- */
.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    position: relative;
}

body, .gradio-container {
    background:
        radial-gradient(ellipse 900px 500px at 15% -5%, rgba(230, 57, 70, 0.30), transparent 60%),
        radial-gradient(ellipse 900px 500px at 85% 0%, rgba(255, 133, 0, 0.22), transparent 60%),
        radial-gradient(ellipse 1200px 700px at 50% 110%, rgba(78, 42, 255, 0.12), transparent 60%),
        repeating-linear-gradient(115deg, rgba(255,255,255,0.02) 0px, rgba(255,255,255,0.02) 2px, transparent 2px, transparent 60px),
        #0b0d12 !important;
    background-attachment: fixed !important;
    animation: hud-glow 12s ease-in-out infinite alternate;
}

@keyframes hud-glow {
    0%   { filter: brightness(1); }
    50%  { filter: brightness(1.06); }
    100% { filter: brightness(1); }
}

footer { visibility: hidden; }

/* ---------- Hero header ---------- */
#hero {
    position: relative;
    text-align: center;
    padding: 34px 16px 20px 16px;
    margin-bottom: 8px;
    overflow: hidden;
    border-radius: 20px;
}
#hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        -45deg,
        rgba(255,255,255,0.05) 0 14px,
        rgba(0,0,0,0.05) 14px 28px
    );
    opacity: 0.35;
    -webkit-mask-image: linear-gradient(to bottom, black, transparent 85%);
            mask-image: linear-gradient(to bottom, black, transparent 85%);
    pointer-events: none;
}
#hero h1 {
    position: relative;
    font-family: 'Orbitron', 'Rajdhani', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: 3px;
    margin: 0 0 6px 0;
    background: linear-gradient(90deg, #ff4d5e, #ff8500 55%, #ffd166);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 40px rgba(255, 133, 0, 0.25);
    text-transform: uppercase;
}
#hero p {
    position: relative;
    color: #b8bcc4;
    font-size: 1.05rem;
    letter-spacing: 0.5px;
    max-width: 640px;
    margin: 0 auto;
}
#hero .flag-strip {
    position: relative;
    width: 220px;
    height: 6px;
    margin: 14px auto 0 auto;
    border-radius: 3px;
    background: repeating-linear-gradient(90deg, #f5f5f5 0 10px, #0b0d12 10px 20px);
    box-shadow: 0 0 18px rgba(255, 255, 255, 0.35);
}

/* ---------- Section headings ---------- */
.step-heading {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    letter-spacing: 1px;
    color: #ffb366 !important;
    text-transform: uppercase;
    border-left: 4px solid #e63946;
    padding-left: 10px;
    margin-bottom: 4px !important;
}

/* ---------- Glass panels ---------- */
.gr-block, .gr-box, .block {
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}

/* ---------- Buttons ---------- */
button.primary {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    box-shadow: 0 6px 24px rgba(230, 57, 70, 0.45) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
button.primary:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 10px 32px rgba(255, 133, 0, 0.55) !important;
}

/* ---------- Transcript / mood output ---------- */
#mood-panel {
    font-size: 1.05rem;
    line-height: 1.5;
}

/* ---------- Dataframe & plot polish ---------- */
.gr-plot, .plot-container {
    border-radius: 14px !important;
    overflow: hidden;
}
table {
    border-radius: 10px !important;
    overflow: hidden;
}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #0b0d12; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #e63946, #ff8500);
    border-radius: 6px;
}
"""

HERO_HTML = """
<div id="hero">
  <h1>🏎️ The Silent Co-Driver</h1>
  <p>Reading driver stress from radio calls — so nobody has to choose between
  watching the telemetry and listening to the driver.</p>
  <div class="flag-strip"></div>
</div>
"""


def build_lap_chart(lap_df: pd.DataFrame, affected_lap: int | None):
    """Builds a matplotlib line chart of lap times, highlighting the affected lap."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#14171f")
    ax.set_facecolor("#14171f")

    ax.plot(lap_df["lap"], lap_df["lap_time_sec"], marker="o", color="#ff8500", linewidth=2.4,
             markerfacecolor="#ffd166", markeredgecolor="#0b0d12", markersize=7)

    if affected_lap is not None and affected_lap in lap_df["lap"].values:
        row = lap_df[lap_df["lap"] == affected_lap].iloc[0]
        ax.scatter([row["lap"]], [row["lap_time_sec"]], color="#e63946", s=180, zorder=5,
                   edgecolors="white", linewidths=1.4,
                   label=f"Radio call (Lap {affected_lap})")
        ax.legend(facecolor="#1a1d26", edgecolor="#e63946", labelcolor="white")

    ax.set_xlabel("Lap", color="#e8e8e8")
    ax.set_ylabel("Lap time (s)", color="#e8e8e8")
    ax.set_title("Lap times", color="#ffffff", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#c8c8c8")
    ax.grid(alpha=0.2, color="#555")
    for spine in ax.spines.values():
        spine.set_color("#3a3d46")
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
    mood_text = f"{result.emoji} **{result.display_label}** \nConfidence: {result.confidence}% (raw label: `{result.raw_emotion}`)"

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
with gr.Blocks(title="The Silent Co-Driver", theme=THEME, css=CUSTOM_CSS) as demo:
    gr.HTML(HERO_HTML)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1 · Upload your clip and lap data", elem_classes=["step-heading"])
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
            analyze_btn = gr.Button("🔎 Analyze Radio Call", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 2 · What was said, and how it was said", elem_classes=["step-heading"])
            transcript_output = gr.Textbox(label="Transcript", lines=3)
            mood_output = gr.Markdown(label="Detected mood", elem_id="mood-panel")

    gr.Markdown("### 3 · Does mood line up with lap performance?", elem_classes=["step-heading"])
    chart_output = gr.Plot(label="Lap times")
    lap_table_output = gr.Dataframe(label="Lap data", interactive=False)

    with gr.Accordion("📋 Expected lap CSV format", open=False):
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
    # Railway (and most PaaS platforms) inject the port to bind to via $PORT.
    # Binding to 0.0.0.0 is required so the platform's proxy can reach the app.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
