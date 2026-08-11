"""
app.py
------
The Silent Co-Driver — Race Engineer Dashboard
A Streamlit app that lets an engineer upload a driver radio clip + a lap-time CSV,
then shows:
  - the transcript (Whisper)
  - the detected mood (Wav2Vec2 speech-emotion-recognition)
  - a lap-time chart with the affected lap highlighted

Run locally:
    streamlit run app.py

Deploy: push this repo to a Hugging Face Space with SDK = Streamlit.
"""

import tempfile
import os

import streamlit as st
import pandas as pd

from backend import (
    load_asr_pipeline,
    load_ser_pipeline,
    analyze_clip,
    load_lap_data,
    nearest_lap_for_timestamp,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="The Silent Co-Driver",
    page_icon="🏎️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .big-metric { font-size: 2.2rem; font-weight: 700; }
    .transcript-box {
        background-color: #1e1e24;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
        font-size: 1.05rem;
        color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏎️ The Silent Co-Driver")
st.caption("Reading driver stress from radio calls — so nobody has to choose between watching the data and listening to the driver.")

# ---------------------------------------------------------------------------
# Cache heavy models so they load once per session, not on every rerun
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading speech-to-text model (Whisper)...")
def get_asr():
    return load_asr_pipeline()


@st.cache_resource(show_spinner="Loading emotion-recognition model (Wav2Vec2)...")
def get_ser():
    return load_ser_pipeline()


# ---------------------------------------------------------------------------
# TOP: uploaders
# ---------------------------------------------------------------------------
st.subheader("1. Upload the radio clip and lap data")

col_a, col_b, col_c = st.columns([1.2, 1.2, 1])

with col_a:
    audio_file = st.file_uploader("Radio audio clip", type=["wav", "mp3", "m4a", "flac"])

with col_b:
    csv_file = st.file_uploader("Lap-time CSV", type=["csv"])

with col_c:
    radio_timestamp = st.number_input(
        "Radio message timestamp (sec into session)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        help="Roughly when in the session this radio call happened. Used to line it up with the lap chart.",
    )

use_sample = st.checkbox("No files handy? Use the bundled sample data instead", value=False)

if use_sample:
    audio_path_for_sample = "sample_data/stressed_clip.wav"
    csv_path_for_sample = "sample_data/lap_times.csv"
    st.info(
        "Using bundled sample_data/. Record your own clips and drop them in that folder "
        "to replace these before your demo."
    )

# ---------------------------------------------------------------------------
# MIDDLE: run analysis
# ---------------------------------------------------------------------------
run = st.button("🔎 Analyze radio call", type="primary", disabled=not (audio_file or use_sample))

if run:
    # Resolve audio path (uploaded file needs to be written to a temp path for the pipeline)
    if audio_file is not None:
        suffix = os.path.splitext(audio_file.name)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_file.read())
            audio_path = tmp.name
        audio_display_source = audio_file
    else:
        audio_path = audio_path_for_sample
        audio_display_source = audio_path

    # Resolve lap CSV
    lap_df = None
    if csv_file is not None:
        lap_df = load_lap_data(csv_file)
    elif use_sample:
        lap_df = load_lap_data(csv_path_for_sample)

    asr_pipe = get_asr()
    ser_pipe = get_ser()

    with st.spinner("Transcribing and analyzing tone..."):
        result = analyze_clip(asr_pipe, ser_pipe, audio_path)

    st.subheader("2. What was said, and how it was said")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Audio playback**")
        st.audio(audio_display_source)

        st.markdown("**Transcript**")
        st.markdown(f'<div class="transcript-box">{result.transcript or "(no speech detected)"}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("**Detected mood**")
        st.markdown(
            f'<div class="big-metric" style="color:{result.color}">'
            f'{result.emoji} {result.display_label}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Model confidence: {result.confidence}% (raw label: {result.raw_emotion})")

    # -----------------------------------------------------------------
    # BOTTOM: lap chart
    # -----------------------------------------------------------------
    st.subheader("3. Does mood line up with lap performance?")

    if lap_df is not None and not lap_df.empty:
        affected_lap = nearest_lap_for_timestamp(lap_df, radio_timestamp)

        chart_df = lap_df.set_index("lap")[["lap_time_sec"]].rename(
            columns={"lap_time_sec": "Lap time (s)"}
        )
        st.line_chart(chart_df)

        if affected_lap is not None:
            row = lap_df[lap_df["lap"] == affected_lap].iloc[0]
            st.markdown(
                f"📍 This radio call lines up closest with **Lap {affected_lap}** "
                f"(**{row['lap_time_sec']:.2f}s**). "
                f"Detected mood at that moment: {result.emoji} **{result.display_label}**."
            )
        st.dataframe(lap_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No lap-time data provided — upload a CSV to see the mood-vs-pace chart.")

else:
    st.info("Upload an audio clip (and optionally a lap CSV), then click **Analyze radio call**.")

with st.expander("Expected lap CSV format"):
    st.code(
        "lap,lap_time_sec,timestamp_sec\n"
        "1,92.4,95\n"
        "2,91.8,190\n"
        "3,95.1,286\n",
        language="csv",
    )
    st.caption(
        "`timestamp_sec` is how many seconds into the session that lap finished — "
        "it's what lets the app line the radio call up with the right lap."
    )
