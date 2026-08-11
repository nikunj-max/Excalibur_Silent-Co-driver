---
title: The Silent Co-Driver
emoji: 🏎️
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: "4.36.0"
app_file: app.py
pinned: false
---

# 🏎️ The Silent Co-Driver
### Reading Driver Stress from Radio Calls

**Problem:** During a race, engineers are watching telemetry and can miss the *tone*
of what the driver is saying over the radio — tired, frustrated, or stressed voices
often carry warning signs the raw numbers don't show.

**Solution:** Upload a radio audio clip and a lap-time CSV. The app:
1. Transcribes the clip (speech → text) with **Whisper**.
2. Classifies the driver's tone (Calm / Stressed / Tired) with a **Wav2Vec2**
   speech-emotion-recognition model — this looks at *acoustic* features, not just words.
3. Lines the moment up against the lap-time chart, so you can see whether stress
   is showing up as slower laps.

---

## Architecture

```
User uploads Audio + Lap CSV
            │
            ▼
    Streamlit Frontend (app.py)
            │
            ▼
      backend.py (processing hub)
       ┌─────────┴─────────┐
       ▼                   ▼
 Whisper (ASR)      Wav2Vec2 (SER)
 openai/whisper-tiny  Dpngtm/wav2vec2-emotion-recognition
       │                   │
       └─────────┬─────────┘
                  ▼
     Transcript + Mood + Confidence
                  │
                  ▼
   Lap-time alignment (nearest_lap_for_timestamp)
                  │
                  ▼
      Dashboard: transcript box, mood badge,
             line chart of lap times
```

Both AI models are pre-trained and pulled directly from the Hugging Face Hub via
`transformers.pipeline` — nothing is trained from scratch, per the competition rules.

---

## Repo structure

```
silent-co-driver/
├── app.py                  # Gradio frontend (Space entrypoint)
├── app_streamlit.py         # Original Streamlit version, kept for reference/local use
├── backend.py               # Model loading, inference, CSV parsing, lap alignment
├── requirements.txt
├── README.md
└── sample_data/
    ├── calm_clip.wav        # mock radio clips for demoing without your own recordings
    ├── stressed_clip.wav
    ├── tired_clip.wav
    └── lap_times.csv        # mock lap data whose "story" matches the clips
```

> Note: `app.py` (Gradio) is the file Hugging Face Spaces runs. `app_streamlit.py` is
> the original Streamlit build — keep it if you ever unlock Docker and want to switch
> back, otherwise it's safe to ignore or delete.

## CSV format

Your lap-time file needs these columns:

| column          | meaning                                              |
|------------------|-------------------------------------------------------|
| `lap`            | lap number                                            |
| `lap_time_sec`   | lap time in seconds                                   |
| `timestamp_sec`  | seconds into the session when that lap finished       |

`timestamp_sec` is what lets the app match a radio call to the lap it most likely affected.

---

## Run locally

```bash
git clone <this-repo-url>
cd silent-co-driver
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Gradio will print a local URL like `http://127.0.0.1:7860`. Open it, pick a clip from
the **sample clip dropdown** (or upload your own), and click **Analyze radio call** to
see it work end to end without recording anything yourself.

## Use your own clips

Record 3–4 short clips of yourself reading driver-style radio messages in different
tones (calm / stressed / tired), drop them anywhere, and upload them through the app.
Build a matching CSV so the lap times dip during the "stressed" moment — that's the
story that makes the demo land.

---

## Deploy to Hugging Face Spaces

1. Create a new **Space** → SDK: **Gradio**.
2. Push (or upload) all files in this repo, including `sample_data/`.
3. The Space auto-installs `requirements.txt` and boots `app.py`.
4. First run will take a little longer while Whisper + Wav2Vec2 weights download —
   after that they're cached on the Space. Models load once at startup (see the
   `print()` lines in `app.py`), so the very first page load can take a minute or two
   while the Space builds and the weights download — check the **Logs** tab if it
   seems stuck.

---

## Models used (Hugging Face Hub)

- **ASR:** [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) — fast, accurate speech-to-text, handles background/radio noise well.
- **SER:** [`Dpngtm/wav2vec2-emotion-recognition`](https://huggingface.co/Dpngtm/wav2vec2-emotion-recognition) — classifies acoustic tone (angry, sad, neutral, happy, etc.), which we map to Calm / Stressed / Tired for the dashboard.

## Notes / next steps

- The emotion label mapping (`EMOTION_DISPLAY_MAP` in `backend.py`) is easy to tune —
  adjust which raw labels count as "Stressed" vs "Tired" as you test with real clips.
- Right now one clip = one mood snapshot lined up against the nearest lap. A nice
  extension: split a longer radio session into segments and plot mood over the
  whole race, not just one moment.
