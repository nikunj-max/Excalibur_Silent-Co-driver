# --- The Silent Co-Driver: Railway deployment image ---
FROM python:3.11-slim

# System libraries needed by librosa / soundfile / whisper for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch first (the default PyPI wheel pulls in ~2GB of CUDA
# libraries you don't need on Railway's CPU-only containers). Then install
# everything else from requirements.txt, skipping the torch line so it isn't
# reinstalled with the GPU build.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    grep -iv '^torch' requirements.txt > requirements.notorch.txt && \
    pip install --no-cache-dir -r requirements.notorch.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# Railway injects PORT at runtime; app.py reads it via os.environ.
EXPOSE 7860

CMD ["python", "app.py"]
