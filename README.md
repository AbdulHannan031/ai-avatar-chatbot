George Avatar Chatbot (Dynavap RAG)

Overview

- Web chatbot that feels like chatting with “George,” the Dynavap expert from YouTube.
- Uses RAG over transcripts from George’s YouTube videos. Transcripts are generated using OpenAI speech-to-text (Whisper via `gpt-4o-mini-transcribe`).
- Starts with animated avatar + text responses; optional in-browser TTS.

Project Structure

- `server/` — FastAPI backend with RAG endpoint and transcript ingestion.
- `web/` — Minimal static chat UI with animated avatar, optional TTS, and source links under responses.
- `data/` — Transcripts and vector index (created after ingest).

Quick Start

1) Prereqs
- Python 3.10+
- An OpenAI API key in `OPENAI_API_KEY`
- Optional: YouTube Data API key in `YOUTUBE_API_KEY` (only for listing videos by channel; transcription is handled by OpenAI, not the YouTube API)

2) Install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
```

3) Ingest transcripts

Option A — Provide a list of video URLs/IDs in `data/videos.txt` (one per line):

```bash
python server/ingest.py --videos-file data/videos.txt --max-videos 50
```

Option B — Use a YouTube channel ID (needs `YOUTUBE_API_KEY`):

```bash
export YOUTUBE_API_KEY=YOUR_KEY
python server/ingest.py --channel-id UCxxxxxxxxxxxx --max-videos 50
```

This downloads audio via `yt-dlp`, transcribes it with OpenAI, then creates `data/transcripts.jsonl` and a FAISS index in `data/index/`.

4) Run the server

```bash
export OPENAI_API_KEY=YOUR_OPENAI_KEY
uvicorn server.app:app --reload --port 8000
```

Open http://localhost:8000 to use the chatbot.

Notes

- Persona: The assistant responds in George’s friendly, knowledgeable style focused on Dynavap. You can customize the prompt in `server/app.py`.
- TTS: The web client uses the browser’s `speechSynthesis` as a fallback. A server TTS provider can be added later.
- Assets: The avatar is a lightweight CSS/SVG animation; you can replace it with a Lottie or video later.
 - Avatar image: Place your George picture as `goerge.png` (or `george.png`) in the project root. The web UI loads it from `/avatar` and displays it as-is. Replace anytime.
 - Transcription model: Set `OPENAI_TRANSCRIBE_MODEL` to change the model (default `gpt-4o-mini-transcribe`).

Troubleshooting downloads/transcription

- Some videos require cookies or get rate-limited. Provide a cookies file (Netscape format, exported from your browser) and/or a proxy. These are passed to `yt-dlp` for the audio download:

```bash
python server/ingest.py --videos-file data/videos.txt \
  --cookies /path/to/youtube_cookies.txt \
  --proxy http://127.0.0.1:8080
```

- You can also set env vars: `YT_COOKIES` and `YT_PROXY`.
- If transcription returns empty, the script will error for visibility. Share the console output if you need help.
