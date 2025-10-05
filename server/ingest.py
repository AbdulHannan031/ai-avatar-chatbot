import os
import re
import json
import time
import argparse
import pathlib
from typing import List, Dict, Any, Optional

import numpy as np
import faiss  # type: ignore
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import yt_dlp
import requests

from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
META_PATH = INDEX_DIR / "meta.json"
INDEX_PATH = INDEX_DIR / "faiss.index"


def extract_video_id(url_or_id: str) -> str:
    url_or_id = url_or_id.strip()
    # If it looks like a bare ID, return it
    if re.fullmatch(r"[\w-]{11}", url_or_id):
        return url_or_id
    # Extract from YouTube URL patterns
    m = re.search(r"v=([\w-]{11})", url_or_id)
    if m:
        return m.group(1)
    m = re.search(r"youtu\.be/([\w-]{11})", url_or_id)
    if m:
        return m.group(1)
    raise ValueError(f"Could not parse video id from: {url_or_id}")


def list_channel_video_ids(channel_id: str, api_key: str, max_videos: int = 50) -> List[str]:
    # Uses YouTube Data API to list uploads
    base = "https://www.googleapis.com/youtube/v3"
    # Get uploads playlist
    r = requests.get(
        f"{base}/channels",
        params={"part": "contentDetails", "id": channel_id, "key": api_key},
        timeout=30,
    )
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        return []
    uploads_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    video_ids: List[str] = []
    page_token: Optional[str] = None
    while len(video_ids) < max_videos:
        r2 = requests.get(
            f"{base}/playlistItems",
            params={
                "part": "contentDetails,snippet",
                "playlistId": uploads_id,
                "maxResults": 50,
                "pageToken": page_token,
                "key": api_key,
            },
            timeout=30,
        )
        r2.raise_for_status()
        j = r2.json()
        for it in j.get("items", []):
            vid = it["contentDetails"]["videoId"]
            video_ids.append(vid)
            if len(video_ids) >= max_videos:
                break
        page_token = j.get("nextPageToken")
        if not page_token:
            break
    return video_ids


def _video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def download_audio(video_id: str, cookies: Optional[str] = None, proxy: Optional[str] = None) -> pathlib.Path:
    # Prefer m4a to avoid needing ffmpeg remux; yt-dlp will fetch a playable audio file
    outdir = DATA_DIR / "tmp"
    outdir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(outdir / f"{video_id}.%(ext)s")
    ydl_opts: Dict[str, Any] = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    if cookies:
        ydl_opts["cookiefile"] = cookies
    if proxy:
        ydl_opts["proxy"] = proxy
    url = _video_url(video_id)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        info = ydl.extract_info(url, download=False)
        # Compute expected filepath
        ext = info.get("ext", "m4a")
        path = outdir / f"{video_id}.{ext}"
        if not path.exists():
            # Fallback: search for any file with this id
            for p in outdir.glob(f"{video_id}.*"):
                path = p
                break
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found for {video_id} in {outdir}")
        return path


def transcribe_audio(client: OpenAI, file_path: pathlib.Path) -> str:
    with open(file_path, "rb") as fh:
        resp = client.audio.transcriptions.create(
            model=os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe"),
            file=fh,
        )
    # openai>=1.0 returns .text
    return getattr(resp, "text", "")


def fetch_transcript(video_id: str, cookies: Optional[str] = None, proxy: Optional[str] = None) -> List[Dict[str, Any]]:
    # Use OpenAI transcription on downloaded audio
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for transcription. Set and retry.")
    client = OpenAI()
    audio_path = download_audio(video_id, cookies=cookies, proxy=proxy)
    try:
        text = transcribe_audio(client, audio_path)
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass
    if not text:
        raise RuntimeError("Transcription returned empty text")
    # Return a compatible segments structure
    return [{"text": text}]


def fetch_title(video_id: str) -> str:
    # lightweight oEmbed call (no API key) for title
    try:
        r = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=20,
        )
        if r.status_code == 200:
            return r.json().get("title", f"YouTube Video {video_id}")
    except Exception:
        pass
    return f"YouTube Video {video_id}"


def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    arr = np.array([d.embedding for d in resp.data], dtype="float32")
    return arr


def build_index(chunks: List[Dict[str, Any]]):
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for embedding. Set and retry.")
    client = OpenAI()

    texts = [c["text"] for c in chunks]
    vecs = embed_texts(client, texts)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    # Normalize for cosine similarity
    faiss.normalize_L2(vecs)
    index.add(vecs)
    return index, chunks


def save_index(index: faiss.Index, meta: List[Dict[str, Any]]):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def read_videos_file(path: pathlib.Path) -> List[str]:
    if not path.exists():
        return []
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            vid = extract_video_id(line)
            ids.append(vid)
        except Exception:
            pass
    return ids


def main():
    parser = argparse.ArgumentParser(description="Ingest YouTube transcripts and build index")
    parser.add_argument("--channel-id", type=str, default="", help="YouTube channel ID")
    parser.add_argument("--videos-file", type=str, default="data/videos.txt", help="File with video URLs/IDs")
    parser.add_argument("--max-videos", type=int, default=50, help="Max videos to fetch")
    parser.add_argument("--cookies", type=str, default=os.getenv("YT_COOKIES", ""), help="Path to YouTube cookies.txt for age/region/consent bypass")
    parser.add_argument("--proxy", type=str, default=os.getenv("YT_PROXY", ""), help="HTTP/HTTPS proxy URL (e.g., http://127.0.0.1:8080)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    transcripts_path = DATA_DIR / "transcripts.jsonl"

    videos: List[str] = []
    if args.channel_id:
        api_key = os.getenv("YOUTUBE_API_KEY", "")
        if not api_key:
            raise SystemExit("YOUTUBE_API_KEY is required to fetch by channel.")
        print(f"Listing videos for channel {args.channel_id} (max {args.max_videos})...")
        videos = list_channel_video_ids(args.channel_id, api_key, args.max_videos)
    else:
        videos_file = BASE_DIR / args.videos_file
        videos = read_videos_file(videos_file)

    if not videos:
        raise SystemExit("No videos to process. Provide --channel-id or a non-empty videos file.")

    print(f"Will ingest {len(videos)} videos...")
    proxy_str: Optional[str] = None
    if args.proxy:
        proxy_str = args.proxy
        print(f"Using proxy for YouTube: {proxy_str}")
    cookies_path: Optional[str] = None
    if args.cookies:
        cp = pathlib.Path(args.cookies)
        if cp.exists():
            cookies_path = str(cp)
            print(f"Using cookies file: {cookies_path}")
        else:
            print(f"Warning: cookies file not found: {cp}")
    all_records: List[Dict[str, Any]] = []
    kept_chunks: List[Dict[str, Any]] = []
    for i, vid in enumerate(videos, 1):
        url = f"https://www.youtube.com/watch?v={vid}"
        title = fetch_title(vid)
        t = fetch_transcript(vid, cookies=cookies_path, proxy=proxy_str)
        # Compose full text then chunk
        full_text = " ".join([seg.get("text", "") for seg in t])
        chunks = chunk_text(full_text)
        for c in chunks:
            kept_chunks.append({
                "video_id": vid,
                "url": url,
                "title": title,
                "text": c,
            })
        all_records.append({
            "video_id": vid,
            "url": url,
            "title": title,
            "segments": t,
        })
        print(f"[{i}/{len(videos)}] {title} — {len(chunks)} chunks")
        time.sleep(0.25)  # be gentle

    with open(transcripts_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Embedding and building index...")
    index, meta = build_index(kept_chunks)
    save_index(index, meta)
    print(f"Done. Index at {INDEX_PATH}, meta at {META_PATH}")


if __name__ == "__main__":
    load_dotenv()
    main()
