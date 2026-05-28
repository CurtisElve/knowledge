"""Fetch YouTube transcripts from URLs in yt.txt."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

ROOT = Path(__file__).resolve().parent.parent
YT_FILE = ROOT / "yt.txt"
OUT_DIR = ROOT / "raw" / "youtube"


def extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url.strip())
        if m:
            return m.group(1)
    return None


def load_urls() -> list[tuple[str, str]]:
    """Return deduplicated (video_id, original_url) pairs preserving order."""
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for line in YT_FILE.read_text(encoding="utf-8").splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        vid = extract_video_id(url)
        if not vid or vid in seen:
            continue
        seen.add(vid)
        result.append((vid, url))
    return result


def fetch_transcript(video_id: str, retries: int = 4) -> str:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            try:
                transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
            except NoTranscriptFound:
                transcript = transcript_list.find_generated_transcript(
                    [t.language_code for t in transcript_list]
                )
            snippets = transcript.fetch()
            return " ".join(s["text"] if isinstance(s, dict) else s.text for s in snippets)
        except Exception as e:
            last_err = e
            err = str(e).lower()
            if "blocking" in err or "ip" in err or "too many" in err:
                wait = 15 * (attempt + 1)
                print(f"rate-limited, wait {wait}s...", end=" ", flush=True)
                time.sleep(wait)
            else:
                raise
    raise last_err  # type: ignore[misc]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    urls = load_urls()
    manifest_path = OUT_DIR / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    ok, fail = 0, 0
    for i, (video_id, url) in enumerate(urls, 1):
        out_file = OUT_DIR / f"{video_id}.txt"
        prev = manifest.get(video_id, {})
        if out_file.exists() and out_file.stat().st_size > 50 and prev.get("status") == "ok":
            print(f"[{i}/{len(urls)}] skip {video_id}")
            ok += 1
            continue

        print(f"[{i}/{len(urls)}] {video_id} ...", end=" ", flush=True)
        try:
            text = fetch_transcript(video_id)
            out_file.write_text(text, encoding="utf-8")
            manifest[video_id] = {"url": url, "status": "ok", "chars": len(text)}
            print(f"ok ({len(text)} chars)")
            ok += 1
            time.sleep(1.5)
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            manifest[video_id] = {"url": url, "status": "no_transcript", "error": str(e)}
            print("no transcript")
            fail += 1
        except VideoUnavailable as e:
            manifest[video_id] = {"url": url, "status": "unavailable", "error": str(e)}
            print("unavailable")
            fail += 1
        except Exception as e:
            manifest[video_id] = {"url": url, "status": "error", "error": str(e)}
            print(f"error: {e}")
            fail += 1
            time.sleep(1)

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone: {ok} ok, {fail} failed, {len(urls)} unique videos")


if __name__ == "__main__":
    main()
