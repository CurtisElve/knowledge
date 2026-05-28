"""Fetch article content as markdown via Jina Reader API."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parent.parent
ARTICLES_FILE = ROOT / "articles.txt"
OUT_DIR = ROOT / "raw" / "articles"

# Jina Reader: https://jina.ai/reader/
import os

JINA_API_KEY = os.environ.get(
    "JINA_API_KEY",
    "jina_d6050128830c4b11a9a7690b3871aaf0vTPl25bv-JxxXjoWOmhr4dJtzQ_1",
)
JINA_BASE = "https://r.jina.ai"


def slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_") or "index"
    host = parsed.netloc.replace(".", "_")
    base = f"{host}_{path}"[:120]
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    safe = re.sub(r"[^\w\-]", "_", base)
    return f"{safe}_{h}"


def load_urls() -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for line in ARTICLES_FILE.read_text(encoding="utf-8").splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def fetch_markdown(url: str) -> str:
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "X-Return-Format": "markdown",
        "Accept": "text/markdown",
    }
    resp = requests.get(f"{JINA_BASE}/{url}", headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.text


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    urls = load_urls()
    manifest_path = OUT_DIR / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    ok, fail = 0, 0
    for i, url in enumerate(urls, 1):
        slug = slug_from_url(url)
        out_file = OUT_DIR / f"{slug}.md"
        if out_file.exists() and out_file.stat().st_size > 100:
            print(f"[{i}/{len(urls)}] skip {slug}")
            ok += 1
            continue

        print(f"[{i}/{len(urls)}] {url[:70]}...", flush=True)
        try:
            md = fetch_markdown(url)
            out_file.write_text(md, encoding="utf-8")
            manifest[slug] = {"url": url, "status": "ok", "chars": len(md)}
            print(f"  ok ({len(md)} chars)")
            ok += 1
            time.sleep(0.5)
        except Exception as e:
            manifest[slug] = {"url": url, "status": "error", "error": str(e)}
            print(f"  error: {e}")
            fail += 1
            time.sleep(2)

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone: {ok} ok, {fail} failed, {len(urls)} articles")


if __name__ == "__main__":
    main()
