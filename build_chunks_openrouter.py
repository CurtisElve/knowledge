"""
Build nutrition-only atomic chunks using OpenRouter (DeepSeek free).

Reads:
  - raw/youtube/*.txt
  - raw/articles/*.md

Writes:
  - chunks_openrouter.json  (final: {id, content, topics, embed: []})
  - chunks_openrouter_checkpoint.json (resume state)

Usage (PowerShell):
  $env:OPENROUTER_API_KEY="sk-or-..."
  .\Scripts\python.exe scripts\build_chunks_openrouter.py

If the free model is upstream rate-limited (429), wait a bit and re-run.

Notes:
  - The script will NOT embed vectors; it leaves "embed": [].
  - It filters to nutrition/diet content only and tries to avoid any added spin.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from pathlib import Path

from openai import OpenAI

try:
    import tiktoken
except ImportError:
    tiktoken = None

ROOT = Path(__file__).resolve().parent.parent
RAW_YT = ROOT / "raw" / "youtube"
RAW_ART = ROOT / "raw" / "articles"

OUT_FILE = ROOT / "chunks_openrouter.json"
CHECKPOINT = ROOT / "chunks_openrouter_checkpoint.json"

MODEL_DEFAULT = "moonshotai/kimi-k2-6:free"

MIN_TOKENS = 400
MAX_TOKENS = 800
PASSAGE_CHARS = 11_000


SYSTEM_PROMPT = """You are extracting a nutrition knowledge base for retrieval-augmented generation.

STRICT RULES:
- Output MUST be valid JSON only (no markdown fences, no commentary).
- ONLY include nutrition/diet/supplement/food-science content. If the passage is not nutrition-related, output [].
- Do NOT add spin, hype, marketing tone, or personal opinions.
- Do NOT invent facts; only condense what is present. If uncertain or unsupported, omit it.
- Ignore ads, CTAs, cookie banners, nav menus, and social prompts.
- Each chunk must express exactly ONE atomic idea (one claim, mechanism, recommendation, or concept).
- Each chunk must be self-contained; do not refer to "this article" or "the author said above".
- Target 400–800 tokens per chunk.
- Preserve important numbers, units, and study details when present.
- topics must be 2–6 short labels (prefer snake_case or short phrases). No duplicates.

Return JSON array of objects:
  [{"content": "...", "topics": ["topic1", "topic2"]}, ...]
"""


def _count_tokens(text: str) -> int:
    if tiktoken:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    return len(text.split()) * 4 // 3


def _load_sources() -> list[dict]:
    sources: list[dict] = []
    for p in sorted(RAW_YT.glob("*.txt")):
        txt = p.read_text(encoding="utf-8", errors="replace").strip()
        if len(txt) > 200:
            sources.append({"type": "youtube", "id": p.stem, "text": txt})
    for p in sorted(RAW_ART.glob("*.md")):
        txt = p.read_text(encoding="utf-8", errors="replace").strip()
        if len(txt) > 200:
            sources.append({"type": "article", "id": p.stem, "text": txt})
    return sources


def _split_passages(text: str, max_chars: int = PASSAGE_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paras = re.split(r"\n{2,}", text)
    out: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if buf and buf_len + len(para) + 2 > max_chars:
            out.append("\n\n".join(buf))
            buf, buf_len = [], 0
        buf.append(para)
        buf_len += len(para) + 2
    if buf:
        out.append("\n\n".join(buf))
    return out


def _openrouter_client(api_key: str) -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_json_array(raw: str) -> list[dict]:
    raw = _strip_code_fences(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[[\s\S]*\]", raw)
        if not m:
            return []
        data = json.loads(m.group(0))

    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        topics = item.get("topics") or []
        if isinstance(topics, str):
            topics = [t.strip() for t in topics.split(",") if t.strip()]
        if not isinstance(topics, list):
            topics = []
        topics_clean: list[str] = []
        seen = set()
        for t in topics:
            if not isinstance(t, str):
                continue
            tt = re.sub(r"\s+", " ", t.strip())
            if not tt:
                continue
            key = tt.lower()
            if key in seen:
                continue
            seen.add(key)
            topics_clean.append(tt[:60])
        out.append({"content": content.strip(), "topics": topics_clean[:6]})
    return out


def _call_llm(client: OpenAI, model: str, passage: str, retries: int = 5) -> list[dict]:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"SOURCE PASSAGE:\n\n{passage[:14000]}"},
                ],
                temperature=0.1,
                extra_body={"reasoning": {"enabled": True}},
            )
            msg = resp.choices[0].message
            raw = msg.content or ""
            return _parse_json_array(raw)
        except Exception as e:
            last_err = e
            # Exponential backoff (OpenRouter free models may upstream-rate-limit)
            time.sleep(min(180, 10 * (2**attempt)))
    raise last_err  # type: ignore[misc]


def _postprocess_chunks(items: list[dict]) -> list[dict]:
    """Enforce basic token bounds; keep as close to 400-800 as possible."""
    out: list[dict] = []
    for it in items:
        content = it["content"]
        tok = _count_tokens(content)
        if tok < 180:
            continue
        if tok > 1100:
            # Hard-trim oversized output (prefer keeping first ~800 tokens)
            words = content.split()
            content = " ".join(words[:650])
            tok = _count_tokens(content)
        out.append({"content": content.strip(), "topics": it.get("topics", [])})
    return out


def _load_checkpoint() -> tuple[list[dict], set[str]]:
    if not CHECKPOINT.exists():
        return ([], set())
    ck = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    chunks = ck.get("chunks", [])
    processed = set(ck.get("processed", []))
    if not isinstance(chunks, list):
        chunks = []
    return (chunks, processed)


def _save_checkpoint(chunks: list[dict], processed: set[str]) -> None:
    CHECKPOINT.write_text(
        json.dumps({"chunks": chunks, "processed": sorted(processed)}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--limit", type=int, default=0, help="Max source docs (0=all)")
    parser.add_argument("--api-key", default="", help="OpenRouter API key (prefer env OPENROUTER_API_KEY)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing OpenRouter key. Set OPENROUTER_API_KEY or pass --api-key.")

    # Quick sanity check: OpenRouter keys are usually sk-or-v1-...
    if not api_key.startswith("sk-or-"):
        print("Warning: OPENROUTER_API_KEY does not start with 'sk-or-'.")

    client = _openrouter_client(api_key)
    sources = _load_sources()
    if args.limit:
        sources = sources[: args.limit]
    print(f"Loaded {len(sources)} source documents")

    chunks, processed = _load_checkpoint()
    seen = {c.get("content", "")[:220] for c in chunks if isinstance(c, dict)}
    print(f"Resuming: {len(chunks)} existing chunks; {len(processed)} sources done")

    for i, src in enumerate(sources, 1):
        src_key = f"{src['type']}:{src['id']}"
        if src_key in processed:
            continue
        print(f"\n[{i}/{len(sources)}] {src_key}")
        passages = _split_passages(src["text"])
        for pi, passage in enumerate(passages, 1):
            print(f"  passage {pi}/{len(passages)}...", end=" ", flush=True)
            extracted = _call_llm(client, args.model, passage)
            extracted = _postprocess_chunks(extracted)
            added = 0
            for it in extracted:
                key = it["content"][:220]
                if key in seen:
                    continue
                seen.add(key)
                chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "content": it["content"],
                        "topics": it.get("topics", []),
                        "embed": [],
                    }
                )
                added += 1
            print(f"+{added}")
            _save_checkpoint(chunks, processed)
            time.sleep(0.25)

        processed.add(src_key)
        _save_checkpoint(chunks, processed)

    # Final output: ensure only requested schema
    final = [
        {"id": c["id"], "content": c["content"], "topics": c.get("topics", []), "embed": []}
        for c in chunks
        if isinstance(c, dict) and c.get("id") and c.get("content")
    ]
    OUT_FILE.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"\nWrote {len(final)} chunks to {OUT_FILE}")


if __name__ == "__main__":
    main()

