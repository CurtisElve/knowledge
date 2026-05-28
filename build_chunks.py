"""
Break collected nutrition content into atomic chunks for RAG.

Uses free/cheap LLM backends (first available):
  - GEMINI_API_KEY  -> Google Gemini Flash
  - GROQ_API_KEY    -> Llama 3.x via Groq

Set one of those env vars, or pass --provider gemini|groq.

Output: chunks.json with {id, content, topics, embed: []}
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

try:
    import tiktoken
except ImportError:
    tiktoken = None

ROOT = Path(__file__).resolve().parent.parent
RAW_YT = ROOT / "raw" / "youtube"
RAW_ART = ROOT / "raw" / "articles"
OUT_FILE = ROOT / "chunks.json"
CHECKPOINT = ROOT / "chunks_checkpoint.json"

# ~400-800 tokens ≈ 300-600 words; we target ~500 tokens
MIN_TOKENS = 400
MAX_TOKENS = 800
CHUNK_BATCH_CHARS = 12_000  # feed LLM manageable slices of source text

SYSTEM_PROMPT = """You extract atomic nutrition knowledge for a RAG vector database.

Rules:
- ONLY nutrition, diet, metabolism, supplements, food science, and closely related health topics.
- Skip ads, CTAs, podcast plugs, unrelated anecdotes, and non-nutrition content.
- Each chunk expresses ONE clear idea (one fact, mechanism, recommendation, or concept).
- Write self-contained prose (no "as mentioned above"). Include enough context to stand alone.
- Target 400-800 tokens per chunk (roughly 300-600 words). Do not go under 300 words unless the idea is truly atomic and short.
- Use precise, educational language. Preserve numbers, study names, and mechanisms when present.
- topics: 2-5 short snake_case or plain labels (e.g. "omega-3", "insulin resistance", "protein synthesis").

Respond with ONLY valid JSON: an array of objects, each:
{"content": "...", "topics": ["topic1", "topic2"]}
No markdown fences. Empty array [] if no nutrition content in this passage."""


def count_tokens(text: str) -> int:
    if tiktoken:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    return len(text.split()) * 4 // 3  # rough estimate


def load_all_sources() -> list[dict]:
    sources: list[dict] = []
    for p in sorted(RAW_YT.glob("*.txt")):
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) > 200:
            sources.append({"id": p.stem, "type": "youtube", "text": text})
    for p in sorted(RAW_ART.glob("*.md")):
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) > 200:
            sources.append({"id": p.stem, "type": "article", "text": text})
    return sources


def split_text(text: str, max_chars: int = CHUNK_BATCH_CHARS) -> list[str]:
    """Split long documents on paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    paragraphs = re.split(r"\n{2,}", text)
    buf: list[str] = []
    buf_len = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if buf_len + len(para) + 2 > max_chars and buf:
            parts.append("\n\n".join(buf))
            buf, buf_len = [], 0
        buf.append(para)
        buf_len += len(para) + 2
    if buf:
        parts.append("\n\n".join(buf))
    return parts


def get_llm_client(provider: str):
    if provider == "gemini":
        import google.generativeai as genai

        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY for Gemini provider")
        genai.configure(api_key=key)
        return ("gemini", genai.GenerativeModel("gemini-2.0-flash"))
    if provider == "groq":
        from groq import Groq

        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Set GROQ_API_KEY for Groq provider")
        return ("groq", Groq(api_key=key))
    raise ValueError(f"Unknown provider: {provider}")


def call_llm(provider: str, client, user_text: str) -> list[dict]:
    prompt = f"{SYSTEM_PROMPT}\n\n---\nSOURCE PASSAGE:\n\n{user_text[:14000]}"

    if provider == "gemini":
        model = client
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 8192},
        )
        raw = resp.text.strip()
    else:
        groq_client = client
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"SOURCE PASSAGE:\n\n{user_text[:14000]}"},
            ],
            temperature=0.2,
            max_tokens=8192,
        )
        raw = resp.choices[0].message.content.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[[\s\S]*\]", raw)
        if not m:
            return []
        data = json.loads(m.group())

    if not isinstance(data, list):
        return []
    valid = []
    for item in data:
        if isinstance(item, dict) and item.get("content"):
            topics = item.get("topics") or []
            if isinstance(topics, str):
                topics = [t.strip() for t in topics.split(",")]
            valid.append({"content": item["content"].strip(), "topics": topics})
    return valid


def detect_provider() -> str | None:
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    return None


def save_checkpoint(chunks: list[dict], processed: set[str]) -> None:
    CHECKPOINT.write_text(
        json.dumps({"chunks": chunks, "processed": list(processed)}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["gemini", "groq"], default=None)
    parser.add_argument("--limit", type=int, default=0, help="Max source files (0=all)")
    args = parser.parse_args()

    provider = args.provider or detect_provider()
    if not provider:
        print("No LLM API key — falling back to rule-based chunker.")
        import chunk_heuristic

        chunk_heuristic.main()
        return

    _, client = get_llm_client(provider)
    print(f"Using provider: {provider}")

    sources = load_all_sources()
    if args.limit:
        sources = sources[: args.limit]
    print(f"Loaded {len(sources)} source documents")

    chunks: list[dict] = []
    processed: set[str] = set()
    if CHECKPOINT.exists():
        ck = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        chunks = ck.get("chunks", [])
        processed = set(ck.get("processed", []))
        print(f"Resuming: {len(chunks)} chunks, {len(processed)} sources done")

    seen_content: set[str] = {c["content"][:200] for c in chunks}

    for si, src in enumerate(sources, 1):
        key = f"{src['type']}:{src['id']}"
        if key in processed:
            continue

        print(f"\n[{si}/{len(sources)}] {key} ({len(src['text'])} chars)")
        passages = split_text(src["text"])
        src_chunks = 0

        for pi, passage in enumerate(passages, 1):
            print(f"  passage {pi}/{len(passages)}...", end=" ", flush=True)
            try:
                extracted = call_llm(provider, client, passage)
            except Exception as e:
                print(f"LLM error: {e}")
                time.sleep(5)
                continue

            for item in extracted:
                content = item["content"]
                tok = count_tokens(content)
                if tok < 150:
                    continue
                if tok > 1000:
                    # Trim oversized chunks at sentence boundary
                    words = content.split()
                    content = " ".join(words[:600])
                    tok = count_tokens(content)

                dedupe_key = content[:200]
                if dedupe_key in seen_content:
                    continue
                seen_content.add(dedupe_key)

                chunk_id = str(uuid.uuid4())
                chunks.append(
                    {
                        "id": chunk_id,
                        "content": content,
                        "topics": item.get("topics", []),
                        "embed": [],
                        "source": key,
                        "tokens": tok,
                    }
                )
                src_chunks += 1

            print(f"+{len(extracted)} raw")
            time.sleep(0.8)

        processed.add(key)
        save_checkpoint(chunks, processed)
        print(f"  => {src_chunks} chunks from this source (total: {len(chunks)})")

    # Final output: strip internal fields
    output = [
        {"id": c["id"], "content": c["content"], "topics": c["topics"], "embed": c["embed"]}
        for c in chunks
    ]
    OUT_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {len(output)} chunks to {OUT_FILE}")


if __name__ == "__main__":
    main()
