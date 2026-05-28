"""
Local nutrition chunking with Hugging Face Transformers (CPU-friendly).

Hardware target: Ryzen 9 7940HS, 64 GB RAM, integrated GPU (not used for inference).
Default model: microsoft/Phi-3.5-mini-instruct (~3.8B) — fits in RAM, good speed on CPU.

Alternatives (pass --model):
  - Qwen/Qwen2.5-3B-Instruct     (faster, slightly smaller)
  - Qwen/Qwen2.5-7B-Instruct     (better quality, ~2–3x slower on CPU; 64 GB OK)

Reads:  raw/youtube/*.txt, raw/articles/*.md
Writes: chunks.json + chunks_local_checkpoint.json

Usage:
  .\\Scripts\\pip.exe install -r requirements.txt
  .\\Scripts\\python.exe scripts\\build_chunks_local.py
  .\\Scripts\\python.exe scripts\\build_chunks_local.py --limit 2   # smoke test
"""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))

except ImportError:

    def count_tokens(text: str) -> int:
        return len(text.split()) * 4 // 3


ROOT = Path(__file__).resolve().parent.parent
RAW_YT = ROOT / "raw" / "youtube"
RAW_ART = ROOT / "raw" / "articles"
OUT_FILE = ROOT / "chunks.json"
CHECKPOINT = ROOT / "chunks_local_checkpoint.json"
PER_SOURCE_DIR = ROOT / "chunks_local" / "by_source"

# Recommended for 64 GB RAM / CPU-only overnight runs
MODEL_DEFAULT = "microsoft/Phi-3.5-mini-instruct"

BATCH_CHARS = 2_400
PASSAGE_CHARS = 10_000
MIN_CHUNK_TOKENS = 180
MAX_CHUNK_TOKENS = 900

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "protein": ["protein", "amino acid", "leucine", "muscle"],
    "carbohydrates": ["carbohydrate", "carb", "glucose", "glycogen", "insulin"],
    "fats": ["fat", "lipid", "saturated", "omega"],
    "ketosis": ["ketosis", "ketone", "keto"],
    "fasting": ["fasting", "time-restricted", "intermittent"],
    "vitamins": ["vitamin", "micronutrient"],
    "minerals": ["magnesium", "zinc", "iron", "selenium", "electrolyte"],
    "gut health": ["gut", "microbiome", "fiber", "butyrate"],
    "supplements": ["supplement", "creatine", "berberine"],
    "metabolic health": ["metabolic", "insulin resistance", "blood sugar"],
}


SYSTEM_PROMPT = """You extract a nutrition knowledge base for RAG.

Rules:
- ONLY nutrition, diet, metabolism, supplements, food science, and closely related health.
- Skip ads, sponsors, CTAs, podcast plugs, and non-nutrition content.
- Do NOT add spin, hype, or opinions. Do NOT invent facts — only condense what is in the text.
- Preserve specific numbers, units, and study names when present.
- Each chunk = ONE atomic idea (one fact, mechanism, or concept).
- Each chunk must be self-contained prose (no "as mentioned above").
- Target roughly 300–600 words per chunk when the source supports it.

Output ONLY valid JSON (no markdown fences): an array of objects:
[{"content": "...", "topics": ["topic1", "topic2"]}]
Use 2–5 short topic labels per chunk. Return [] if no nutrition content."""


def infer_topics(text: str) -> list[str]:
    lower = text.lower()
    hits = [t for t, kws in TOPIC_KEYWORDS.items() if any(k in lower for k in kws)]
    return hits[:5] if hits else ["nutrition"]


def load_sources() -> list[dict]:
    sources: list[dict] = []
    for p in sorted(RAW_YT.glob("*.txt")):
        t = p.read_text(encoding="utf-8", errors="replace").strip()
        if len(t) > 200:
            sources.append({"type": "youtube", "id": p.stem, "text": t, "path": str(p)})
    for p in sorted(RAW_ART.glob("*.md")):
        t = p.read_text(encoding="utf-8", errors="replace").strip()
        if len(t) > 200:
            sources.append({"type": "article", "id": p.stem, "text": t, "path": str(p)})
    return sources


def split_batches(text: str, size: int = BATCH_CHARS) -> list[str]:
    if len(text) <= size:
        return [text]
    batches: list[str] = []
    overlap = 300
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            break_at = text.rfind("\n\n", start, end)
            if break_at > start + size // 2:
                end = break_at
        batches.append(text[start:end].strip())
        start = max(start + 1, end - overlap)
    return [b for b in batches if b]


def split_passages(text: str, max_chars: int = PASSAGE_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    paras = re.split(r"\n{2,}", text)
    buf: list[str] = []
    buf_len = 0
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if buf and buf_len + len(para) + 2 > max_chars:
            parts.append("\n\n".join(buf))
            buf, buf_len = [], 0
        buf.append(para)
        buf_len += len(para) + 2
    if buf:
        parts.append("\n\n".join(buf))
    return parts


def parse_json_array(raw: str) -> list[dict]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
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
        if not isinstance(topics, list) or not topics:
            topics = infer_topics(content)
        out.append({"content": content.strip(), "topics": topics[:6]})
    return out


class LocalChunker:
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        max_new_tokens: int = 1024,
    ):
        print(f"Loading {model_id} on {device} (first run downloads weights)...")
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        # transformers 5.x DynamicCache vs Phi-3 generate() — disable KV cache
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = False
        print("Model ready.")

    def _generate_messages(self, messages: list[dict], max_new_tokens: int) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def extract_chunks(self, passage: str) -> list[dict]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"SOURCE PASSAGE:\n\n{passage[:12_000]}"},
        ]
        raw = self._generate_messages(messages, self.max_new_tokens)
        return parse_json_array(raw)

    def clean_passage(self, raw: str) -> str:
        """Strip filler; keep dense nutrition prose (reference script style)."""
        messages = [
            {
                "role": "system",
                "content": "You are a precise editor. Output prose only.",
            },
            {
                "role": "user",
                "content": (
                    "Extract only substantive nutritional and health claims from this text. "
                    "Remove filler, transitions, sponsor mentions, and non-nutrition content. "
                    "Never paraphrase specific numbers or citations. "
                    "Output clean dense prose only — no bullet labels, no JSON.\n\n"
                    f"{raw[:8000]}"
                ),
            },
        ]
        return self._generate_messages(messages, 600)


def postprocess(items: list[dict]) -> list[dict]:
    out: list[dict] = []
    for it in items:
        content = it["content"]
        tok = count_tokens(content)
        if tok < MIN_CHUNK_TOKENS:
            continue
        if tok > MAX_CHUNK_TOKENS:
            content = " ".join(content.split()[:650])
        topics = it.get("topics") or infer_topics(content)
        out.append({"content": content.strip(), "topics": topics})
    return out


def _per_source_path(src_id: str) -> Path:
    return PER_SOURCE_DIR / f"{src_id}.json"


def _load_per_source_chunks(src_id: str) -> list[dict]:
    path = _per_source_path(src_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) and data else []


def reconcile_checkpoint(chunks: list[dict], processed: set[str]) -> tuple[list[dict], set[str]]:
    """
    Drop 'processed' markers that only exist because a prior run failed
    (empty per-source cache files).
    """
    valid_processed: set[str] = set()
    rebuilt = list(chunks)

    for key in processed:
        src_id = key.split(":", 1)[-1]
        file_chunks = _load_per_source_chunks(src_id)
        if file_chunks:
            valid_processed.add(key)
            existing = {c.get("content", "")[:220] for c in rebuilt}
            for it in file_chunks:
                ck = it.get("content", "")[:220]
                if ck and ck not in existing:
                    rebuilt.append(it)
                    existing.add(ck)

    dropped = len(processed) - len(valid_processed)
    if dropped:
        print(f"Reconciled checkpoint: dropped {dropped} empty/failed 'processed' markers")
    return rebuilt, valid_processed


def load_checkpoint() -> tuple[list[dict], set[str]]:
    if not CHECKPOINT.exists():
        return [], set()
    ck = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    chunks = ck.get("chunks", [])
    processed = set(ck.get("processed", []))
    if not isinstance(chunks, list):
        chunks = []
    return reconcile_checkpoint(chunks, processed)


def save_checkpoint(chunks: list[dict], processed: set[str]) -> None:
    CHECKPOINT.write_text(
        json.dumps({"chunks": chunks, "processed": sorted(processed)}, indent=2),
        encoding="utf-8",
    )


def process_source(chunker: LocalChunker, src: dict, *, two_pass: bool) -> list[dict]:
    text = src["text"]
    if two_pass:
        batches = split_batches(text)
        cleaned_parts: list[str] = []
        for bi, batch in enumerate(batches, 1):
            try:
                print(f"    clean {bi}/{len(batches)}...", flush=True)
                cleaned_parts.append(chunker.clean_passage(batch))
            except Exception as e:
                print(f"    clean batch {bi} failed: {e}")
        text = "\n\n".join(p for p in cleaned_parts if p.strip())

    all_items: list[dict] = []
    for passage in split_passages(text):
        try:
            items = chunker.extract_chunks(passage)
            all_items.extend(postprocess(items))
        except Exception as e:
            print(f"    chunk passage failed: {e}")
    return all_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Local HF chunking for nutrition RAG")
    parser.add_argument("--model", default=MODEL_DEFAULT, help="HuggingFace model id")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--limit", type=int, default=0, help="Max sources (0=all)")
    parser.add_argument(
        "--two-pass",
        action="store_true",
        help="Clean each 2.4k batch first, then chunk (slower, often cleaner)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear checkpoint and per-source cache, start fresh",
    )
    args = parser.parse_args()

    if args.reset:
        for p in (CHECKPOINT,):
            if p.exists():
                p.unlink()
        if PER_SOURCE_DIR.exists():
            for f in PER_SOURCE_DIR.glob("*.json"):
                f.unlink()
        print("Reset: cleared checkpoint and per-source cache\n")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    PER_SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    chunker = LocalChunker(args.model, device=device, max_new_tokens=args.max_new_tokens)
    sources = load_sources()
    if args.limit:
        sources = sources[: args.limit]

    print(f"\nProcessing {len(sources)} sources")
    print(f"Model: {args.model} | device: {device}")
    print(f"Mode: {'clean + chunk' if args.two_pass else 'chunk only (faster)'}\n")

    chunks, processed = load_checkpoint()
    seen = {c.get("content", "")[:220] for c in chunks if isinstance(c, dict)}
    print(f"Checkpoint: {len(chunks)} chunks, {len(processed)} sources done\n")

    t0 = time.time()

    for i, src in enumerate(sources, 1):
        key = f"{src['type']}:{src['id']}"
        if key in processed:
            continue

        cached = _load_per_source_chunks(src["id"])
        if cached and not args.limit:
            print(f"[{i}/{len(sources)}] skip (cached, {len(cached)} chunks) {key}")
            for it in cached:
                ck = it.get("content", "")[:220]
                if ck and ck not in seen:
                    seen.add(ck)
                    chunks.append(it)
            processed.add(key)
            save_checkpoint(chunks, processed)
            continue

        print(f"[{i}/{len(sources)}] {key}")
        src_start = time.time()
        extracted = process_source(chunker, src, two_pass=args.two_pass)

        added = 0
        file_chunks: list[dict] = []
        for it in extracted:
            dedupe = it["content"][:220]
            if dedupe in seen:
                continue
            seen.add(dedupe)
            row = {
                "id": str(uuid.uuid4()),
                "content": it["content"],
                "topics": it.get("topics", infer_topics(it["content"])),
                "embed": [],
            }
            chunks.append(row)
            file_chunks.append(row)
            added += 1

        _per_source_path(src["id"]).write_text(
            json.dumps(file_chunks, indent=2), encoding="utf-8"
        )
        if added > 0:
            processed.add(key)
            save_checkpoint(chunks, processed)
        else:
            print("  (0 chunks — not marking done; will retry next run)")
        elapsed = time.time() - src_start
        print(f"  +{added} chunks ({elapsed:.0f}s)")

    final = [
        {"id": c["id"], "content": c["content"], "topics": c.get("topics", []), "embed": []}
        for c in chunks
        if c.get("id") and c.get("content")
    ]
    OUT_FILE.write_text(json.dumps(final, indent=2), encoding="utf-8")
    total_min = (time.time() - t0) / 60
    print(f"\nWrote {len(final)} chunks -> {OUT_FILE}")
    print(f"Elapsed this run: {total_min:.1f} min")


if __name__ == "__main__":
    main()
