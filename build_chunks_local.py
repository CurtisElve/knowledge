"""
Local nutrition chunking with Hugging Face Transformers (Apple Silicon / MPS).

Hardware target: MacBook Air M3, 16 GB unified memory — uses Metal (MPS) for inference.
Default model: Qwen/Qwen2.5-3B-Instruct (~3B, 32k context) — fast on MPS, fits 16 GB.

Alternatives (pass --model):
  - microsoft/Phi-3.5-mini-instruct  (slightly larger; use --attn eager if MPS errors)
  - Qwen/Qwen2.5-1.5B-Instruct       (fastest; good for overnight smoke runs)

Reads:  raw/youtube/*.txt, raw/articles/*.md
Writes: chunks.json + chunks_local_checkpoint.json

Usage:
  pip install -r requirements.txt
  python build_chunks_local.py
  python build_chunks_local.py --limit 2          # smoke test
  python build_chunks_local.py --reset            # clear bad checkpoint
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


ROOT = Path(__file__).resolve().parent
RAW_YT = ROOT / "raw" / "youtube"
RAW_ART = ROOT / "raw" / "articles"
OUT_FILE = ROOT / "chunks.json"
CHECKPOINT = ROOT / "chunks_local_checkpoint.json"
PER_SOURCE_DIR = ROOT / "chunks_local" / "by_source"

# M3 16 GB + MPS: 3B fp16 ~6 GB; leaves headroom for long passages
MODEL_DEFAULT = "Qwen/Qwen2.5-3B-Instruct"

# Larger windows → fewer LLM calls on long transcripts (Qwen 3B supports 32k tokens)
BATCH_CHARS = 4_000
PASSAGE_CHARS = 22_000
PASSAGE_INPUT_CHARS = 20_000
# Local 3B models often emit 1–3 sentence facts (~20–45 tokens). Do not set this high.
MIN_CHUNK_TOKENS = 15
MAX_CHUNK_TOKENS = 320

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


SYSTEM_PROMPT = """You extract atomic nutrition facts for a RAG vector database.

Rules:
- ONLY nutrition, diet, metabolism, supplements, food science, and closely related health.
- Skip ads, sponsors, CTAs, podcast plugs, and non-nutrition content.
- Do NOT add spin, hype, or opinions. Do NOT invent facts — only condense what is in the text.
- Preserve specific numbers, units, and study names when present.
- Each chunk = ONE atomic statement (one fact, mechanism, dose, or recommendation).
- Write 2–6 self-contained sentences (no "as mentioned above"). Enough context to stand alone.
- Each content field: at least 2 complete sentences with concrete detail (doses, mechanisms, or numbers when present).
- Split compound ideas into separate chunks; do not merge unrelated facts.

Output ONLY valid JSON (no markdown fences): an array of objects:
[{"content": "...", "topics": ["topic1", "topic2"]}]
Use 2–5 short topic labels per chunk. Return [] if no nutrition content.
JSON rules: double-quote all strings; escape internal quotes as \\"; no raw line breaks inside strings; no trailing commas."""


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


def split_passages(text: str, max_chars: int | None = None) -> list[str]:
    limit = PASSAGE_CHARS if max_chars is None else max_chars
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    paras = re.split(r"\n{2,}", text)
    buf: list[str] = []
    buf_len = 0
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if buf and buf_len + len(para) + 2 > limit:
            parts.append("\n\n".join(buf))
            buf, buf_len = [], 0
        buf.append(para)
        buf_len += len(para) + 2
    if buf:
        parts.append("\n\n".join(buf))
    return parts


def _strip_code_fences(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```\s*$", "", raw)
    return raw.strip()


def _sanitize_json_text(raw: str) -> str:
    """Fix common LLM JSON issues before parsing."""
    raw = raw.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    # Invalid unescaped control chars break json.loads (except tab/newline/carriage return)
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", raw)
    return raw


def _fix_trailing_commas(raw: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", raw)


def _find_balanced_brace(text: str, start: int) -> int:
    if start >= len(text) or text[start] != "{":
        return -1
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _repair_object_fragment(fragment: str) -> str:
    fragment = _fix_trailing_commas(
        _escape_newlines_in_json_strings(_sanitize_json_text(fragment.strip()))
    )
    if fragment.count('"') % 2 == 1:
        fragment += '"'
    missing = fragment.count("{") - fragment.count("}")
    if missing > 0:
        fragment += "}" * missing
    return fragment


def _escape_newlines_in_json_strings(raw: str) -> str:
    """Turn literal newlines inside JSON strings into spaces (invalid JSON otherwise)."""
    out: list[str] = []
    in_str = False
    esc = False
    for ch in raw:
        if in_str:
            if esc:
                esc = False
                out.append(ch)
            elif ch == "\\":
                esc = True
                out.append(ch)
            elif ch == '"':
                in_str = False
                out.append(ch)
            elif ch in "\n\r":
                out.append(" ")
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_str = True
            out.append(ch)
    return "".join(out)


def _try_load_json(raw: str) -> object | None:
    candidates = [
        raw,
        _fix_trailing_commas(_sanitize_json_text(raw)),
        _fix_trailing_commas(_escape_newlines_in_json_strings(_sanitize_json_text(raw))),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _salvage_json_objects(raw: str) -> list[dict]:
    """Pull complete {...} objects out of truncated or malformed JSON arrays."""
    raw = _strip_code_fences(raw)
    start = raw.find("[")
    scan_from = start + 1 if start >= 0 else 0
    objects: list[dict] = []
    seen: set[str] = set()
    i = scan_from
    while i < len(raw):
        if raw[i] != "{":
            i += 1
            continue
        end = _find_balanced_brace(raw, i)
        if end < 0:
            break
        fragment = raw[i : end + 1]
        obj = _try_load_json(fragment)
        if obj is None:
            obj = _try_load_json(_repair_object_fragment(fragment))
        if isinstance(obj, dict):
            key = json.dumps(obj, sort_keys=True)[:200]
            if key not in seen:
                seen.add(key)
                objects.append(obj)
        i = end + 1
    return objects


def _normalize_chunk_item(item: dict) -> dict | None:
    content = item.get("content")
    if not isinstance(content, str) or not content.strip():
        for alt in ("text", "chunk", "body", "statement"):
            v = item.get(alt)
            if isinstance(v, str) and v.strip():
                content = v
                break
    if not isinstance(content, str) or not content.strip():
        return None
    topics = item.get("topics") or []
    if isinstance(topics, str):
        topics = [t.strip() for t in topics.split(",") if t.strip()]
    if not isinstance(topics, list) or not topics:
        topics = infer_topics(content)
    return {"content": content.strip(), "topics": topics[:6]}


def parse_json_array(raw: str) -> list[dict]:
    """
    Parse model JSON output. Never raises — malformed LLM JSON is salvaged when possible.
    """
    if not raw or not raw.strip():
        return []

    cleaned = _sanitize_json_text(_strip_code_fences(raw))
    data: object | None = None

    data = _try_load_json(cleaned)
    if data is None:
        m = re.search(r"\[[\s\S]*\]", cleaned)
        if m:
            data = _try_load_json(m.group(0))

    if data is None:
        salvaged = _salvage_json_objects(cleaned)
        if salvaged:
            print(f"    JSON salvage: recovered {len(salvaged)} object(s) from malformed output")
        return [x for obj in salvaged if (x := _normalize_chunk_item(obj))]

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []

    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        norm = _normalize_chunk_item(item)
        if norm:
            out.append(norm)
    return out


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_attn_implementation(model_id: str, device: str, requested: str) -> str:
    if requested != "auto":
        return requested
    # Phi-3 on MPS often needs eager; Qwen is fine with SDPA on recent torch
    if "phi" in model_id.lower() or device == "cpu":
        return "eager"
    return "sdpa"


class LocalChunker:
    def __init__(
        self,
        model_id: str,
        device: str = "mps",
        max_new_tokens: int = 2048,
        attn_implementation: str = "auto",
    ):
        attn = pick_attn_implementation(model_id, device, attn_implementation)
        print(f"Loading {model_id} on {device} (attn={attn}; first run downloads weights)...")
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        load_kwargs: dict = {
            "dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, attn_implementation=attn, **load_kwargs
            )
        except Exception as e:
            if attn != "eager":
                print(f"  attn {attn} failed ({e}); retrying with eager")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, attn_implementation="eager", **load_kwargs
                )
            else:
                raise
        if device != "cpu":
            self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = True
        print("Model ready.")

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _generate_messages(self, messages: list[dict], max_new_tokens: int) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self._model_device()) for k, v in inputs.items()}
        with torch.no_grad():
            if self.device == "mps":
                # MPS does not support all ops in float16; autocast keeps matmul on GPU
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            else:
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def extract_chunks(self, passage: str, *, return_raw: bool = False) -> tuple[list[dict], str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"SOURCE PASSAGE:\n\n{passage[:PASSAGE_INPUT_CHARS]}",
            },
        ]
        raw = self._generate_messages(messages, self.max_new_tokens)
        return parse_json_array(raw), raw

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
                    f"{raw[:12_000]}"
                ),
            },
        ]
        return self._generate_messages(messages, 800)


def postprocess(items: list[dict], *, min_tokens: int = MIN_CHUNK_TOKENS) -> list[dict]:
    out: list[dict] = []
    dropped_short = 0
    for it in items:
        content = it["content"]
        tok = count_tokens(content)
        if tok < min_tokens:
            dropped_short += 1
            continue
        if tok > MAX_CHUNK_TOKENS:
            words = content.split()
            content = " ".join(words[: int(MAX_CHUNK_TOKENS * 0.75)])
        topics = it.get("topics") or infer_topics(content)
        out.append({"content": content.strip(), "topics": topics})
    if items and not out and dropped_short:
        print(
            f"    warning: model returned {len(items)} chunks but all were "
            f"under {min_tokens} tokens (shortest ~"
            f"{min(count_tokens(i['content']) for i in items)} tokens)"
        )
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


def estimate_passage_calls(sources: list[dict], passage_chars: int) -> int:
    return sum(
        max(1, len(split_passages(s["text"], passage_chars))) for s in sources
    )


def process_source(
    chunker: LocalChunker,
    src: dict,
    *,
    passage_chars: int,
    two_pass: bool,
    min_chunk_tokens: int,
    verbose: bool,
) -> list[dict]:
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

    passages = split_passages(text, passage_chars)
    all_items: list[dict] = []
    for pi, passage in enumerate(passages, 1):
        try:
            items, raw = chunker.extract_chunks(passage)
            if verbose and not items:
                print(f"    passage {pi} raw (no parse): {raw[:500]!r}", flush=True)
            elif not items and raw.strip():
                print(
                    f"    passage {pi}: model output could not be parsed "
                    f"({len(raw)} chars); retry with --verbose to inspect",
                    flush=True,
                )
            kept = postprocess(items, min_tokens=min_chunk_tokens)
            if len(passages) > 1 or verbose or len(items) != len(kept):
                print(
                    f"    passage {pi}: parsed {len(items)}, kept {len(kept)}",
                    flush=True,
                )
            all_items.extend(kept)
        except Exception as e:
            print(f"    chunk passage failed: {e}")
    return all_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Local HF chunking for nutrition RAG")
    parser.add_argument("--model", default=MODEL_DEFAULT, help="HuggingFace model id")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="auto prefers CUDA, then Apple MPS, then CPU",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max sources (0=all)")
    parser.add_argument(
        "--two-pass",
        action="store_true",
        help="Clean each batch first, then chunk (~2x slower; skip unless quality suffers)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Generation cap per passage (raise for very dense sources)",
    )
    parser.add_argument(
        "--attn",
        default="auto",
        choices=["auto", "sdpa", "eager"],
        help="Attention backend (auto: sdpa on Qwen+MPS, eager on Phi)",
    )
    parser.add_argument(
        "--passage-chars",
        type=int,
        default=PASSAGE_CHARS,
        help="Max chars per passage split (larger = fewer calls, more context)",
    )
    parser.add_argument(
        "--min-chunk-tokens",
        type=int,
        default=MIN_CHUNK_TOKENS,
        help="Drop chunks shorter than this (local models often emit ~20–40 tokens)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log raw model output when JSON parses to zero chunks",
    )
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

    passage_chars = args.passage_chars

    device = resolve_device(args.device)
    if device == "mps":
        # Avoid MPS OOM on 16 GB when swapping between large tensors
        import os

        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    PER_SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    chunker = LocalChunker(
        args.model,
        device=device,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=args.attn,
    )
    sources = load_sources()
    if args.limit:
        sources = sources[: args.limit]

    passage_calls = estimate_passage_calls(sources, passage_chars)
    if args.two_pass:
        batch_calls = sum(len(split_batches(s["text"])) for s in sources)
        est_sec = passage_calls * 8 + batch_calls * 5
    else:
        est_sec = passage_calls * 8
    est_min = est_sec / 60

    print(f"\nProcessing {len(sources)} sources")
    print(f"Model: {args.model} | device: {device}")
    print(f"Passage window: {passage_chars:,} chars | max_new_tokens: {args.max_new_tokens}")
    print(f"Mode: {'clean + chunk' if args.two_pass else 'chunk only (faster)'}")
    print(f"~{passage_calls} passage LLM calls (rough ETA {est_min:.0f}–{est_min * 1.8:.0f} min on M3 MPS)\n")

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
        extracted = process_source(
            chunker,
            src,
            passage_chars=passage_chars,
            two_pass=args.two_pass,
            min_chunk_tokens=args.min_chunk_tokens,
            verbose=args.verbose,
        )

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

        cache_path = _per_source_path(src["id"])
        if file_chunks:
            cache_path.write_text(json.dumps(file_chunks, indent=2), encoding="utf-8")
        elif cache_path.exists():
            cache_path.unlink()
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
