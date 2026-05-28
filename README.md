# Nutrition RAG Knowledge Base

Pipeline to collect nutrition content and produce `chunks.json` for vector embedding.

## Setup

```powershell
cd C:\Users\Curtisburtis\eatright\knowledge
.\Scripts\pip.exe install -r requirements.txt
```

## 1. Collect raw content

**YouTube** (`yt.txt` → `raw/youtube/*.txt`):

```powershell
.\Scripts\python.exe scripts\collect_youtube.py
```

**Articles** (`articles.txt` → `raw/articles/*.md` via Jina Reader):

```powershell
.\Scripts\python.exe scripts\collect_articles.py
```

Or both:

```powershell
.\Scripts\python.exe scripts\run_pipeline.py
```

Scripts skip already-downloaded files. Re-run to retry failures.

> **YouTube rate limits:** After many requests YouTube may block your IP. Wait 30–60 minutes and re-run `collect_youtube.py`. Only 38/113 succeeded on first pass due to blocking.

## 2. Build chunks

**Local (recommended for overnight runs)** — Phi-3.5-mini on CPU, 64 GB RAM:

```powershell
.\Scripts\python.exe scripts\build_chunks_local.py
# higher quality, slower:
.\Scripts\python.exe scripts\build_chunks_local.py --two-pass
# stronger model (slower):
.\Scripts\python.exe scripts\build_chunks_local.py --model Qwen/Qwen2.5-7B-Instruct
```

**Rule-based** (no model download):

```powershell
.\Scripts\python.exe scripts\chunk_heuristic.py
```

**Cloud LLM** (Gemini / Groq / OpenRouter):

```powershell
.\Scripts\python.exe scripts\build_chunks_openrouter.py
```

Output: `chunks.json`

```json
{
  "id": "uuid",
  "content": "Self-contained nutrition fact...",
  "topics": ["protein", "ketosis"],
  "embed": []
}
```

`embed` is left empty for a separate embedding step.

## Files

| Path | Purpose |
|------|---------|
| `yt.txt` | YouTube URLs |
| `articles.txt` | Article URLs |
| `raw/youtube/` | Transcript text |
| `raw/articles/` | Jina markdown |
| `chunks.json` | Final chunk database |
| `chunks_local_checkpoint.json` | Local chunking resume state |
| `chunks_local/by_source/` | Per-file chunk cache |
