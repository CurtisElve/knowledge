"""
Rule-based nutrition chunker (no API key required).
Merges paragraphs into ~400-800 token chunks and assigns topics via keyword matching.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

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

MIN_TOKENS = 400
MAX_TOKENS = 800

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "protein": ["protein", "amino acid", "leucine", "muscle protein"],
    "carbohydrates": ["carbohydrate", "carb", "glucose", "glycogen", "insulin"],
    "fats": ["fat", "lipid", "saturated", "monounsaturated", "triglyceride"],
    "omega-3": ["omega-3", "omega 3", "dha", "epa", "fish oil"],
    "seed oils": ["seed oil", "linoleic", "vegetable oil", "soybean oil", "canola"],
    "ketosis": ["ketosis", "ketone", "keto", "beta-hydroxybutyrate"],
    "fasting": ["fasting", "time-restricted", "intermittent fast"],
    "metabolic health": ["metabolic", "insulin resistance", "blood sugar", "hba1c"],
    "vitamins": ["vitamin", "micronutrient", "deficiency"],
    "minerals": ["magnesium", "zinc", "iron", "selenium", "electrolyte"],
    "gut health": ["gut", "microbiome", "intestinal", "fiber", "butyrate"],
    "supplements": ["supplement", "creatine", "berberine", "metformin", "nad"],
    "aging": ["longevity", "aging", "autophagy", "telomere", "senescence"],
    "exercise": ["exercise", "training", "recovery", "athlete"],
    "red meat": ["red meat", "beef", "carnivore", "organ meat"],
    "processed food": ["ultra-processed", "processed food", "additive"],
    "caloric restriction": ["calorie", "caloric restriction", "energy balance"],
    "hydration": ["hydration", "water", "electrolyte", "sodium"],
    "alcohol": ["alcohol", "ethanol", "drinking"],
    "caffeine": ["coffee", "caffeine"],
}

NUTRITION_SIGNAL = re.compile(
    r"\b(diet|nutrition|food|eat|calori|protein|carb|fat|vitamin|mineral|"
    r"metabol|insulin|glucose|supplement|fasting|keto|fiber|gut|meal|"
    r"nutrient|macro|micro|cholesterol|triglyceride|obesity|weight)\b",
    re.I,
)


def infer_topics(text: str) -> list[str]:
    lower = text.lower()
    hits = [topic for topic, kws in TOPIC_KEYWORDS.items() if any(k in lower for k in kws)]
    return hits[:5] if hits else ["nutrition"]


def is_nutrition_relevant(text: str) -> bool:
    return len(NUTRITION_SIGNAL.findall(text)) >= 3


def load_sources() -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for p in sorted(RAW_YT.glob("*.txt")):
        t = p.read_text(encoding="utf-8", errors="replace").strip()
        if len(t) > 200:
            items.append((f"youtube:{p.stem}", t))
    for p in sorted(RAW_ART.glob("*.md")):
        t = p.read_text(encoding="utf-8", errors="replace").strip()
        if len(t) > 200:
            items.append((f"article:{p.stem}", t))
    return items


def split_paragraphs(text: str) -> list[str]:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    paras = [p.strip() for p in re.split(r"\n{2,}|\n(?=[A-Z#])", text) if p.strip()]
    if not paras:
        paras = [text]
    # YouTube transcripts are often one block — split into sentences
    units: list[str] = []
    for para in paras:
        if count_tokens(para) > MAX_TOKENS:
            sents = re.split(r"(?<=[.!?])\s+", para)
            units.extend(s.strip() for s in sents if s.strip())
        else:
            units.append(para)
    return units


def merge_units(units: list[str]) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for unit in units:
        if not is_nutrition_relevant(unit) and buf_tokens < MIN_TOKENS:
            continue
        ut = count_tokens(unit)
        if ut > MAX_TOKENS:
            if buf:
                chunks.append(" ".join(buf))
                buf, buf_tokens = [], 0
            words = unit.split()
            step = max(80, len(words) * MIN_TOKENS // max(ut, 1))
            for i in range(0, len(words), step):
                part = " ".join(words[i : i + step])
                if count_tokens(part) >= MIN_TOKENS // 2:
                    chunks.append(part)
            continue

        if buf_tokens + ut > MAX_TOKENS and buf_tokens >= MIN_TOKENS:
            chunks.append(" ".join(buf))
            buf, buf_tokens = [unit], ut
        else:
            buf.append(unit)
            buf_tokens += ut

        if buf_tokens >= MAX_TOKENS:
            chunks.append(" ".join(buf))
            buf, buf_tokens = [], 0

    if buf:
        if buf_tokens >= MIN_TOKENS or (chunks and buf_tokens >= 150):
            if chunks and buf_tokens < MIN_TOKENS:
                chunks[-1] = chunks[-1] + " " + " ".join(buf)
            else:
                chunks.append(" ".join(buf))
    return chunks


def main() -> None:
    sources = load_sources()
    print(f"Loaded {len(sources)} sources")
    all_chunks: list[dict] = []

    for src_id, text in sources:
        units = split_paragraphs(text)
        merged = merge_units(units)
        for content in merged:
            tok = count_tokens(content)
            if tok < 200:
                continue
            all_chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "content": content,
                    "topics": infer_topics(content),
                    "embed": [],
                }
            )

    OUT_FILE.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_chunks)} chunks to {OUT_FILE}")


if __name__ == "__main__":
    main()
