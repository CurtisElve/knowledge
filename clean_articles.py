"""
Strip junk lines from Jina-scraped article markdown.

Removes lines matching noise patterns (URLs, images, headers, CTAs, etc.)
then collapses excess blank lines.

Usage:
  python scripts/clean_articles.py
  python scripts/clean_articles.py --dry-run
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = ROOT / "raw" / "articles"

# --- line must drop if any of these match (case-insensitive) ---

IMAGE_EXT = re.compile(r"\.(jpe?g|png|gif|webp|svg|ico)\b", re.I)
HAS_URL = re.compile(r"https?://|www\.\w+\.", re.I)

# Nav / boilerplate / marketing
BUZZWORDS = re.compile(
    r"\b("
    r"subscribe|unsubscribe|newsletter|sign\s*up|join\s+my|follow\s+us|"
    r"share\s+this|cookie|cookieyes|opt[- ]?out|do\s+not\s+sell|"
    r"personal\s+information|powered\s+by|skip\s+to\s+content|"
    r"save\s+my\s+preferences|revisit\s+consent|banner\s+closes|"
    r"thanks\s+for\s+signing|welcome\s+to\s+our\s+community|something\s+went\s+wrong|"
    r"submitting\s+the\s+form|click\s+here|read\s+more|learn\s+more|"
    r"shop\s+now|buy\s+now|limited\s+time|discount\s+code|promo\s+code|"
    r"affiliate\s+link|sponsored\s+by|advertisement|all\s+rights\s+reserved|"
    r"privacy\s+policy|terms\s+of\s+(service|use)|contact\s+us|"
    r"instagram|facebook|twitter|tiktok|youtube\.com|linkedin|"
    r"add\s+to\s+cart|checkout|free\s+shipping|use\s+code|"
    r"psmd\s+newsletter|weekly\s+newsletter|premium\s+articles|topic\s+guides|"
    r"browse\s*\)|\[browse\]|sign\s+up\s+for"
    r")\b",
    re.I,
)

# Jina / scrape metadata prefixes
META_PREFIX = re.compile(
    r"^(title|url\s+source|published\s+time|markdown\s+content)\s*:",
    re.I,
)

# Mostly markdown chrome, no real prose
ONLY_MARKDOWN = re.compile(
    r"^[\s\[\]\(\)!#*_\-|>`~•·‍]+$"
)

# Empty or image-only markdown links
EMPTY_LINK_LINE = re.compile(r"^\[\]\(https?://")
IMAGE_ONLY_LINE = re.compile(r"^!\[")

# Checkbox / form UI
FORM_UI = re.compile(r"^-\s*\[[ x]\]|^cancel\s+save|^your\s+opt-?out", re.I)

# Date-only or section labels (short caps)
ONLY_LABEL = re.compile(r"^[A-Z][A-Z0-9\s/&\-]{2,40}$")


def word_count(line: str) -> int:
    """Count words after stripping inline markdown."""
    text = line
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_#>`|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return 0
    return len(text.split())


def should_drop(line: str, min_words: int) -> bool:
    raw = line.rstrip("\n")
    stripped = raw.strip()

    if not stripped:
        return False  # keep blank lines for now; collapse later

    if stripped.startswith("#"):
        return True

    if HAS_URL.search(stripped):
        return True

    if IMAGE_EXT.search(stripped):
        return True

    if META_PREFIX.match(stripped):
        return True

    if BUZZWORDS.search(stripped):
        return True

    if EMPTY_LINK_LINE.match(stripped) or IMAGE_ONLY_LINE.match(stripped):
        return True

    if FORM_UI.match(stripped):
        return True

    if ONLY_MARKDOWN.match(stripped):
        return True

    if ONLY_LABEL.match(stripped) and word_count(stripped) < min_words:
        return True

    # Lines that are mostly URL fragments without http (rare)
    if re.fullmatch(r"[\w\-\./%#&?=]+", stripped) and "." in stripped and "/" in stripped:
        return True

    if word_count(stripped) < min_words:
        return True

    return False


def clean_text(text: str, min_words: int) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        if should_drop(line, min_words):
            continue
        kept.append(line.rstrip())

    # Collapse 3+ blank lines to 1
    out: list[str] = []
    blank_run = 0
    for line in kept:
        if not line.strip():
            blank_run += 1
            if blank_run <= 1:
                out.append("")
        else:
            blank_run = 0
            out.append(line)

    result = "\n".join(out).strip()
    return result + "\n" if result else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(ARTICLES_DIR), help="Articles directory")
    parser.add_argument("--min-words", type=int, default=5, help="Minimum words per line")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = Path(args.dir)
    files = sorted(base.glob("*.md"))
    changed = 0

    for path in files:
        original = path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_text(original, args.min_words)
        if cleaned == original:
            continue
        changed += 1
        before_lines = len(original.splitlines())
        after_lines = len(cleaned.splitlines())
        rel = path.relative_to(ROOT)
        print(f"{'would clean' if args.dry_run else 'cleaned'}: {rel} ({before_lines} -> {after_lines} lines)")
        if not args.dry_run:
            path.write_text(cleaned, encoding="utf-8")

    print(f"\n{changed}/{len(files)} files {'would be ' if args.dry_run else ''}updated")


if __name__ == "__main__":
    main()
