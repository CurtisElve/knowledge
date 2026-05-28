"""
Normalize and force line breaks in raw text/markdown files.

- Collapses runs of whitespace (including transcript line-wrap artifacts)
- Inserts line breaks after sentence endings (. ! ?)
- Optionally word-wraps lines longer than --width (default: 100)

Usage:
  python scripts/wrap_text_files.py
  python scripts/wrap_text_files.py --dirs raw/youtube raw/articles
  python scripts/wrap_text_files.py --width 0   # sentence breaks only, no wrap
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Split after sentence punctuation; keep the punctuation on the line
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def collapse_whitespace(text: str) -> str:
    """Single spaces; preserve paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned = []
    for para in paragraphs:
        line = " ".join(para.split())
        if line:
            cleaned.append(line)
    return "\n\n".join(cleaned) if cleaned else ""


def break_sentences(paragraph: str) -> list[str]:
    parts = SENTENCE_END.split(paragraph.strip())
    return [p.strip() for p in parts if p.strip()]


def wrap_line(line: str, width: int) -> str:
    if width <= 0 or len(line) <= width:
        return line
    return "\n".join(textwrap.wrap(line, width=width, break_long_words=False))


def format_text(text: str, width: int) -> str:
    text = collapse_whitespace(text)
    if not text:
        return text

    out_paragraphs: list[str] = []
    for para in text.split("\n\n"):
        sentences = break_sentences(para)
        if not sentences:
            continue
        lines = [wrap_line(s, width) for s in sentences]
        out_paragraphs.append("\n".join(lines))

    return "\n\n".join(out_paragraphs) + "\n"


def process_file(path: Path, width: int, dry_run: bool) -> bool:
    original = path.read_text(encoding="utf-8", errors="replace")
    formatted = format_text(original, width)
    if formatted == original:
        return False
    if not dry_run:
        path.write_text(formatted, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Force line breaks in text files")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["raw/youtube", "raw/articles"],
        help="Directories to process (relative to project root)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=100,
        help="Max line width (0 = sentence breaks only)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = parser.parse_args()

    changed = 0
    scanned = 0
    for dir_rel in args.dirs:
        base = ROOT / dir_rel
        if not base.is_dir():
            print(f"skip missing dir: {base}")
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            if path.name == "manifest.json":
                continue
            scanned += 1
            if process_file(path, args.width, args.dry_run):
                changed += 1
                action = "would update" if args.dry_run else "updated"
                print(f"{action}: {path.relative_to(ROOT)}")

    print(f"\n{changed}/{scanned} files {'would be ' if args.dry_run else ''}reformatted")


if __name__ == "__main__":
    main()
