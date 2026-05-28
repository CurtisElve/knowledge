"""Run full RAG knowledge collection pipeline."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def run(script: str) -> int:
    print(f"\n{'='*60}\n>>> {script}\n{'='*60}")
    return subprocess.call([sys.executable, str(SCRIPTS / script)], cwd=ROOT)


def main() -> None:
    steps = ["collect_youtube.py", "collect_articles.py"]
    for s in steps:
        code = run(s)
        if code != 0:
            print(f"Warning: {s} exited with {code}")

    print("\n" + "=" * 60)
    print("Collection done. Run chunking when you have an API key:")
    print("  set GEMINI_API_KEY=your_key")
    print("  python scripts/build_chunks.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
