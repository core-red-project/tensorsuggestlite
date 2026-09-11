#!/usr/bin/env python3
"""Standalone, Zero-TensorFlow inference CLI script.

Usage:
    python scripts/standalone_infer.py <model_dir> "Text to classify"
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path if running from source tree
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if src_dir.is_dir() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from tensorsuggestlite.lite_runtime import LiteInferenceRunner  # noqa: E402


def main() -> None:
    if len(sys.argv) < 3:
        print('Usage: python standalone_infer.py <model_dir> "Text to classify"')
        sys.exit(1)

    model_dir = sys.argv[1]
    text = sys.argv[2]

    try:
        runner = LiteInferenceRunner(model_dir)
        res = runner.predict(text)
        print(f"Top Category: {res.top_category} ({res.top_confidence * 100:.2f}%)")
        print(f"Latency:      {res.latency_ms:.2f} ms")
        print(f"Runtime:      {res.runtime}")
        print("Scores:")
        for score in res.scores:
            bar = "█" * int(score.confidence * 25)
            print(f"  {score.category:>12}: {score.confidence * 100:6.2f}% {bar}")
    except Exception as err:
        print(f"Inference error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
