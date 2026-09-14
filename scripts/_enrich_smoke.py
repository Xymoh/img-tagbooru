"""Quick smoke test for enrich_tags - 3 seed sets x 3 modes."""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.description_tagger import get_description_tagger

SEEDS = [
    ["1girl", "beach", "volleyball"],
    ["1girl", "witch_hat", "forest"],
    ["1girl", "1boy", "bedroom"],
]
MODES = ["safe", "creative", "mature"]


def main() -> None:
    tagger = get_description_tagger()
    if not tagger.check_connection():
        print("ERROR: Ollama not reachable")
        sys.exit(1)
    print(f"Model: {tagger.model}\n")

    for seeds in SEEDS:
        print(f"\n{'='*70}")
        print(f"SEEDS: {', '.join(seeds)}")
        print(f"{'='*70}")
        for mode in MODES:
            start = time.time()
            try:
                res = tagger.enrich_tags(seeds, creativity=mode)
                elapsed = time.time() - start
                seeds_set = set(seeds)
                new_tags = [t for t in res.tags if t not in seeds_set]
                print(f"\n[{mode.upper()}] {len(res.tags)} total ({len(new_tags)} new) in {elapsed:.1f}s")
                print(f"  All: {', '.join(res.tags)}")
            except Exception as e:
                elapsed = time.time() - start
                print(f"\n[{mode.upper()}] ERROR ({elapsed:.1f}s): {e}")


if __name__ == "__main__":
    main()
