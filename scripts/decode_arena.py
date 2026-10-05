#!/usr/bin/env python3
"""Decode obfuscated arena battle results from .arena_data/*.bin files.

Usage:
    python scripts/decode_arena.py                    # decode all, print summary
    python scripts/decode_arena.py --full             # decode all, print full JSON
    python scripts/decode_arena.py --out results.json # decode all, save to file
"""
import argparse
import base64
import json
from pathlib import Path


def decode_all(data_dir: Path) -> list[dict]:
    results = []
    for f in sorted(data_dir.glob("*.bin"), key=lambda p: p.stat().st_mtime):
        try:
            raw = base64.b64decode(f.read_bytes())
            results.append(json.loads(raw))
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")
    return results


def print_summary(results: list[dict]):
    wins: dict[str, dict] = {}
    for r in results:
        for side in ("model_a", "model_b"):
            mid = r[side]["display"]
            if mid not in wins:
                wins[mid] = {"wins": 0, "losses": 0, "ties": 0, "total": 0}
            wins[mid]["total"] += 1

        if r["preference"] == "a":
            wins[r["model_a"]["display"]]["wins"] += 1
            wins[r["model_b"]["display"]]["losses"] += 1
        elif r["preference"] == "b":
            wins[r["model_b"]["display"]]["wins"] += 1
            wins[r["model_a"]["display"]]["losses"] += 1
        elif r["preference"] == "tie":
            wins[r["model_a"]["display"]]["ties"] += 1
            wins[r["model_b"]["display"]]["ties"] += 1

    print(f"\n  Arena Results: {len(results)} battles\n")
    print(f"  {'Model':<25} {'W':>4} {'L':>4} {'T':>4} {'Total':>6} {'Win%':>6}")
    print(f"  {'─'*25} {'─'*4} {'─'*4} {'─'*4} {'─'*6} {'─'*6}")
    for model in sorted(wins, key=lambda m: wins[m]["wins"], reverse=True):
        s = wins[model]
        wr = round(s["wins"] / max(s["total"], 1) * 100)
        print(f"  {model:<25} {s['wins']:>4} {s['losses']:>4} {s['ties']:>4} {s['total']:>6} {wr:>5}%")

    print(f"\n  Per-battle breakdown:\n")
    for i, r in enumerate(results, 1):
        winner = r.get("winner", "?")
        pref = r["preference"]
        label = {"a": f"A ({r['model_a']['display']})", "b": f"B ({r['model_b']['display']})",
                 "tie": "Tie", "both_bad": "Both bad"}[pref]
        print(f"  {i:3}. {r['created_at'][:16]}  A={r['model_a']['display']:<20} B={r['model_b']['display']:<20} -> {label}")


def main():
    parser = argparse.ArgumentParser(description="Decode arena results")
    parser.add_argument("--dir", default=".arena_data", help="Data directory")
    parser.add_argument("--full", action="store_true", help="Print full JSON")
    parser.add_argument("--out", help="Save decoded JSON to file")
    args = parser.parse_args()

    data_dir = Path(args.dir)
    if not data_dir.exists():
        print(f"  No data directory found at {data_dir}")
        return

    results = decode_all(data_dir)
    if not results:
        print("  No battles found.")
        return

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"  Saved {len(results)} battles to {args.out}")

    if args.full:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print_summary(results)


if __name__ == "__main__":
    main()
