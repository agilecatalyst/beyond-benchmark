#!/usr/bin/env python3
"""Lightweight correlation sketch for Beyond Benchmark sample runs.

Reads datasets/sample_run.csv and reports:
  - mean latency by entropy_dip_flag
  - Pearson-style correlation between min_entropy and resonance_rating
  - Pearson-style correlation between latency_ms and resonance_rating

No external dependencies beyond the Python standard library.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "datasets" / "sample_run.csv"


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def load_rows(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(
                {
                    "run_id": raw["run_id"],
                    "latency_ms": float(raw["latency_ms"]),
                    "tokens_per_sec": float(raw["tokens_per_sec"]),
                    "avg_attention_entropy": float(raw["avg_attention_entropy"]),
                    "min_entropy": float(raw["min_entropy"]),
                    "entropy_dip_flag": int(raw["entropy_dip_flag"]),
                    "resonance_rating": float(raw["resonance_rating"]),
                }
            )
    return rows


def main() -> None:
    rows = load_rows(CSV_PATH)
    dipped = [r for r in rows if r["entropy_dip_flag"] == 1]
    plain = [r for r in rows if r["entropy_dip_flag"] == 0]

    print(f"Loaded {len(rows)} runs from {CSV_PATH.relative_to(ROOT)}")
    print()
    print("Mean latency (ms)")
    print(f"  entropy dip flagged: {statistics.mean(r['latency_ms'] for r in dipped):.1f} (n={len(dipped)})")
    print(f"  no dip:              {statistics.mean(r['latency_ms'] for r in plain):.1f} (n={len(plain)})")
    print()
    print("Mean resonance rating")
    print(f"  entropy dip flagged: {statistics.mean(r['resonance_rating'] for r in dipped):.2f}")
    print(f"  no dip:              {statistics.mean(r['resonance_rating'] for r in plain):.2f}")
    print()

    r_entropy = pearson(
        [float(r["min_entropy"]) for r in rows],
        [float(r["resonance_rating"]) for r in rows],
    )
    r_latency = pearson(
        [float(r["latency_ms"]) for r in rows],
        [float(r["resonance_rating"]) for r in rows],
    )
    print("Correlations with resonance_rating")
    print(f"  min_entropy: {r_entropy:.3f}" if r_entropy is not None else "  min_entropy: n/a")
    print(f"  latency_ms:  {r_latency:.3f}" if r_latency is not None else "  latency_ms: n/a")
    print()
    print("Note: toy sample only — interpret as a template, not a result.")


if __name__ == "__main__":
    main()
