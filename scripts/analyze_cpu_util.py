"""Report effective busy cores from an mpstat -P ALL log."""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


MPSTAT_LINE = re.compile(r"^(\d\d:\d\d:\d\d (?:AM|PM))\s+(\S+)\s+(\S+)")


@dataclass
class CpuSample:
    ts: datetime
    per_cpu: dict[str, float] = field(default_factory=dict)  # cpu_label -> %usr
    all_usr: float = 0.0


def parse_mpstat(path: Path) -> list[CpuSample]:
    samples: list[CpuSample] = []
    current: CpuSample | None = None
    for line in path.read_text().splitlines():
        m = MPSTAT_LINE.match(line)
        if not m:
            continue
        ts_str, cpu_label, usr_str = m.group(1), m.group(2), m.group(3)
        # mpstat's header rows have "CPU" as cpu_label; skip them
        if cpu_label == "CPU":
            continue
        try:
            usr = float(usr_str)
        except ValueError:
            continue
        ts = datetime.strptime(ts_str, "%I:%M:%S %p")
        if current is None or current.ts != ts:
            if current is not None:
                samples.append(current)
            current = CpuSample(ts=ts)
        if cpu_label == "all":
            current.all_usr = usr
        else:
            current.per_cpu[cpu_label] = usr
    if current is not None:
        samples.append(current)
    return samples


def summarize_window(samples: list[CpuSample]) -> dict[str, float]:
    if not samples:
        return {}
    # Effective busy cores = sum of per-core %usr / 100
    busy_cores = [sum(s.per_cpu.values()) / 100.0 for s in samples]
    peak_per_core = [max(s.per_cpu.values()) if s.per_cpu else 0.0 for s in samples]
    return {
        "n": len(samples),
        "mean_busy_cores": sum(busy_cores) / len(busy_cores),
        "peak_busy_cores": max(busy_cores),
        "mean_peak_per_core": sum(peak_per_core) / len(peak_per_core),
        "p99_per_core": max(peak_per_core),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", required=True, help="mpstat log")
    ap.add_argument("--bench", help="bench stdout log (informational)")
    args = ap.parse_args()

    samples = parse_mpstat(Path(args.cpu))
    print(f"[analyze] {len(samples)} mpstat samples")
    if not samples:
        return 0
    print(f"[analyze] span: {samples[0].ts.strftime('%H:%M:%S')} "
          f"-> {samples[-1].ts.strftime('%H:%M:%S')}")
    print(f"[analyze] sample N cores: {len(samples[0].per_cpu)}")

    overall = summarize_window(samples)
    print(f"\noverall  n={overall['n']:>4}  mean_busy={overall['mean_busy_cores']:>5.1f} cores  "
          f"peak={overall['peak_busy_cores']:>5.1f}  mean_peak/core={overall['mean_peak_per_core']:>5.1f}%")

    active = [s for s in samples if s.per_cpu and max(s.per_cpu.values()) >= 50.0]
    print(f"\nactive  n={len(active):>4}  (>=50% on any one core)")
    if active:
        a = summarize_window(active)
        print(f"  mean_busy={a['mean_busy_cores']:>5.1f} cores  "
              f"peak={a['peak_busy_cores']:>5.1f}  mean_peak/core={a['mean_peak_per_core']:>5.1f}%")

    # Top 5 windows by busy_cores
    busy_sorted = sorted(samples, key=lambda s: -sum(s.per_cpu.values()))
    print("\ntop 10 busy windows:")
    for s in busy_sorted[:10]:
        b = sum(s.per_cpu.values()) / 100.0
        pk = max(s.per_cpu.values()) if s.per_cpu else 0
        n_above_50 = sum(1 for v in s.per_cpu.values() if v >= 50.0)
        print(f"  {s.ts.strftime('%H:%M:%S')}  busy_cores={b:>5.1f}  "
              f"peak_core={pk:>5.1f}%  cores>=50%={n_above_50:>2}/{len(s.per_cpu)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
