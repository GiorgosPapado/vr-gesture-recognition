"""Statistics for the data card: file sizes (all on-disk files), filename date range, and frames / duration /
sampling rate / empty-hand pattern from a seeded random sample of included recordings.

    python -m dataset_v2.data_stats --manifest dataset_v2/manifest.csv --recordings-root <root> --out stats.json
"""
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .loader import read_manifest

NAME_RE = re.compile(r"^\S+ \d+ \S+ (\d+) (\w+) (\d+) (\d+) \d+\.json$")
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                      "September", "October", "November", "December"], 1)}


def describe(a):
    a = np.asarray(a, float)
    return {"min": float(a.min()), "p5": float(np.percentile(a, 5)), "median": float(np.median(a)),
            "mean": float(a.mean()), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def file_date_range(rel_paths):
    """Earliest / latest (month, day, HH:MM) found in the file names (no year is stored in the names)."""
    stamps = []
    for p in rel_paths:
        m = NAME_RE.match(p.rsplit("/", 1)[-1])
        if m and m[2] in MONTHS:
            stamps.append((MONTHS[m[2]], int(m[1]), int(m[3]), int(m[4]), m[2]))
    if not stamps:
        return None
    lo, hi = min(stamps), max(stamps)
    return {"first": f"{lo[4]} {lo[1]:02d} {lo[2]:02d}:{lo[3]:02d}", "last": f"{hi[4]} {hi[1]:02d} {hi[2]:02d}:{hi[3]:02d}",
            "months": dict(Counter(s[4] for s in stamps)), "note": "year is not encoded in the file names"}


def sample_stats(rows, root, n=300, seed=0):
    from .loader import read_recording

    inc = [r for r in rows if str(r["included"]) == "True" and str(r["class_in_vocabulary"]) == "True"]
    rng = np.random.RandomState(seed)
    pick = rng.choice(len(inc), size=min(n, len(inc)), replace=False)
    frames, dur, hz, empty = [], [], [], Counter()
    for i in pick:
        rec = read_recording(os.path.join(root, inc[i]["relative_path"]))
        T = len(rec["timestamps"])
        frames.append(T)
        span = float(rec["timestamps"][-1] - rec["timestamps"][0]) if T > 1 else 0.0
        dur.append(span)
        if span > 0:
            hz.append((T - 1) / span)
        pattern = ("L" if rec["left_pos"] is not None else "-") + ("R" if rec["right_pos"] is not None else "-")
        empty[f"{inc[i]['hand']}:{pattern}"] += 1
    return {"n_sampled": len(pick), "seed": seed, "frames": describe(frames), "duration_s": describe(dur),
            "sampling_rate_hz": describe(hz), "hands_present_by_class_hand": dict(sorted(empty.items()))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=str(Path(__file__).parent / "manifest.csv"))
    p.add_argument("--recordings-root", default=os.environ.get("VRG_V2_RECORDINGS_ROOT"))
    p.add_argument("--out", default=str(Path(__file__).parent / "data_stats.json"))
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if not a.recordings_root:
        p.error("--recordings-root or VRG_V2_RECORDINGS_ROOT required")
    rows = read_manifest(a.manifest)
    on_disk = [r for r in rows if r["relative_path"]]
    inc = [r for r in rows if r["included"] == "True" and r["class_in_vocabulary"] == "True"]
    sizes = [os.path.getsize(os.path.join(a.recordings_root, r["relative_path"])) for r in on_disk]
    inc_sizes = [os.path.getsize(os.path.join(a.recordings_root, r["relative_path"])) for r in inc]
    res = {"on_disk_files": len(on_disk), "on_disk_bytes": int(sum(sizes)), "file_size_bytes_on_disk": describe(sizes),
           "included_files": len(inc), "included_bytes": int(sum(inc_sizes)),
           "file_size_bytes_included": describe(inc_sizes),
           "filename_date_range_included": file_date_range([r["relative_path"] for r in inc]),
           "sample": sample_stats(rows, a.recordings_root, a.n, a.seed)}
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
