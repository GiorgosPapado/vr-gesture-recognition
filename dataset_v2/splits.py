"""Session-grouped train/val/test splits for VRGestures v2.

All recordings of one session folder go to the same split. Sessions are only a PROXY for participants:
participant ids are not stored in the files, so one person may appear in several sessions (and thus in several
splits); results can therefore be optimistic with respect to unseen people.

Sessions are shuffled with a seed and assigned 70/15/15 by session count. Because most sessions contain only
a subset of the classes, ``tries`` consecutive seeds (seed, seed+1, ...) are scanned and the first-best split is
kept: fewest classes missing from any split, then closest recording fractions. The seed actually used is
recorded. Run: python -m dataset_v2.splits --manifest dataset_v2/manifest.csv --out dataset_v2/splits.csv
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")


def _assign(sessions, rng, fractions):
    order = list(sessions)
    rng.shuffle(order)
    n = len(order)
    n_val = max(1, round(n * fractions[1])) if n >= 3 else 0
    n_test = max(1, round(n * fractions[2])) if n >= 3 else 0
    n_train = n - n_val - n_test
    return {s: ("train" if i < n_train else "val" if i < n_train + n_val else "test") for i, s in enumerate(order)}


def make_splits(rows, seed=0, fractions=(0.7, 0.15, 0.15), tries=500):
    """rows: manifest dicts (included only are used). Returns (session->split, report dict)."""
    rows = [r for r in rows if str(r["included"]) == "True"]
    per_sess = defaultdict(Counter)
    for r in rows:
        per_sess[r["session_id"]][r["class_name_canonical"]] += 1
    sessions = sorted(per_sess)
    all_classes = {c for v in per_sess.values() for c in v}
    total = sum(sum(v.values()) for v in per_sess.values())
    best = None
    for t in range(max(1, tries)):
        rng = np.random.RandomState(seed + t)
        a = _assign(sessions, rng, fractions)
        counts = {s: Counter() for s in SPLITS}
        for sess, cc in per_sess.items():
            counts[a[sess]].update(cc)
        missing = sum(len(all_classes - set(c)) for c in counts.values())
        dev = sum(abs(sum(counts[s].values()) / total - f) for s, f in zip(SPLITS, fractions))
        if best is None or (missing, dev) < best[0]:
            best = ((missing, dev), a, counts, seed + t)
        if missing == 0 and dev < 0.05:
            break
    _, assign, counts, used = best
    report = {
        "seed_requested": seed, "seed_used": used, "fractions": list(fractions),
        "n_sessions": {s: sum(1 for v in assign.values() if v == s) for s in SPLITS},
        "n_recordings": {s: int(sum(counts[s].values())) for s in SPLITS},
        "class_counts": {s: dict(sorted(counts[s].items())) for s in SPLITS},
        "warnings": [f"split '{s}' has no recordings of class {c}" for s in SPLITS
                     for c in sorted(all_classes - set(counts[s]))],
        "note": "sessions are a proxy for participants (participant ids are not in the files)",
    }
    return assign, report


def write_splits(rows, assign, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relative_path", "session_id", "class_name_canonical", "class_id", "split"])
        for r in rows:
            if str(r["included"]) == "True":
                w.writerow([r["relative_path"], r["session_id"], r["class_name_canonical"], r["class_id"],
                            assign[r["session_id"]]])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=str(Path(__file__).parent / "manifest.csv"))
    p.add_argument("--out", default=str(Path(__file__).parent / "splits.csv"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tries", type=int, default=500)
    p.add_argument("--fractions", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    a = p.parse_args(argv)
    with open(a.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assign, report = make_splits(rows, a.seed, tuple(a.fractions), a.tries)
    write_splits(rows, assign, a.out)
    rep = Path(a.out).with_name(Path(a.out).stem + "_report.json")
    rep.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("seed_used", "n_sessions", "n_recordings", "warnings")}, indent=1))


if __name__ == "__main__":
    main()
