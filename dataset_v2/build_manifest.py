"""Build the VRGestures v2 manifest from a recordings tree and the author's check file.

Rules (author's decisions): check value false -> excluded (flagged_false); not in the check file -> excluded
(unchecked) until verified; check-file entries with no file on disk are listed (not_found_on_disk); files
directly in the root are part of the dataset (session_id "(root)"); folders listed in --exclude-dirs -> excluded_folder.
Check-file keys are bare file names; they are matched against the file's base name.

    python -m dataset_v2.build_manifest --recordings-root <.../Gestures> --check-file <check.json>
Environment fallbacks: VRG_V2_RECORDINGS_ROOT, VRG_V2_CHECK_FILE.
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from .classes import CLASS_ID, CLASSES, hand_of, normalise

FIELDS = ["relative_path", "session_id", "class_name_canonical", "class_id", "hand", "raw_class_token",
          "checked", "check_value", "included", "exclusion_reason", "class_in_vocabulary"]
ROOT_SESSION = "(root)"
REASONS = ["flagged_false", "unchecked", "not_found_on_disk", "excluded_folder"]
DEFAULT_EXCLUDE_DIRS = ["AggelosTest", "IoannaTest"]  # named after individuals; consent unknown


def raw_token(filename):
    """Leading name token of '<Class> <n> <hash> <DD> <Month> <HH> <MM> <k>.json' (whole name if no spaces)."""
    return filename.split(" ")[0]


def scan(root, exclude_dirs):
    """Yield (relative posix path, session_id, in_excluded_folder) for every .json under root."""
    root = Path(root)
    excl = {d.lower() for d in exclude_dirs}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        rel_dir = Path(dirpath).relative_to(root)
        parts = rel_dir.parts
        for f in sorted(filenames):
            if not f.lower().endswith(".json"):
                continue
            rel = (rel_dir / f).as_posix()
            if not parts:
                yield rel, ROOT_SESSION, False  # loose file in the root: included, grouped as one pseudo-session
            else:
                yield rel, parts[0], any(p.lower() in excl for p in parts)


def _row(rel, session, tok, checked, val, reason):
    canon, _ = normalise(tok)
    name = canon or tok.removesuffix(".json")  # out-of-vocabulary names (LeftOK, RightOK) keep their own name
    return {"relative_path": rel, "session_id": session, "class_name_canonical": name,
            "class_id": CLASS_ID.get(canon, -1), "hand": hand_of(name), "class_in_vocabulary": canon is not None,
            "raw_class_token": tok, "checked": checked, "check_value": "" if val is None else bool(val),
            "included": reason == "", "exclusion_reason": reason}


def build(root, check, exclude_dirs=DEFAULT_EXCLUDE_DIRS):
    rows, anomalies = [], []
    by_name = defaultdict(list)
    for rel, session, ex in scan(root, exclude_dirs):
        fname = rel.rsplit("/", 1)[-1]
        by_name[fname].append(rel)
        checked, val = fname in check, check.get(fname)
        if ex:
            reason = "excluded_folder"
        elif not checked:
            reason = "unchecked"
        elif val is not True:
            reason = "flagged_false"
        else:
            reason = ""
        rows.append(_row(rel, session, raw_token(fname), checked, val, reason))
    for name, paths in by_name.items():
        if len(paths) > 1:
            anomalies.append({"type": "duplicate_basename", "name": name, "paths": paths})
    for name, val in check.items():
        if name not in by_name:
            rows.append(_row("", "", raw_token(name), True, val, "not_found_on_disk"))
    tokens = Counter(r["raw_class_token"] for r in rows)
    for tok, n in sorted(tokens.items()):
        canon, exact = normalise(tok)
        if canon is None:
            anomalies.append({"type": "unmapped_class_token", "token": tok, "count": n})
        elif not exact:
            anomalies.append({"type": "variant_class_token", "token": tok, "mapped_to": canon, "count": n})
    return rows, anomalies, tokens


def summarise(rows, anomalies, tokens, n_check):
    inc = [r for r in rows if r["included"]]
    on_disk = [r for r in rows if r["exclusion_reason"] != "not_found_on_disk"]
    sess = defaultdict(Counter)
    for r in on_disk:
        sess[r["session_id"]][r["exclusion_reason"] or "included"] += 1
    oov = [r["relative_path"] for r in inc if not r["class_in_vocabulary"]]
    per_class_hand = {c: {"included": 0, "flagged_false": 0, "unchecked": 0, "not_found_on_disk": 0,
                          "excluded_folder": 0} for c in CLASSES}
    for r in rows:
        if r["class_in_vocabulary"]:
            per_class_hand[r["class_name_canonical"]][r["exclusion_reason"] or "included"] += 1
    return {
        "n_rows": len(rows), "n_on_disk": len(on_disk), "n_check_entries": n_check, "n_included": len(inc),
        "n_included_in_vocabulary": len(inc) - len(oov), "included_outside_vocabulary": oov,
        "n_on_disk_in_check_file": sum(1 for r in on_disk if r["checked"]),
        "n_on_disk_not_in_check_file": sum(1 for r in on_disk if not r["checked"]),
        "exclusion_reasons": dict(Counter(r["exclusion_reason"] for r in rows if r["exclusion_reason"])),
        "n_distinct_raw_class_tokens": len(tokens),
        "n_canonical_classes_in_included": len({r["class_name_canonical"] for r in inc if r["class_in_vocabulary"]}),
        "included_per_class": {c: per_class_hand[c]["included"] for c in CLASSES},
        "per_class_all_outcomes": per_class_hand,
        "included_per_hand": dict(Counter(r["hand"] for r in inc)),
        "check_value_counts_on_disk": dict(Counter(str(r["check_value"]) for r in on_disk if r["checked"])),
        "n_sessions": len([k for k, v in sess.items() if k != ROOT_SESSION and sum(v.values()) > v["excluded_folder"]]),
        "sessions": {k: {**v, "total": sum(v.values())} for k, v in sorted(sess.items())},
        "anomalies": anomalies,
    }


def to_markdown(s):
    L = ["# VRGestures v2 manifest summary", "",
         f"- rows: {s['n_rows']}; on disk: {s['n_on_disk']}; check-file entries: {s['n_check_entries']}; "
         f"**included: {s['n_included']}** ({s['n_included_in_vocabulary']} in the 24-class vocabulary)",
         "- included in the manifest, outside the 24-class vocabulary, thumbs-up gestures per author statement (not verified): "
         + (", ".join(f"`{p}`" for p in s["included_outside_vocabulary"]) or "none"),
         f"- on disk and in check file: {s['n_on_disk_in_check_file']}; on disk, not in check file: "
         f"{s['n_on_disk_not_in_check_file']}; sessions: {s['n_sessions']}", "",
         "## Exclusion reasons", "", "| reason | count |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in s["exclusion_reasons"].items()]
    L += ["", "## Per class and hand", "",
          "| class | hand | included | flagged_false | unchecked | not_found_on_disk | excluded_folder |",
          "|---|---|---|---|---|---|---|"]
    for c, d in s["per_class_all_outcomes"].items():
        L.append(f"| {c} | {hand_of(c)} | {d['included']} | {d['flagged_false']} | {d['unchecked']} | "
                 f"{d['not_found_on_disk']} | {d['excluded_folder']} |")
    L += ["", "## Per session (files on disk)", "",
          "| session | included | flagged_false | unchecked | excluded_folder | total |", "|---|---|---|---|---|---|"]
    for k, v in s["sessions"].items():
        L.append(f"| {k} | {v.get('included', 0)} | {v.get('flagged_false', 0)} | {v.get('unchecked', 0)} | "
                 f"{v.get('excluded_folder', 0)} | {v['total']} |")
    L += ["", "## Class-name anomalies", ""]
    L += [f"- `{json.dumps(a, ensure_ascii=False)}`" for a in s["anomalies"]] or ["- none"]
    return "\n".join(L) + "\n"


def write_outputs(rows, summary, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                   encoding="utf-8")
    (out_dir / "manifest_summary.md").write_text(to_markdown(summary), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recordings-root", default=os.environ.get("VRG_V2_RECORDINGS_ROOT"))
    p.add_argument("--check-file", default=os.environ.get("VRG_V2_CHECK_FILE"))
    p.add_argument("--out-dir", default=str(Path(__file__).parent))
    p.add_argument("--exclude-dirs", nargs="*", default=DEFAULT_EXCLUDE_DIRS)
    a = p.parse_args(argv)
    if not a.recordings_root or not a.check_file:
        p.error("--recordings-root and --check-file (or VRG_V2_RECORDINGS_ROOT / VRG_V2_CHECK_FILE) are required")
    check = json.loads(Path(a.check_file).read_text(encoding="utf-8"))
    rows, anomalies, tokens = build(a.recordings_root, check, a.exclude_dirs)
    summary = summarise(rows, anomalies, tokens, len(check))
    write_outputs(rows, summary, a.out_dir)
    keys = ("n_rows", "n_on_disk", "n_included", "exclusion_reasons", "n_sessions", "n_distinct_raw_class_tokens")
    print(json.dumps({k: summary[k] for k in keys}, indent=1))


if __name__ == "__main__":
    main()
