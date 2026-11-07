import csv
import json
from collections import Counter

import numpy as np
from dataset_v2 import baselines, build_manifest, loader, splits
from dataset_v2.classes import CLASSES, normalise


def _manifest(tree):
    root, _, check, _ = tree
    rows, anomalies, tokens = build_manifest.build(root, check)
    return rows, anomalies, tokens


def test_24_classes_and_normalisation():
    assert len(CLASSES) == 24 and len(set(CLASSES)) == 24
    assert normalise("LeftBack") == ("LeftBack", True)
    assert normalise("LeftBack.json") == ("LeftBack", False)
    assert normalise("rightopenpalm") == ("RightOpenPalm", False)
    assert normalise("Right_Push") == ("RightPush", False)
    assert normalise("LeftOK.json") == (None, False)


def test_manifest_rules(tree):
    rows, anomalies, tokens = _manifest(tree)
    reasons = Counter(r["exclusion_reason"] for r in rows)
    assert reasons["flagged_false"] == 1
    assert reasons["unchecked"] == 1
    assert reasons["not_found_on_disk"] == 1
    assert reasons["excluded_folder"] == 1  # AggelosTest only; loose root files are included
    inc = [r for r in rows if r["included"]]
    assert len(inc) == tree[3] - 2 + 3  # 3 loose root files (LeftClick, LeftOK, Zoom)
    oov = [r for r in inc if not r["class_in_vocabulary"]]
    assert [r["relative_path"] for r in oov] == ["LeftOK.json"]
    assert oov[0]["class_name_canonical"] == "LeftOK" and oov[0]["class_id"] == -1
    assert {r["session_id"] for r in inc if "/" not in r["relative_path"]} == {"(root)"}
    assert all(r["checked"] and r["check_value"] is True for r in inc)
    assert not any("AggelosTest" in r["relative_path"] for r in inc)
    missing = [r for r in rows if r["exclusion_reason"] == "not_found_on_disk"][0]
    assert missing["relative_path"] == "" and missing["class_name_canonical"] == "Zoom"
    assert {"type": "variant_class_token", "token": "LeftClick.json", "mapped_to": "LeftClick", "count": 1} in anomalies


def test_manifest_cli_outputs(tree, tmp_path):
    root, cf, _, n = tree
    build_manifest.main(["--recordings-root", str(root), "--check-file", str(cf), "--out-dir", str(tmp_path)])
    s = json.loads((tmp_path / "manifest_summary.json").read_text(encoding="utf-8"))
    assert s["n_included"] == n + 1 and s["n_included_in_vocabulary"] == n and s["n_sessions"] == 8
    assert (tmp_path / "manifest.csv").exists() and "included" in (tmp_path / "manifest_summary.md").read_text()


def test_loader_shapes_and_resample(tree):
    root = tree[0]
    f = next((root / "abc0001 October 11 10").glob("LeftClick*.json"))
    rec = loader.read_recording(f)
    assert rec["right_pos"] is None and rec["left_pos"].shape[1:] == (24, 3)
    s = loader.recording_to_series(rec)
    assert s.shape == (len(rec["timestamps"]), 150)
    assert np.all(s[:, 75:] == 0)  # right hand absent
    arr, nfr, name = loader.load_fixed(f, length=16)
    assert arr.shape == (16, 150) and arr.dtype == np.float32 and name == "LeftClick"
    assert loader.load_fixed(f, length=40, pad=True)[0].shape == (40, 150)
    lin = loader.resample_linear(np.arange(5.0)[:, None], 9)
    assert np.allclose(lin[:, 0], np.linspace(0, 4, 9))
    rot = loader.recording_to_series(rec, ("local_rot", "global_rot"))
    assert rot.shape[1] == 2 * (96 + 4)


def test_build_and_save_npz(tree, tmp_path):
    rows, _, _ = _manifest(tree)
    X, y, sess, sessions, paths, nfr, problems = loader.build_arrays(rows, tree[0], length=8, workers=2)
    assert X.shape == (tree[3] - 2 + 2, 8, 150) and len(sessions) == 9 and not problems
    assert len(y) == len(sess) == len(paths) == len(nfr)
    out = loader.save_npz(tmp_path / "d.npz", X, y, sess, sessions, CLASSES,
                          loader.DEFAULT_FEATURES, 8, False, True, paths)
    meta = json.loads(out.with_name("d.meta.json").read_text())
    assert meta["n"] == len(X) and meta["has_subject"] is False and meta["n_channels"] == 150
    d = np.load(out)
    assert set(d.files) == {"X", "y", "session"}


def test_splits_disjoint_and_seeded(tree, tmp_path):
    rows, _, _ = _manifest(tree)
    a1, rep = splits.make_splits(rows, seed=3)
    a2, _ = splits.make_splits(rows, seed=3)
    assert a1 == a2
    assert set(a1.values()) == {"train", "val", "test"} and len(a1) == 9
    splits.write_splits(rows, a1, tmp_path / "s.csv")

    with open(tmp_path / "s.csv", encoding="utf-8") as fh:
        got = list(csv.DictReader(fh))
    by_sess = {}
    for r in got:
        by_sess.setdefault(r["session_id"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_sess.values())
    assert sum(rep["n_recordings"].values()) == len(got)


def test_baselines_run(tree):
    rows, _, _ = _manifest(tree)
    X, y, sess, sessions, _, _, _ = loader.build_arrays(rows, tree[0], length=8, workers=1)
    assign, _ = splits.make_splits(rows, seed=0)
    split = np.asarray([assign[sessions[s]] for s in sess])
    res = baselines.run_baselines(X, y, split, feat_len=4)
    for m in ("logreg", "lda"):
        assert 0.0 <= res["models"][m]["test"]["macro_f1"] <= 1.0
    assert res["majority_class_chance"]["test_accuracy"] <= 1.0


def test_baselines_extras(tree):
    rows, _, _ = _manifest(tree)
    X, y, sess, sessions, _, _, _ = loader.build_arrays(rows, tree[0], length=8, workers=1)
    assign, _ = splits.make_splits(rows, seed=0)
    split = np.asarray([assign[sessions[s]] for s in sess])
    groups = np.asarray([sessions[s] for s in sess])
    res = baselines.run_baselines(X, y, split, feat_len=4, groups=groups, n_boot=20)
    assert res["bootstrap"]["n_sessions"] == len(set(groups[split == "test"]))
    lo, hi = res["bootstrap"]["models"]["lda"]["accuracy_ci95"]
    assert 0.0 <= lo <= hi <= 1.0
    assert 0.0 <= res["hand_presence_baseline"]["test"]["accuracy"] <= 1.0
    assert 0.0 <= res["models"]["logreg"]["within_hand"]["overall"] <= 1.0
    assert res["ablation_local_pos_only"]["n_features"] < res["n_features"]
