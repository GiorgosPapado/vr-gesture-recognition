"""Simple baselines on the session-grouped split of VRGestures v2 (numpy + scikit-learn only).

Features: the stored (T, C) series is subsampled to ``--feat-len`` steps (linear index grid), flattened and
standardised (statistics from train only). Models: multinomial logistic regression and shrinkage LDA. For each
model the hyper-parameter is chosen on val by macro-F1 only, then the chosen model is scored on test ONCE.
Reported: accuracy, macro-F1, per-class recall, majority-class chance, test metrics with and without the loose
``(root)`` pseudo-session, a session-cluster bootstrap CI on test, within-hand accuracy, a hand-presence-only
baseline and a local-joint-position-only ablation.

    python -m dataset_v2.baselines --npz <dir>/vrgestures_v2.npz --splits dataset_v2/splits.csv
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

GRIDS = {
    "logreg": [("C", c) for c in (0.001, 0.01, 0.1, 1.0, 10.0)],
    "lda": [("shrinkage", v) for v in (0.001, 0.01, 0.1, 0.5, 0.9)],
}
MAX_ITER = 1000
ROOT_SESSION = "(root)"
N_BOOT = 1000


def flatten_features(X, feat_len):
    T = X.shape[1]
    idx = np.unique(np.round(np.linspace(0, T - 1, min(feat_len, T))).astype(int))
    return X[:, idx, :].reshape(len(X), -1).astype(np.float64)


def make_model(name, param, seed):
    _, v = param
    if name == "logreg":
        return LogisticRegression(C=v, max_iter=MAX_ITER, random_state=seed)
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=v)


def metrics(y_true, y_pred, labels):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "per_class_recall": {int(lab): float(r) for lab, r in zip(
                labels, recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0))}}


def hand_group(class_ids):
    """0 = left (ids 0-10), 1 = right (ids 11-21), 2 = both (22, 23)."""
    c = np.asarray(class_ids)
    return np.where(c <= 10, 0, np.where(c <= 21, 1, 2))


def hand_presence(X):
    """Which hand is present, from the data only. The absent hand is zero-filled; channels are laid out as
    [left 72 local + 3 root | right 72 local + 3 root]."""
    h = X.shape[2] // 2
    return (np.abs(X[:, :, :h]).max(axis=(1, 2)) > 0), (np.abs(X[:, :, h:]).max(axis=(1, 2)) > 0)


def local_pos_channels(n_channels):
    """Boolean mask of the local joint-position channels (drops the 3 root channels of each hand)."""
    h = n_channels // 2
    m = np.zeros(n_channels, dtype=bool)
    m[:h - 3] = True
    m[h:n_channels - 3] = True
    return m


def cluster_bootstrap(y, preds, groups, labels, n_boot=N_BOOT, seed=0):
    """Resample test sessions with replacement (same resamples for every model -> paired differences)."""
    sess = np.unique(groups)
    idx = {s: np.flatnonzero(groups == s) for s in sess}
    rng = np.random.RandomState(seed)
    acc = {m: [] for m in preds}
    f1 = {m: [] for m in preds}
    for _ in range(n_boot):
        pick = np.concatenate([idx[s] for s in rng.choice(sess, size=len(sess), replace=True)])
        for m, p in preds.items():
            acc[m].append(accuracy_score(y[pick], p[pick]))
            f1[m].append(f1_score(y[pick], p[pick], labels=labels, average="macro", zero_division=0))

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    out = {"n_boot": n_boot, "n_sessions": int(len(sess)), "seed": seed,
           "models": {m: {"accuracy_ci95": ci(acc[m]), "macro_f1_ci95": ci(f1[m])} for m in preds}}
    if "logreg" in preds and "lda" in preds:
        da = np.asarray(acc["logreg"]) - np.asarray(acc["lda"])
        df = np.asarray(f1["logreg"]) - np.asarray(f1["lda"])
        out["logreg_minus_lda"] = {"accuracy_ci95": ci(da), "macro_f1_ci95": ci(df),
                                   "accuracy_differs": bool(ci(da)[0] > 0 or ci(da)[1] < 0),
                                   "macro_f1_differs": bool(ci(df)[0] > 0 or ci(df)[1] < 0)}
    return out


def within_hand_accuracy(model, Fte, y):
    """Accuracy when the prediction is restricted to the classes of the true hand group (11-way for left/right
    recordings, 2-way for Teleportation/Zoom). Removes the left-vs-right decision."""
    proba = model.predict_proba(Fte)
    cls = np.asarray(model.classes_)
    grp_of_class = hand_group(cls)
    g = hand_group(y)
    pred = np.empty(len(y), dtype=int)
    for i in range(len(y)):
        cols = np.flatnonzero(grp_of_class == g[i])
        pred[i] = cls[cols[np.argmax(proba[i, cols])]]
    out = {"overall": float((pred == y).mean())}
    for name, k in (("left", 0), ("right", 1), ("both", 2)):
        out[name] = {"n": int((g == k).sum()), "accuracy": float((pred[g == k] == y[g == k]).mean())}
    full = cls[np.argmax(proba, axis=1)]
    out["hand_group_accuracy_of_unrestricted_prediction"] = float((hand_group(full) == g).mean())
    return out


def _run_models(F, y, tr, va, te, groups, labels, seed, models):
    sc = StandardScaler().fit(F[tr])
    Ftr, Fva, Fte = sc.transform(F[tr]), sc.transform(F[va]), sc.transform(F[te])
    out, test_pred = {}, {}
    nonroot = (groups[te] != ROOT_SESSION) if groups is not None else None
    for name in models:
        tried, best = [], None
        for param in GRIDS[name]:
            t0 = time.time()
            m = make_model(name, param, seed).fit(Ftr, y[tr])
            print(f"fit {name} {param} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
            f1 = f1_score(y[va], m.predict(Fva), labels=labels, average="macro", zero_division=0)
            tried.append({"param": {param[0]: param[1]}, "val_macro_f1": float(f1)})
            if best is None or f1 > best[0]:
                best = (f1, param, m)
        _, param, m = best
        pred = m.predict(Fte)  # test touched once per model; all test-side numbers below reuse this model
        test_pred[name] = pred
        grid_vals = [p[1] for p in GRIDS[name]]
        entry = {"selected": {param[0]: param[1]}, "val_search": tried,
                 "selected_on_grid_edge": bool(param[1] in (min(grid_vals), max(grid_vals))),
                 "max_iter": MAX_ITER if name == "logreg" else None,
                 "n_iter_reached": int(np.max(m.n_iter_)) if name == "logreg" else None,
                 "val": metrics(y[va], m.predict(Fva), labels), "test": metrics(y[te], pred, labels),
                 "within_hand": within_hand_accuracy(m, Fte, y[te])}
        if nonroot is not None and (~nonroot).any() and nonroot.any():
            entry["test_excl_root"] = metrics(y[te][nonroot], pred[nonroot], labels)
            entry["test_root_only"] = {"accuracy": float(accuracy_score(y[te][~nonroot], pred[~nonroot])),
                                       "n": int((~nonroot).sum())}
        out[name] = entry
    return out, test_pred


def run_baselines(X, y, split, feat_len=16, seed=0, models=("logreg", "lda"), groups=None, n_boot=N_BOOT,
                  ablation=True):
    """``groups`` = per-recording session id (enables the (root) split, the cluster bootstrap)."""
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    if not (tr.any() and va.any() and te.any()):
        raise ValueError("train/val/test must all be non-empty")
    groups = None if groups is None else np.asarray(groups)
    F = flatten_features(X, feat_len)
    labels = sorted(int(v) for v in np.unique(y))
    maj = np.bincount(y[tr]).argmax()
    out = {"n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}, "n_features": int(F.shape[1]),
           "feat_len": feat_len, "labels": labels, "seed": seed,
           "grids": {k: [p[1] for p in v] for k, v in GRIDS.items()},
           "majority_class_chance": {"class_id": int(maj),
                                     "val_accuracy": float((y[va] == maj).mean()),
                                     "test_accuracy": float((y[te] == maj).mean()),
                                     "uniform_chance": 1.0 / len(labels)}}
    out["models"], test_pred = _run_models(F, y, tr, va, te, groups, labels, seed, models)
    # hand-presence-only baseline: cell = (left present, right present); predict the train-majority class of the cell
    lp, rp = hand_presence(X)
    key = lp.astype(int) * 2 + rp.astype(int)
    pred = np.empty(int(te.sum()), dtype=int)
    for i, k in enumerate(key[te]):
        pool = y[tr][key[tr] == k]
        pred[i] = np.bincount(pool).argmax() if len(pool) else maj
    out["hand_presence_baseline"] = {
        "description": "features = (left hand present, right hand present); predict the train-majority class of that cell",
        "presence_agrees_with_class_hand": float((hand_group(y) == np.where(lp & rp, 2, np.where(lp, 0, 1))).mean()),
        "test": metrics(y[te], pred, labels)}
    if groups is not None:
        sess = groups[te]
        out["bootstrap"] = cluster_bootstrap(y[te], test_pred, sess, labels, n_boot, seed)
        if (sess == ROOT_SESSION).any():
            nr = sess != ROOT_SESSION
            out["bootstrap_excl_root"] = cluster_bootstrap(
                y[te][nr], {k: v[nr] for k, v in test_pred.items()}, sess[nr], labels, n_boot, seed)
    if ablation:
        ch = local_pos_channels(X.shape[2])
        mx = np.abs(X[:, :, ch]).max(axis=(0, 1))
        out["local_pos_stats"] = {"n_channels": int(ch.sum()), "n_channels_max_abs_above_1e-6": int((mx > 1e-6).sum()),
                                  "median_channel_max_abs": float(np.median(mx)), "max_abs": float(mx.max())}
        Fl = flatten_features(X[:, :, ch], feat_len)
        abl, _ = _run_models(Fl, y, tr, va, te, groups, labels, seed, models)
        out["ablation_local_pos_only"] = {"n_features": int(Fl.shape[1]), "models": abl}
    return out


def load_split_vector(npz_paths_file, splits_csv):
    sp = {}
    with open(splits_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sp[r["relative_path"]] = r["split"]
    paths = Path(npz_paths_file).read_text(encoding="utf-8").splitlines()
    return np.asarray([sp[p] for p in paths])


def load_session_vector(npz_paths_file, splits_csv):
    sp = {}
    with open(splits_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sp[r["relative_path"]] = r["session_id"]
    paths = Path(npz_paths_file).read_text(encoding="utf-8").splitlines()
    return np.asarray([sp[p] for p in paths])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True)
    p.add_argument("--splits", default=str(Path(__file__).parent / "splits.csv"))
    p.add_argument("--feat-len", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="results json (default: next to the npz)")
    a = p.parse_args(argv)
    npz = Path(a.npz)
    d = np.load(npz)
    pf = npz.with_name(npz.stem + ".paths.txt")
    res = run_baselines(d["X"], d["y"], load_split_vector(pf, a.splits), a.feat_len, a.seed,
                        groups=load_session_vector(pf, a.splits))
    here = Path(__file__).parent
    with open(a.splits, newline="", encoding="utf-8") as f:
        in_npz = set(pf.read_text(encoding="utf-8").splitlines())
        res["splits_rows_not_in_npz"] = sorted(r["relative_path"] for r in csv.DictReader(f)
                                               if r["relative_path"] not in in_npz)
    res["inputs_sha256"] = {n: hashlib.sha256((here / n).read_bytes()).hexdigest()
                            for n in ("manifest.csv", "splits.csv")}
    out = Path(a.out) if a.out else npz.with_name(npz.stem + ".baselines.json")
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n": res["n"], "majority": res["majority_class_chance"],
                      **{m: {"selected": v["selected"], "val_macro_f1": v["val"]["macro_f1"],
                             "test_acc": v["test"]["accuracy"], "test_macro_f1": v["test"]["macro_f1"]}
                         for m, v in res["models"].items()}}, indent=1))


if __name__ == "__main__":
    main()
