"""Simple baselines on the session-grouped split of VRGestures v2 (numpy + scikit-learn only).

Features: the stored (T, C) series is subsampled to ``--feat-len`` steps (linear index grid), flattened and
standardised (statistics from train only). Models: multinomial logistic regression and shrinkage LDA. For each
model the hyper-parameter is chosen on val by macro-F1, then the chosen model is scored on test ONCE.
Reported: accuracy, macro-F1, per-class recall, majority-class chance (train majority applied to the test set).

    python -m dataset_v2.baselines --npz <dir>/vrgestures_v2.npz --splits dataset_v2/splits.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

GRIDS = {
    "logreg": [("C", 0.001), ("C", 0.01), ("C", 0.1)],
    "lda": [("shrinkage", 0.1), ("shrinkage", 0.5), ("shrinkage", 0.9)],
}


def flatten_features(X, feat_len):
    T = X.shape[1]
    idx = np.unique(np.round(np.linspace(0, T - 1, min(feat_len, T))).astype(int))
    return X[:, idx, :].reshape(len(X), -1).astype(np.float64)


def make_model(name, param, seed):
    _, v = param
    if name == "logreg":
        return LogisticRegression(C=v, max_iter=300, random_state=seed)
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=v)


def metrics(y_true, y_pred, labels):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "per_class_recall": {int(l): float(r) for l, r in zip(
                labels, recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0))}}


def run_baselines(X, y, split, feat_len=16, seed=0, models=("logreg", "lda")):
    F = flatten_features(X, feat_len)
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    if not (tr.any() and va.any() and te.any()):
        raise ValueError("train/val/test must all be non-empty")
    sc = StandardScaler().fit(F[tr])
    Ftr, Fva, Fte = sc.transform(F[tr]), sc.transform(F[va]), sc.transform(F[te])
    labels = sorted(int(v) for v in np.unique(y))
    maj = np.bincount(y[tr]).argmax()
    out = {"n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}, "n_features": int(F.shape[1]),
           "feat_len": feat_len, "labels": labels,
           "majority_class_chance": {"class_id": int(maj),
                                     "val_accuracy": float((y[va] == maj).mean()),
                                     "test_accuracy": float((y[te] == maj).mean()),
                                     "uniform_chance": 1.0 / len(labels)},
           "models": {}}
    for name in models:
        tried = []
        best = None
        for param in GRIDS[name]:
            m = make_model(name, param, seed).fit(Ftr, y[tr])
            f1 = f1_score(y[va], m.predict(Fva), labels=labels, average="macro", zero_division=0)
            tried.append({"param": {param[0]: param[1]}, "val_macro_f1": float(f1)})
            if best is None or f1 > best[0]:
                best = (f1, param, m)
        _, param, m = best
        res = metrics(y[te], m.predict(Fte), labels)  # test touched exactly once per model
        out["models"][name] = {"selected": {param[0]: param[1]}, "val_search": tried,
                               "val": metrics(y[va], m.predict(Fva), labels), "test": res}
    return out


def load_split_vector(npz_paths_file, splits_csv):
    sp = {}
    with open(splits_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sp[r["relative_path"]] = r["split"]
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
    split = load_split_vector(npz.with_name(npz.stem + ".paths.txt"), a.splits)
    res = run_baselines(d["X"], d["y"], split, a.feat_len, a.seed)
    out = Path(a.out) if a.out else npz.with_name(npz.stem + ".baselines.json")
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n": res["n"], "majority": res["majority_class_chance"],
                      **{m: {"selected": v["selected"], "val_macro_f1": v["val"]["macro_f1"],
                             "test_acc": v["test"]["accuracy"], "test_macro_f1": v["test"]["macro_f1"]}
                         for m, v in res["models"].items()}}, indent=1))


if __name__ == "__main__":
    main()
