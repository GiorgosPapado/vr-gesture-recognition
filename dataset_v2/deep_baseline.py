"""Small neural baselines (1D-CNN, GRU) on the stored (N, T, C) series of VRGestures v2. Needs torch (optional).

Same npz and ``splits.csv`` as ``baselines.py``. Channels are standardised with train statistics only; for each
model the epoch is chosen on val macro-F1 only (best epoch's weights are kept), then the chosen model is scored on
test ONCE. Reported: accuracy, macro-F1, session-cluster bootstrap 95% CIs (``baselines.cluster_bootstrap``) and
the paired difference against the logistic regression of ``baselines_results.json`` (refit with its selected C and
feat_len, same bootstrap resamples).

    python -m dataset_v2.deep_baseline --npz <dir>/vrgestures_v2.npz --splits dataset_v2/splits.csv
"""
import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from .baselines import (
    N_BOOT,
    ROOT_SESSION,
    cluster_bootstrap,
    flatten_features,
    load_session_vector,
    load_split_vector,
    make_model,
    metrics,
)

MODELS = ("cnn", "gru")


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise SystemExit("deep_baseline needs torch: pip install torch --index-url https://download.pytorch.org/whl/cpu") from e
    return torch


def build_model(name, n_channels, n_classes, hidden=64, dropout=0.3):
    torch = _torch()
    nn = torch.nn

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_channels, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

        def forward(self, x):  # x: (B, T, C)
            return self.net(x.transpose(1, 2))

    class GRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(n_channels, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

        def forward(self, x):
            _, h = self.rnn(x)
            return self.head(h[-1])

    return CNN() if name == "cnn" else GRU()


def standardise(X, tr):
    """Per-channel mean/std over train samples and time; applied to all."""
    mu = X[tr].mean(axis=(0, 1), keepdims=True)
    sd = X[tr].std(axis=(0, 1), keepdims=True)
    return ((X - mu) / np.where(sd < 1e-6, 1.0, sd)).astype(np.float32)


def _predict(model, X, bs=512):
    torch = _torch()
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(X[i:i + bs])).argmax(1).numpy() for i in range(0, len(X), bs)])


def train_one(name, Xtr, ytr, Xva, yva, n_classes, seed, epochs=30, lr=2e-3, bs=64, weight_decay=1e-4):
    """Train with Adam; keep the weights of the epoch with the best val macro-F1 (val only). -> model, log."""
    import copy

    from sklearn.metrics import f1_score
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = build_model(name, Xtr.shape[2], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    labels = list(range(n_classes))
    best, log = (-1.0, None, -1), []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        tot = 0.0
        for i in range(0, len(xt), bs):
            b = perm[i:i + bs]
            if len(b) < 2:  # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        f1 = float(f1_score(yva, _predict(model, Xva), labels=labels, average="macro", zero_division=0))
        log.append({"epoch": ep + 1, "train_loss": tot / len(xt), "val_macro_f1": f1})
        if f1 > best[0]:
            best = (f1, copy.deepcopy(model.state_dict()), ep + 1)
    model.load_state_dict(best[1])
    return model, {"selected_epoch": best[2], "best_val_macro_f1": best[0], "epochs": log,
                   "hit_epoch_cap": best[2] == epochs}


def run_deep(X, y, split, groups=None, seeds=(0,), models=MODELS, epochs=30, n_boot=N_BOOT, logreg=None, feat_len=16):
    """``logreg`` = {"C": float[, "stored_test_accuracy": float]} adds a refit logistic regression for the paired
    difference. Each model is trained once per seed; ``out["models"][m]`` holds the FIRST seed's run (its test
    predictions feed the bootstrap, ``ci_scope``), ``per_seed`` / ``mean_std`` summarise all seeds."""
    seeds = [int(s) for s in ([seeds] if np.isscalar(seeds) else seeds)]
    seed = seeds[0]
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    if not (tr.any() and va.any() and te.any()):
        raise ValueError("train/val/test must all be non-empty")
    labels = sorted(int(v) for v in np.unique(y))
    idx = {c: i for i, c in enumerate(labels)}
    yi = np.asarray([idx[int(v)] for v in y], dtype=np.int64)
    Xs = standardise(X.astype(np.float32), tr)
    out = {"n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}, "labels": labels,
           "seed": seed, "seeds": seeds, "epochs": epochs,
           "ci_scope": f"seed {seed} run only (first of seeds); mean/std over seeds in models[*].mean_std",
           "input_shape": list(X.shape[1:]), "models": {}}
    test_pred = {}
    for name in models:
        runs = []
        for sd in seeds:
            t0 = time.time()
            m, info = train_one(name, Xs[tr], yi[tr], Xs[va], yi[va], len(labels), sd, epochs)
            print(f"trained {name} seed {sd} {time.time() - t0:.0f}s best epoch {info['selected_epoch']}",
                  file=sys.stderr, flush=True)
            pv, pt = _predict(m, Xs[va]), _predict(m, Xs[te])  # test predicted once per trained model
            pred = np.asarray(labels)[pt]
            runs.append({**info, "seed": sd, "n_params": int(sum(p.numel() for p in m.parameters())),
                         "train_seconds": time.time() - t0,
                         "val": metrics(y[va], np.asarray(labels)[pv], labels), "test": metrics(y[te], pred, labels)})
            if sd == seed:
                test_pred[name] = pred
        acc = np.array([r["test"]["accuracy"] for r in runs])
        f1 = np.array([r["test"]["macro_f1"] for r in runs])
        out["models"][name] = {
            **runs[0],
            "per_seed": [{"seed": r["seed"], "test_accuracy": r["test"]["accuracy"],
                          "test_macro_f1": r["test"]["macro_f1"], "selected_epoch": r["selected_epoch"],
                          "hit_epoch_cap": r["hit_epoch_cap"]} for r in runs],
            "mean_std": {"test_accuracy_mean": float(acc.mean()), "test_accuracy_std": float(acc.std(ddof=0)),
                         "test_macro_f1_mean": float(f1.mean()), "test_macro_f1_std": float(f1.std(ddof=0))},
            "hit_epoch_cap_any": any(r["hit_epoch_cap"] for r in runs)}
    if logreg is not None:
        F = flatten_features(X, feat_len)
        sc = StandardScaler().fit(F[tr])
        lr = make_model("logreg", ("C", logreg["C"]), seed).fit(sc.transform(F[tr]), y[tr])
        test_pred["logreg"] = lr.predict(sc.transform(F[te]))
        out["logreg_refit"] = {"C": logreg["C"], "feat_len": feat_len, "test": metrics(y[te], test_pred["logreg"], labels)}
        stored = logreg.get("stored_test_accuracy")
        if stored is not None:
            ok = abs(out["logreg_refit"]["test"]["accuracy"] - stored) <= 1e-6
            out["logreg_refit_matches_stored"] = bool(ok)
            if not ok:
                warnings.warn(f"logreg refit test accuracy {out['logreg_refit']['test']['accuracy']!r} != stored "
                              f"{stored!r}; paired differences are against a different model than published")
    if groups is not None:
        sess = np.asarray(groups)[te]
        pairs = [(m, "logreg") for m in models] if logreg is not None else []
        out["bootstrap"] = cluster_bootstrap(y[te], test_pred, sess, labels, n_boot, seed, diff_pairs=pairs)
        if (sess == ROOT_SESSION).any():
            nr = sess != ROOT_SESSION
            out["bootstrap_excl_root"] = cluster_bootstrap(
                y[te][nr], {k: v[nr] for k, v in test_pred.items()}, sess[nr], labels, n_boot, seed, diff_pairs=pairs)
    return out


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent, timeout=10).stdout.strip()
    except Exception:  # not a git checkout / git missing
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True)
    p.add_argument("--splits", default=str(Path(__file__).parent / "splits.csv"))
    p.add_argument("--baselines-json", default=str(Path(__file__).parent / "baselines_results.json"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="train every model once per seed; bootstrap CIs use the first seed")
    p.add_argument("--out", default=str(Path(__file__).parent / "deep_baseline_results.json"))
    a = p.parse_args(argv)
    torch = _torch()
    npz = Path(a.npz)
    d = np.load(npz)
    pf = npz.with_name(npz.stem + ".paths.txt")
    base = json.loads(Path(a.baselines_json).read_text(encoding="utf-8"))
    res = run_deep(d["X"], d["y"], load_split_vector(pf, a.splits), load_session_vector(pf, a.splits), a.seeds,
                   epochs=a.epochs, logreg={"C": base["models"]["logreg"]["selected"]["C"],
                                            "stored_test_accuracy": base["models"]["logreg"]["test"]["accuracy"]},
                   feat_len=base["feat_len"])
    res["reference_logreg_in_baselines_json"] = base["models"]["logreg"]["test"]
    res["provenance"] = {"command": "python -m dataset_v2.deep_baseline " + " ".join(argv if argv is not None else sys.argv[1:]),
                         "seeds": a.seeds, "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                         "splits_sha256": _sha(a.splits), "baselines_results_sha256": _sha(a.baselines_json),
                         "git_commit": _git_commit(), "torch_num_threads": torch.get_num_threads(),
                         "deterministic_algorithms": "warn_only",
                         "torch_version": torch.__version__, "python": platform.python_version(),
                         "device": "cpu"}
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n": res["n"], **{m: {"epoch": v["selected_epoch"], "val_macro_f1": v["val"]["macro_f1"],
                                           "test_acc": v["test"]["accuracy"], "test_macro_f1": v["test"]["macro_f1"]}
                                        for m, v in res["models"].items()}}, indent=1))


if __name__ == "__main__":
    main()
