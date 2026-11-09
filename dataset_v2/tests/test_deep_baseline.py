import numpy as np
import pytest

pytest.importorskip("torch")

from dataset_v2.deep_baseline import run_deep, standardise  # noqa: E402


def _data(seed=0):
    rng = np.random.RandomState(seed)
    n, T, C = 120, 16, 8
    y = np.repeat(np.arange(3), n // 3)
    X = rng.randn(n, T, C).astype(np.float32) * 0.3
    X[:, :, 0] += y[:, None]  # class-dependent signal
    split = np.array((["train"] * 6 + ["val"] * 2 + ["test"] * 2) * (n // 10))
    # session ids are split-specific: disjoint across train/val/test
    groups = np.array([f"{sp}_s{i % 3}" for i, sp in enumerate(split)])
    return X, y, split, groups


def test_standardise_uses_train_only():
    X, y, split, _ = _data()
    tr = split == "train"
    Z = standardise(X, tr)
    assert abs(Z[tr].mean()) < 1e-4
    X2 = X.copy()
    X2[~tr] += 100.0 * np.random.RandomState(1).randn(*X2[~tr].shape).astype(np.float32)  # perturb val/test only
    Z2 = standardise(X2, tr)
    np.testing.assert_array_equal(Z2[tr], Z[tr])  # mu/sd come from train only -> train rows unchanged
    mu = X[tr].mean(axis=(0, 1), keepdims=True)
    sd = X[tr].std(axis=(0, 1), keepdims=True)
    np.testing.assert_allclose(Z2[~tr], (X2[~tr] - mu) / sd, rtol=1e-4, atol=1e-4)


def test_groups_disjoint_across_splits():
    _, _, split, groups = _data()
    sets = [set(groups[split == s]) for s in ("train", "val", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


@pytest.mark.parametrize("model", ["cnn", "gru"])
def test_run_deep_synthetic(model):
    X, y, split, groups = _data()
    r = run_deep(X, y, split, groups, seeds=(0, 1), models=(model,), epochs=8, n_boot=20,
                 logreg={"C": 1.0}, feat_len=4)
    m = r["models"][model]
    assert m["test"]["accuracy"] > 0.6
    assert 1 <= m["selected_epoch"] <= 8
    assert f"{model}_minus_logreg" in r["bootstrap"]["paired"]
    assert [q["seed"] for q in m["per_seed"]] == [0, 1]
    assert {"test_accuracy_mean", "test_accuracy_std"} <= set(m["mean_std"])
    assert isinstance(m["hit_epoch_cap"], bool) and "ci_scope" in r
    r2 = run_deep(X, y, split, groups, seeds=(0, 1), models=(model,), epochs=8, n_boot=20, logreg={"C": 1.0}, feat_len=4)
    assert r2["models"][model]["test"] == m["test"]  # seeded
