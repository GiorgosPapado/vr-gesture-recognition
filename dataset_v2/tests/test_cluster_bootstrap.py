import numpy as np
from dataset_v2.baselines import cluster_bootstrap, metrics


def _data():
    y = np.tile(np.arange(3), 20)
    groups = np.repeat([f"s{i}" for i in range(10)], 6)
    return y, groups


def test_identical_predictions_give_zero_diff_ci():
    y, g = _data()
    p = (y + (np.arange(len(y)) % 4 == 0)) % 3
    r = cluster_bootstrap(y, {"a": p, "b": p.copy()}, g, [0, 1, 2], n_boot=50, seed=1, diff_pairs=[("a", "b")])
    d = r["paired"]["a_minus_b"]
    assert d["accuracy_ci95"] == [0.0, 0.0] and d["macro_f1_ci95"] == [0.0, 0.0]
    assert not d["accuracy_differs"] and not d["macro_f1_differs"]


def test_perfect_vs_always_wrong_diff_ci_strictly_positive():
    y, g = _data()
    r = cluster_bootstrap(y, {"good": y.copy(), "bad": (y + 1) % 3}, g, [0, 1, 2], n_boot=50, seed=1,
                          diff_pairs=[("good", "bad")])
    d = r["paired"]["good_minus_bad"]
    assert d["accuracy_ci95"][0] > 0 and d["macro_f1_ci95"][0] > 0
    assert d["accuracy_differs"] and d["macro_f1_differs"]


def test_macro_f1_counts_absent_class_as_zero():
    # Behaviour of baselines.py (kept unchanged, published numbers depend on it): with labels= given and
    # zero_division=0, a class absent from both y_true and y_pred (e.g. missing in a bootstrap resample or the
    # test split) contributes F1 = 0 to the macro mean instead of being skipped.
    y = np.array([0, 0, 1, 1])
    m = metrics(y, y, [0, 1, 2])
    assert m["accuracy"] == 1.0
    assert abs(m["macro_f1"] - 2 / 3) < 1e-12
