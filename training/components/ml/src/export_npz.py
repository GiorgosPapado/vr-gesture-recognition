"""Convert the controller / hand gesture datasets to the generic npz format used by efficient-inference/ofa_transformer.

Output: ``<out>.npz`` with ``X`` (N, T, C) float32, ``y`` (N,) int64 (the dataset's GestureID) and
``<out>.meta.json`` (feature names, channel names, resample length, id -> gesture name map).
Every gesture recording has a different length, so each is linearly resampled to ``--length`` steps (use
``--pad`` to zero-pad / crop instead). The datasets carry no subject id, so no ``subject`` array is written.

Run from this folder (it reuses ``dataset/controller.py`` and ``dataset/hands.py``)::

    python export_npz.py --loader controller --dataset-dir <folder with *.bin> --out out/controller.npz
    python export_npz.py --loader hand --dataset-dir <folder with *.bin> --out out/hand.npz --handedness right
"""
import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_FEATURES = {
    "controller": ["ControllerPos", "ControllerOrientation"],
    "hand": ["HandPos", "HandOrientation"],
}


def to_fixed_length(series_list, length, pad):
    """(C pd.Series of equal length L) -> (length, C) float32; linear resample, or zero-pad / crop when ``pad``."""
    cols = [np.asarray(s, dtype=np.float64) for s in series_list]
    arr = np.stack(cols, axis=1)  # (L, C)
    n = arr.shape[0]
    if pad:
        out = np.zeros((length, arr.shape[1]), np.float32)
        m = min(n, length)
        out[:m] = arr[:m]
        return out
    if n == 1:
        return np.repeat(arr, length, axis=0).astype(np.float32)
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, length)
    return np.stack([np.interp(dst, src, arr[:, j]) for j in range(arr.shape[1])], axis=1).astype(np.float32)


def convert(df, labels, length, pad):
    X = np.stack([to_fixed_length(df.iloc[i].tolist(), length, pad) for i in range(len(df))])
    return X, np.asarray(labels, dtype=np.int64)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--loader", choices=sorted(DEFAULT_FEATURES), required=True)
    p.add_argument("--dataset-dir", required=True, help="folder containing the *.bin gesture recordings")
    p.add_argument("--out", required=True, help="output .npz path (a .meta.json is written next to it)")
    p.add_argument("--length", type=int, default=64, help="time steps T per example (default 64)")
    p.add_argument("--pad", action="store_true", help="zero-pad / crop to --length instead of resampling")
    p.add_argument("--features", nargs="+", default=None, help="feature names (default: pos + orientation)")
    p.add_argument("--handedness", choices=["left", "right"], default=None)
    p.add_argument("--index-file", default=None, help="optional split csv (e.g. controller_train_split.txt)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    features = args.features or DEFAULT_FEATURES[args.loader]
    if args.loader == "controller":
        from dataset.controller import load_dataset
    else:
        from dataset.hands import load_dataset
    df, labels = load_dataset(args.dataset_dir, feature_names=features, index_file=args.index_file,
                              handedness=args.handedness)
    if len(df) == 0:
        raise SystemExit("No recordings matched; check --dataset-dir / --handedness / --index-file")
    X, y = convert(df, labels, args.length, args.pad)
    out = Path(args.out).with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y)
    meta = {"source": f"vr-gesture-recognition {args.loader} dataset", "features": features,
            "n_channels": int(X.shape[2]), "window_len": int(X.shape[1]), "n": len(X),
            "length_mode": "pad_crop" if args.pad else "linear_resample", "handedness": args.handedness,
            "class_ids": sorted(int(v) for v in np.unique(y)), "has_subject": False}
    out.with_name(out.stem + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}: X{tuple(X.shape)} y{tuple(y.shape)} classes={meta['class_ids']}")


if __name__ == "__main__":
    main()
