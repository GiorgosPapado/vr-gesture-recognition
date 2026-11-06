"""Loader for VRGestures v2 recordings (numpy only) and exporter to the generic npz format.

Recording schema (inspected on real files)::

    {"class_name": str, "class_ID": int,
     "frameData": {
        "right_hand": [frame, ...], "left_hand": [frame, ...],      # one frame per timestamp; the hand that was
        "hmd": {"pos": [xyz...], "rotation": [xyzw...]},             # not used is an EMPTY list (len 0)
        "right_hand_global": {"pos": [...], "rotation": [...]},     # one pose per frame (empty if hand unused)
        "left_hand_global":  {"pos": [...], "rotation": [...]},
        "timestamps": [float, ...]}}                                 # seconds, ~30-40 Hz, T = len(timestamps)
    frame = {"pos": [24 x {x,y,z}], "rotation": [24 x {x,y,z,w}]}  # 24 joints, local to the hand root

The per-joint ``pos`` are hand-local: joint 0 (the hand root) is always (0, 0, 0), so they are already
wrist-relative. ``*_global`` is the hand root pose in world/tracking space and ``hmd`` the headset pose.
Single-hand gestures fill one hand; Teleportation and Zoom were recorded with both hands present (and
sometimes empty lists; missing hands are zero-filled).

Features (per hand, left block then right block, zeros for an absent hand)::

    local_pos  72  joint positions, hand-local (wrist-relative)
    local_rot  96  joint quaternions xyzw, hand-local
    global_pos  3  hand-root position; minus the HMD position when ``hmd_relative`` (default), else absolute
    global_rot  4  hand-root quaternion xyzw (world space; not made HMD-relative)
Default features: local_pos + global_pos(hmd-relative) -> 75 per hand, 150 channels. Quaternion double-cover
(q vs -q) is not normalised.

CLI (streams files with a multiprocessing pool; only a few files are in memory at a time)::

    python -m dataset_v2.loader --manifest dataset_v2/manifest.csv --recordings-root <root> --out <dir>/vrgestures_v2.npz
"""
import argparse
import csv
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

N_JOINTS = 24
FEATURE_DIMS = {"local_pos": N_JOINTS * 3, "local_rot": N_JOINTS * 4, "global_pos": 3, "global_rot": 4}
DEFAULT_FEATURES = ("local_pos", "global_pos")


def read_recording(path):
    """Parse one JSON. Returns dict with timestamps (T,), per hand arrays and hmd pos, all float64.

    keys: timestamps, left_pos (T,24,3)|None, left_rot (T,24,4)|None, left_gpos (T,3)|None, left_grot (T,4)|None,
    same for right_*, hmd_pos (T,3), class_name, class_id."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    fd = d["frameData"]
    ts = np.asarray(fd["timestamps"], dtype=np.float64)
    T = len(ts)

    def xyz(lst, keys):
        return np.asarray([[p[k] for k in keys] for p in lst], dtype=np.float64)

    out = {"timestamps": ts, "class_name": d.get("class_name"), "class_id": d.get("class_ID"),
           "hmd_pos": xyz(fd["hmd"]["pos"], "xyz") if fd["hmd"]["pos"] else np.zeros((T, 3))}
    for side in ("left", "right"):
        frames = fd[f"{side}_hand"]
        g = fd[f"{side}_hand_global"]
        if len(frames) == 0:
            for k in ("pos", "rot", "gpos", "grot"):
                out[f"{side}_{k}"] = None
            continue
        out[f"{side}_pos"] = np.asarray([[[j["x"], j["y"], j["z"]] for j in fr["pos"]] for fr in frames])
        out[f"{side}_rot"] = np.asarray([[[j["x"], j["y"], j["z"], j["w"]] for j in fr["rotation"]] for fr in frames])
        out[f"{side}_gpos"] = xyz(g["pos"], "xyz")
        out[f"{side}_grot"] = xyz(g["rotation"], "xyzw")
    return out


def recording_to_series(rec, features=DEFAULT_FEATURES, hmd_relative=True):
    """-> (T, C) float64 with C = 2 * sum(FEATURE_DIMS[f]); left hand block first, absent hand zero-filled."""
    T = len(rec["timestamps"])
    blocks = []
    for side in ("left", "right"):
        for f in features:
            dim = FEATURE_DIMS[f]
            if rec[f"{side}_pos"] is None:
                blocks.append(np.zeros((T, dim)))
                continue
            if f == "local_pos":
                a = rec[f"{side}_pos"].reshape(T, -1)
            elif f == "local_rot":
                a = rec[f"{side}_rot"].reshape(T, -1)
            elif f == "global_pos":
                a = rec[f"{side}_gpos"] - (rec["hmd_pos"] if hmd_relative else 0.0)
            elif f == "global_rot":
                a = rec[f"{side}_grot"]
            else:
                raise ValueError(f"unknown feature {f!r}; choose from {sorted(FEATURE_DIMS)}")
            blocks.append(np.asarray(a, dtype=np.float64).reshape(T, dim))
    return np.concatenate(blocks, axis=1)


def feature_names(features=DEFAULT_FEATURES):
    names = []
    for side in ("left", "right"):
        for f in features:
            if f == "local_pos":
                names += [f"{side}_j{j}_{a}" for j in range(N_JOINTS) for a in "xyz"]
            elif f == "local_rot":
                names += [f"{side}_j{j}_q{a}" for j in range(N_JOINTS) for a in "xyzw"]
            elif f == "global_pos":
                names += [f"{side}_root_{a}" for a in "xyz"]
            else:
                names += [f"{side}_root_q{a}" for a in "xyzw"]
    return names


def resample_linear(arr, length):
    """(n, C) -> (length, C) by linear interpolation over normalised time; n == 1 is repeated."""
    n = arr.shape[0]
    if n == 0:
        raise ValueError("empty recording")
    if n == 1:
        return np.repeat(arr, length, axis=0)
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, length)
    return np.stack([np.interp(dst, src, arr[:, j]) for j in range(arr.shape[1])], axis=1)


def pad_crop(arr, length):
    """(n, C) -> (length, C): crop to the first ``length`` frames or zero-pad at the end."""
    out = np.zeros((length, arr.shape[1]), dtype=arr.dtype)
    m = min(arr.shape[0], length)
    out[:m] = arr[:m]
    return out


def load_fixed(path, length=64, pad=False, features=DEFAULT_FEATURES, hmd_relative=True):
    """Read one file and return ((length, C) float32, n_frames, class_name_in_file)."""
    rec = read_recording(path)
    s = recording_to_series(rec, features, hmd_relative)
    fixed = pad_crop(s, length) if pad else resample_linear(s, length)
    return fixed.astype(np.float32), s.shape[0], rec["class_name"]


def _work(args):
    path, kw = args
    try:
        return load_fixed(path, **kw)
    except Exception as e:  # report, do not kill a long job
        return repr(e)


def build_arrays(rows, root, length=64, pad=False, features=DEFAULT_FEATURES, hmd_relative=True, workers=4):
    """Stream the recordings of manifest ``rows`` (included ones only) with a process pool.

    Returns X (N, T, C) float32, y (N,) int64, session (N,) int64, sessions (list of names, index = session id),
    paths (list), n_frames (N,) int32, problems (list of (path, message)); problem files are dropped."""
    rows = [r for r in rows if str(r["included"]) == "True"]
    kw = dict(length=length, pad=pad, features=tuple(features), hmd_relative=hmd_relative)
    C = 2 * sum(FEATURE_DIMS[f] for f in features)
    X = np.zeros((len(rows), length, C), np.float32)
    keep, nfr, problems = [], [], []
    jobs = ((os.path.join(root, r["relative_path"]), kw) for r in rows)
    if workers <= 1:
        results = map(_work, jobs)
        pool = None
    else:
        pool = Pool(workers)
        results = pool.imap(_work, jobs, chunksize=8)  # ordered; results are consumed as they arrive
    try:
        for i, res in enumerate(results):
            if isinstance(res, str):
                problems.append((rows[i]["relative_path"], res))
                continue
            arr, n, cname = res
            if cname is not None and cname != rows[i]["class_name_canonical"]:
                problems.append((rows[i]["relative_path"], f"class_name in file {cname!r} != manifest "
                                 f"{rows[i]['class_name_canonical']!r} (kept)"))
            X[len(keep)] = arr
            keep.append(i)
            nfr.append(n)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    X = X[: len(keep)]
    sel = [rows[i] for i in keep]
    sessions = sorted({r["session_id"] for r in sel})
    sid = {s: k for k, s in enumerate(sessions)}
    y = np.asarray([int(r["class_id"]) for r in sel], dtype=np.int64)
    session = np.asarray([sid[r["session_id"]] for r in sel], dtype=np.int64)
    return X, y, session, sessions, [r["relative_path"] for r in sel], np.asarray(nfr, np.int32), problems


def save_npz(out, X, y, session, sessions, class_names, features, length, pad, hmd_relative, paths=None):
    """Write ``<out>.npz`` (X, y, session) and ``<out>.meta.json`` in the layout of export_npz.py.

    ``session`` is NOT a subject id (participants are unknown); the npz therefore has no ``subject`` array and
    meta has_subject is false."""
    out = Path(out).with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, session=session)
    meta = {"source": "VRGestures v2 (vr-gesture-recognition/dataset_v2)", "features": list(features),
            "channel_names": feature_names(features), "n_channels": int(X.shape[2]), "window_len": int(X.shape[1]),
            "n": int(len(X)), "length_mode": "pad_crop" if pad else "linear_resample", "handedness": None,
            "hmd_relative": hmd_relative, "class_ids": sorted(int(v) for v in np.unique(y)),
            "class_names": {str(i): n for i, n in enumerate(class_names)}, "has_subject": False,
            "has_session": True, "sessions": list(sessions)}
    out.with_name(out.stem + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if paths is not None:
        out.with_name(out.stem + ".paths.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    return out


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    from .classes import CLASSES

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=str(Path(__file__).parent / "manifest.csv"))
    p.add_argument("--recordings-root", default=os.environ.get("VRG_V2_RECORDINGS_ROOT"))
    p.add_argument("--out", required=True, help="output .npz path (keep it outside the repository)")
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--pad", action="store_true", help="zero-pad / crop instead of linear resampling")
    p.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES), choices=sorted(FEATURE_DIMS))
    p.add_argument("--no-hmd-relative", action="store_true", help="use absolute global_pos")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    if not a.recordings_root:
        p.error("--recordings-root or VRG_V2_RECORDINGS_ROOT required")
    rows = read_manifest(a.manifest)
    hr = not a.no_hmd_relative
    X, y, session, sessions, paths, nfr, problems = build_arrays(
        rows, a.recordings_root, a.length, a.pad, a.features, hr, a.workers)
    out = save_npz(a.out, X, y, session, sessions, CLASSES, a.features, a.length, a.pad, hr, paths)
    np.save(out.with_name(out.stem + ".n_frames.npy"), nfr)
    print(f"wrote {out}: X{X.shape} y{y.shape} sessions={len(sessions)} problems={len(problems)}")
    for pth, msg in problems[:20]:
        print("  problem:", pth, msg)


if __name__ == "__main__":
    main()
