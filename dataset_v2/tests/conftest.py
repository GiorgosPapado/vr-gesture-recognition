import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset_v2.classes import CLASSES


def write_recording(path, cls, T, rng):
    """Synthetic recording in the real schema; the unused hand is an empty list."""
    def pose(n):
        return {"pos": [dict(zip("xyz", rng.randn(3).tolist())) for _ in range(n)],
                "rotation": [dict(zip("xyzw", rng.randn(4).tolist())) for _ in range(n)]}

    def hand():
        return [{"pos": [dict(zip("xyz", rng.randn(3).tolist())) for _ in range(24)],
                 "rotation": [dict(zip("xyzw", rng.randn(4).tolist())) for _ in range(24)]} for _ in range(T)]

    left = cls.startswith("Left") or not cls.startswith("Right")
    right = cls.startswith("Right") or not cls.startswith("Left")
    fd = {"right_hand": hand() if right else [], "left_hand": hand() if left else [], "hmd": pose(T),
          "right_hand_global": pose(T) if right else {"pos": [], "rotation": []},
          "left_hand_global": pose(T) if left else {"pos": [], "rotation": []},
          "timestamps": (np.arange(T) * 0.03).tolist()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"frameData": fd, "class_name": cls, "class_ID": 0}), encoding="utf-8")


@pytest.fixture(scope="session")
def tree(tmp_path_factory):
    """8 sessions x 6 classes x 3 recordings; plus loose sample, excluded folder, unchecked, false, missing."""
    root = tmp_path_factory.mktemp("rec") / "Gestures"
    rng = np.random.RandomState(0)
    classes = [CLASSES[0], CLASSES[3], CLASSES[11], CLASSES[14], "Teleportation", "Zoom"]
    check = {}
    n = 0
    for s in range(8):
        sess = f"abc{s:04d} October 1{s} 10"
        for ci, c in enumerate(classes):
            for k in range(3):
                name = f"{c} {ci} h{n:04d} 0{s + 1} October 1{s} 10 {k}.json"
                write_recording(root / sess / name, c, 10 + 5 * k + s, rng)
                check[name] = True
                n += 1
    first = sorted((root / "abc0000 October 10 10").iterdir())
    check[first[0].name] = False  # flagged false
    del check[first[1].name]  # unchecked
    check["Zoom 5 gone 01 October 10 10 9.json"] = True  # no such file
    write_recording(root / "LeftClick.json", "LeftClick", 8, rng)  # loose sample in root
    check["LeftClick.json"] = True
    for oov in ("LeftOK", "Zoom"):
        write_recording(root / f"{oov}.json", oov, 8, rng)
        check[f"{oov}.json"] = True
    write_recording(root / "ExcludedTest" / "LeftClick 9 x 01 October 10 10 1.json", "LeftClick", 8, rng)
    check["LeftClick 9 x 01 October 10 10 1.json"] = True
    cf = root.parent / "check.json"
    cf.write_text(json.dumps(check), encoding="utf-8")
    return root, cf, check, n
