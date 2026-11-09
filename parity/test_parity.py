import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from reference_py import translation_transform

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "sample.json"
ATOL = 1e-6


def _find_node():
    for cand in (os.environ.get("VRG_NODE"), shutil.which("node"), r"C:\Program Files\nodejs\node.exe"):
        if cand and Path(cand).exists():
            return cand
    return None


@pytest.fixture(scope="module")
def js():
    node = _find_node()
    required = os.environ.get("VRG_REQUIRE_JS") == "1"  # set in CI: a missing toolchain must fail, not skip
    skip = pytest.fail if required else pytest.skip
    if node is None:
        skip("node not found (set VRG_NODE)")
    if not (HERE / "node_modules" / "three").exists():
        skip("run `npm ci` in parity/ first")
    out = subprocess.run(
        [node, str(HERE / "js_runner.mjs"), str(FIXTURE)], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out)


@pytest.fixture(scope="module")
def fixture():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def ref(fixture):
    return translation_transform(np.array(fixture["positions_xyz_by_time"]))


def _js_matrix(js, case):
    dims = js[case]["dims"]
    return np.array(js[case]["data"], dtype=np.float32).reshape(dims[1], dims[2])


def test_js_tensor_layout_is_1xCxT(js, ref):
    assert js["identity_ref_orientation"]["dims"][0] == 1
    assert js["identity_ref_orientation"]["dims"][2] == ref.shape[1]


@pytest.mark.xfail(strict=True, reason="JS engine emits 4 channels (x,y,z,timestamp) (webxr/gesture_engine.js:38,61); "
                   "torch translation transform yields 3 channels")
def test_channel_count(js, ref):
    assert _js_matrix(js, "identity_ref_orientation").shape[0] == ref.shape[0]


def test_xy_match_with_identity_orientation(js, ref):
    m = _js_matrix(js, "identity_ref_orientation")
    np.testing.assert_allclose(m[:2], ref[:2], atol=ATOL)


@pytest.mark.xfail(strict=True, reason="JS negates z after the transform (webxr/gesture_engine.js:49); reference does not")
def test_z_matches(js, ref):
    m = _js_matrix(js, "identity_ref_orientation")
    np.testing.assert_allclose(m[2], ref[2], atol=ATOL)


def test_z_is_exactly_negated(js, ref):
    m = _js_matrix(js, "identity_ref_orientation")
    np.testing.assert_allclose(m[2], -ref[2], atol=ATOL)


def test_timestamp_channel_is_passthrough(js, fixture):
    m = _js_matrix(js, "identity_ref_orientation")
    np.testing.assert_allclose(m[3], np.array(fixture["timestamps_synthetic"], dtype=np.float32), atol=ATOL)


@pytest.mark.xfail(strict=True, reason="JS rotates by inverse of the first orientation (webxr/gesture_engine.js:48); "
                   "the torch translation transform does not rotate")
def test_xy_match_with_rotated_reference_orientation(js, ref):
    m = _js_matrix(js, "rot90_about_y_ref_orientation")
    np.testing.assert_allclose(m[:2], ref[:2], atol=ATOL)


def test_output_invariant_to_constant_position_offset(js, fixture):
    """Adding a constant offset to every position must not change the tensor (pos[0] is subtracted), identity ref."""
    a = _js_matrix(js, "identity_ref_orientation")
    b = _js_matrix(js, "identity_ref_orientation_offset")
    np.testing.assert_allclose(b, a, atol=1e-5)  # float32 round-off of x+offset-(x0+offset)


def test_python_reference_invariant_to_constant_position_offset(fixture):
    pos = np.array(fixture["positions_xyz_by_time"], dtype=np.float64)
    off = np.array(fixture["cases"][2]["position_offset_xyz"])[:, None]
    np.testing.assert_allclose(translation_transform(pos + off), translation_transform(pos), atol=1e-5)
