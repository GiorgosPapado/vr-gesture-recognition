# Preprocessing parity

Compares the preprocessing of `webxr/gesture_engine.js` (run unchanged under Node) with a NumPy reference
of `TSControllerTranslationCoordinateTransform` (`training/components/ml/src/export/torch/preproc/controller_transform.py`).
No model is run; the `ort` global is stubbed to capture the tensor that would be fed to the model.

Fixture `fixtures/sample.json`: positions are the real `test[0]` recording of `webxr/gesture_data_test.json`.
That file stores neither orientations nor timestamps, so the two reference orientations and the timestamps in the
fixture are synthetic and only exercise code paths.

The engine imports `../build/three.module.js`, which is not in the repo. `js_runner.mjs` writes a temp copy of the
engine with only that import rewritten to the npm `three` pinned in `package.json`; the engine file is not edited.

## Run

```
cd parity && npm ci && cd ..
node parity/js_runner.mjs parity/fixtures/sample.json   # prints dims + data per case as JSON
python -m pytest parity -q
```

Set `VRG_NODE` if `node` is not on PATH. Tests skip when node or `parity/node_modules` is missing.
Case `identity_ref_orientation_offset` adds a constant offset (`position_offset_xyz`, applied by `js_runner.mjs` to all
positions) and must give the same tensor as the unshifted case, for both the JS engine and the Python reference
(subtraction of the first sample).
Known differences (4th timestamp channel, z flip, rotation by inverse first orientation) are `xfail(strict=True)`.
C# is not executed; see the table in `docs/ARCHITECTURE.md`.
