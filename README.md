# vr-gesture-recognition

Real-time VR controller gesture recognition: train a model in Python, export it to ONNX, and run it in Unity (C#/ML.NET) or in the browser (WebXR/JavaScript).

> TODO: add a short GIF of a gesture being drawn and recognized.

## Repository layout

```
unity/      Unity package (CERTH gestures update.unitypackage), C# scripts, gesture legend prefab
webxr/      browser version: gesture_engine.js, index.html, webxr_vr_paint.html, example data
training/   Hydra + Nevergrad hyperparameter search and sktime benchmarks (components/, research-report.md)
unity-webxr-bridge/  Unity-to-WebXR bridge samples: ML.NET/ONNX inference script, JS gesture engine, JS integration page, example data and ONNX models
docs/       VRController-Gesture Recognition.docx
```

## Quick start

### Unity
1. Import `unity/CERTH gestures update.unitypackage` into your project.
2. Add `VRControllerGestureInput` to your controller object and copy the ONNX model `unity/vr-controller-gestrec-ext-wneg.onnx` to `Assets/StreamingAssets/gestures/` (the inference component loads it from there; the file name and an optional absolute path override are serialized fields on `MLNETVRControllerGestureInference`).
3. TODO: Unity version, XR plugin requirements, ML.NET/ONNX runtime package versions.

### WebXR / browser
```bash
cd webxr
npx light-server -s . -p 8080
```
Open http://localhost:8080/ in a WebXR-capable browser. (`npx` ships with npm; no separate install needed.)

### Training
Run from `training/components/ml/src` (Hydra configs live in `conf/`).

**Python version:** the training requirements (numpy 1.22, sktime 0.15.1, ...) are pinned for Python 3.10 only; use a Python 3.10 environment. Other Python versions were not tested for training (the pins, e.g. numpy 1.22 and tensorflow 2.11, predate newer interpreters); CI only runs the CPU `dataset_v2` tests and lint. The `dataset_v2` tools are separate and need only numpy and scikit-learn.

Nevergrad is pinned to a git commit in `requirements.txt` because a release was not verified compatible with the patched sweeper (see the comment there).

Install dependencies, including the vendored Nevergrad sweeper (an edited copy of the PyPI archive, patched for numpy/nevergrad version incompatibilities):
```bash
cd training/components/ml
pip install -r requirements.txt
pip install deps/hydra-nevergrad-sweeper-1.2.0.tar
```

Machine-specific settings come from `conf/environment/local.yaml`, overridable through environment variables:
`VRG_PYTHON`, `VRG_SRC_DIR` (also the MLflow `mlruns` location), `VRG_CONTROLLER_DATASET_DIR`, `VRG_HAND_DATASET_DIR`, `VRG_ONNX_MODEL` (model used by `evaluate_onnx`).
```bash
cd training/components/ml/src
export VRG_CONTROLLER_DATASET_DIR=/path/to/controller-gesture-dataset
python train.py --config-name train-all -m
```
- `train.py` has no default `train` config file, so pass `--config-name` (available: `train-all`, `train-alex`, `train-george`; other entry points: `bench.py` with `bench-all`, `bench-controller`, `bench-left-hands`, `bench-right-hands`; `evaluate_cv.py` (default config `test`, used by `run_evaluation.bat`); `export_onnx_cli.py` and `test_onnx.py` (default config `onnx`); `bench.py` defaults to `bench-all`). See the `run_*.bat` scripts for Windows examples.
- `.vscode/launch.json` reads dataset locations from `VRG_DATASET_ROOT_A` (root holding `controller-gesture-dataset/` and `hand-gesture-dataset/`) and `VRG_DATASET_ROOT_B` (root holding the `vcl3d-*` datasets, e.g. `vcl3d-hand-gesture-dataset/`). Set both before using those launch configurations.
- Dataset format and download location: TODO.
- Sweep results and ONNX export workflow: TODO.
- Run outputs (`mlruns/`, `multirun/`) are git-ignored.

## VRGestures v2 dataset tooling (`dataset_v2/`)

Manifest, loader, session-grouped splits and baselines for the cleaned hand-gesture set (24 classes: 11 gestures per hand + Teleportation + Zoom). **No recordings are stored in this repository** (npz/data are git-ignored); `dataset_v2/` holds only code and derived metadata (`manifest.csv`, `splits.csv`, summaries). Recordings are kept locally (path via the `VRG_V2_RECORDINGS_ROOT` env var or `--recordings-root`) and are never committed. See `dataset_v2/DATA_CARD.md`.

```bash
python -m dataset_v2.build_manifest --recordings-root <.../Gestures> --check-file <check.json>   # or VRG_V2_RECORDINGS_ROOT / VRG_V2_CHECK_FILE
python -m dataset_v2.splits
python -m dataset_v2.loader --recordings-root <.../Gestures> --out <outside-repo>/vrgestures_v2.npz --workers 4
python -m dataset_v2.baselines --npz <outside-repo>/vrgestures_v2.npz
python -m pytest -q dataset_v2
```

Verified from the data: file counts, class names, session folders, JSON schema, frame statistics. Only a recording the author's check file marks `true` is included (7,182 in the manifest, 7,180 in the 24-class vocabulary); `false` and unchecked recordings are excluded. **Not verified / unknown** (marked `TODO(author)` in the data card): headset and tracking SDK, capture protocol, number of participants, exact consent/ethics wording (the author states participants consented), licence for v2, and the relation to the Zenodo v1 record. Participant ids are not in the files, so splits are grouped by recording session, which is only a proxy: no session-to-participant mapping exists, so participant independence between splits cannot be guaranteed.

## Results (VRGestures v2, 24 classes, session-grouped split)

Test split: 1002 recordings from 9 sessions. Chance: 0.030 (majority class), 0.042 (uniform). Neural models: 3 training seeds (0, 1, 2), 60 epochs, CPU; epoch chosen on validation macro-F1 only, test scored once per run. Logistic regression is a single deterministic fit.

| model | test accuracy | test macro-F1 | per-seed accuracy |
|---|---|---|---|
| logistic regression (16-step features) | 0.397 | 0.397 | n/a |
| GRU (43k params) | 0.409 ± 0.022 | 0.372 ± 0.021 | 0.381, 0.408, 0.436 |
| 1D-CNN (70k params) | 0.219 ± 0.029 | 0.219 ± 0.025 | 0.179, 0.237, 0.242 |

Mean ± population std over the 3 seeds. 95% session-bootstrap intervals (1000 resamples of the 9 test sessions) exist for the seed-0 run only: logistic regression accuracy [0.323, 0.473]; GRU [0.278, 0.488]; CNN [0.120, 0.228]. They reflect which test sessions were drawn, not seed variance, and the seed-0 GRU happens to be its weakest seed.

What this supports: the CNN is clearly worse than logistic regression (paired difference CI excludes 0 on both metrics). The GRU is not distinguishable from logistic regression on accuracy, and its seed-0 macro-F1 is lower (paired CI [-0.107, -0.014], uncorrected for multiple comparisons); its 3-seed mean macro-F1 (0.372) is below the logistic regression value (0.397). **No neural model beat the linear baseline on this split.** Only 3 seeds and 9 test sessions, so small differences are not resolved. No run hit the epoch cap, but no augmentation or tuning beyond epoch selection was tried. Participant independence between splits is not guaranteed (see the data note above).

Sources: `dataset_v2/baselines_results.json` and `dataset_v2/deep_baseline_results.json` (command, seeds, npz/splits/baseline hashes, git commit, torch version recorded in the latter; the logistic-regression refit reproduced the stored accuracy). Reproduce: `python -m dataset_v2.deep_baseline --npz <outside-repo>/vrgestures_v2.npz --seeds 0 1 2 --epochs 60` (needs `pip install torch`; about 1 hour on CPU; CI skips these tests when torch is absent).

## Architecture and preprocessing consistency

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the pipeline diagram and a side-by-side table of the preprocessing each client applies. The Unity, bridge C# and WebXR JS clients do **not** preprocess identically (channel count, reference frame, z sign, time channel). `parity/` runs the real `webxr/gesture_engine.js` under Node against a Python reference; the known differences are recorded as strict expected failures (`python -m pytest parity -q`, needs `npm ci --prefix parity`). The reference is the translation-only transform, so this shows the clients differ from each other, not yet how any of them differs from the training preprocessing. Model files are listed with hashes in [MODELS.md](MODELS.md); which ONNX was trained on what is unknown.

## Gestures

Gesture set shown in `unity/` legend images. Recognition is based on the controller trajectory (see `GestureType.cs`).

## Models

| File | Used by | Notes |
|---|---|---|
| unity/vr-controller-gestrec-ext-wneg.onnx | Unity (canonical copy; also byte-identical to the former `training/components/ml/src/model.onnx`, sha256 852cc3a7...a05da5, which was removed - set `VRG_ONNX_MODEL` to evaluate it) | TODO: classes, input shape, accuracy |
| webxr/pyrocket3.onnx | WebXR | TODO |
| unity-webxr-bridge/pyrocket2.onnx | bridge samples | TODO: purpose and differences from pyrocket3 |
| unity-webxr-bridge/vr-controller-gestrec-ext-wneg.onnx | bridge samples | same file name as the Unity model but contents differ (binary compare); TODO: document the difference |

## Research notes

`training/research-report.md` lists the goals, the sktime benchmark and related work.

## Citation

```
TODO: paper reference (first author: G. Papado)
```

## License

MIT, see [LICENSE](LICENSE).
