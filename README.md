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
2. Add `VRControllerGestureInput` to your controller object and assign the ONNX model `unity/vr-controller-gestrec-ext-wneg.onnx`.
3. TODO: Unity version, XR plugin requirements, ML.NET/ONNX runtime package versions.

### WebXR / browser
```bash
cd webxr
npx light-server -s . -p 8080
```
Open http://localhost:8080/ in a WebXR-capable browser. (`npx` ships with npm; no separate install needed.)

### Training
Run from `training/components/ml/src` (Hydra configs live in `conf/`).

Install dependencies, including the vendored Nevergrad sweeper (an edited copy of the PyPI archive, patched for numpy/nevergrad version incompatibilities):
```bash
cd training/components/ml
pip install -r requirements.txt
pip install deps/hydra-nevergrad-sweeper-1.2.0.tar
```

Machine-specific settings come from `conf/environment/local.yaml`, overridable through environment variables:
`VRG_PYTHON`, `VRG_SRC_DIR` (also the MLflow `mlruns` location), `VRG_CONTROLLER_DATASET_DIR`, `VRG_HAND_DATASET_DIR`.
```bash
cd training/components/ml/src
export VRG_CONTROLLER_DATASET_DIR=/path/to/controller-gesture-dataset
python train.py --config-name train-all -m
```
- `train.py` has no default `train` config file, so pass `--config-name` (available: `train-all`, `train-alex`, `train-george`; other entry points: `bench.py` with `bench-all`, `bench-controller`, `bench-left-hands`, `bench-right-hands`; `test.py`; `onnx.py`). See the `run_*.bat` scripts for Windows examples.
- Dataset format and download location: TODO.
- Sweep results and ONNX export workflow: TODO.
- Run outputs (`mlruns/`, `multirun/`) are git-ignored.

## Gestures

Gesture set shown in `unity/` legend images. Recognition is based on the controller trajectory (see `GestureType.cs`).

## Models

| File | Used by | Notes |
|---|---|---|
| unity/vr-controller-gestrec-ext-wneg.onnx | Unity | TODO: classes, input shape, accuracy |
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
