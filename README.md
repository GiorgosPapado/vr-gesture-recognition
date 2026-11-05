# vr-gesture-recognition

Real-time VR controller gesture recognition: train a model in Python, export it to ONNX, and run it in Unity (C#/ML.NET) or in the browser (WebXR/JavaScript).

> TODO: add a short GIF of a gesture being drawn and recognized.

## Repository layout

```
unity/      Unity package (CERTH gestures update.unitypackage), C# scripts, gesture legend prefab
webxr/      browser version: gesture_engine.js, index.html, webxr_vr_paint.html, example data
training/   Hydra + Nevergrad hyperparameter search and sktime benchmarks (components/, research-report.md)
models/     exported ONNX models (vr-controller-gestrec-ext-wneg.onnx, pyrocket2.onnx, pyrocket3.onnx)
docs/       VRController-Gesture Recognition.docx
```

## Quick start

### Unity
1. Import `unity/CERTH gestures update.unitypackage` into your project.
2. Add `VRControllerGestureInput` to your controller object and assign the ONNX model from `models/`.
3. TODO: Unity version, XR plugin requirements, ML.NET/ONNX runtime package versions.

### WebXR / browser
```bash
cd webxr
npx light-server -s . -p 8080
```
Open http://localhost:8080/ in a WebXR-capable browser. (`npx` ships with npm; no separate install needed.)

### Training
TODO: how to run the Hydra/Nevergrad sweep and export ONNX.

## Gestures

Gesture set shown in `unity/` legend images. Recognition is based on the controller trajectory (see `GestureType.cs`).

## Models

| File | Used by | Notes |
|---|---|---|
| vr-controller-gestrec-ext-wneg.onnx | Unity | TODO: classes, input shape, accuracy |
| pyrocket2.onnx / pyrocket3.onnx | WebXR | TODO |

## Research notes

`training/research-report.md` lists the goals, the sktime benchmark and related work.

## Citation

```
TODO: paper reference (first author: G. Papado)
```

## License

MIT, see [LICENSE](LICENSE).
