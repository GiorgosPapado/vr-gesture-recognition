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
TODO: how to run the Hydra/Nevergrad sweep and export ONNX.

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
