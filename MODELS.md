# Model registry

Every `*.onnx` tracked by git (`git ls-files`). Hashes and sizes come from `sha256sum` and `stat` on the working tree. Fields marked "unknown" cannot be established from the repository contents.

| Path | Size (bytes) | sha256 | Loaded by (in repo) | Classes | Accuracy | Training run |
|---|---|---|---|---|---|---|
| `webxr/pyrocket3.onnx` | 231705 | `a151088029b13496fd17a609a17d137a8177912afc7adca278248ee5100974a5` | `webxr/gesture_engine.js:14` (`./pyrocket3.onnx`) | unknown | unknown | unknown |
| `unity-webxr-bridge/pyrocket2.onnx` | 133600 | `fef49d7c92d0bb8c5db3d37d9a1a0652c4aba25d728f8be39bf6374e4fa48875` | `unity-webxr-bridge/gesture_engine.js:9` (`./pyrocket2.onnx`) | unknown | unknown | unknown |
| `unity-webxr-bridge/vr-controller-gestrec-ext-wneg.onnx` | 231705 | `0d1ed10ffa1b9ebe3f24469dca4d331afb581e62fbd33236addaa0a096e6d99d` | `unity-webxr-bridge/MLNETVRControllerGestureInference.cs:16` (default `m_ModelFileName`) | unknown | unknown | unknown |
| `unity/vr-controller-gestrec-ext-wneg.onnx` | 1020688 | `852cc3a75e4a6c8d3d197496bbe8d47c1425fe16ae66f0ba308aa0cefda05da5` | `unity/MLNETVRControllerGestureInference.cs:16` (default `m_ModelFileName`) | unknown | unknown | unknown |
| `training/components/ml/src/model1.onnx` | 1021332 | `a7d29535176ce0a28edbb733e424ac32ca4079080977589351c8eb59e84fc703` | none found | unknown | unknown | unknown |

## Notes

- The `unity/` and `unity-webxr-bridge/` copies of `vr-controller-gestrec-ext-wneg.onnx` share a filename but are different files: 1020688 vs 231705 bytes, different sha256.
- `unity-webxr-bridge/vr-controller-gestrec-ext-wneg.onnx` has the same size as `webxr/pyrocket3.onnx` (231705 bytes) but a different sha256; any relationship between them is unknown.
- `training/components/ml/src/model1.onnx` is not referenced by any code, config, script or doc in the repository (searched all tracked non-`mlruns` files). It is kept, not deleted. Its size is close to, but not the same as, the `unity/` model (1021332 vs 1020688 bytes); the relationship is unknown.
- Which model is the "shipped" one is not determinable from the repository.
- Clients refer to models by filename only; the Unity default can be overridden through the serialized field, so the table shows code defaults, not necessarily what a scene uses.
