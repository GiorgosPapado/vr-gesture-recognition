Install requirements.txt
Install deps/hydra-nevergrad-sweeper-1.2.0.tar (an edited version of the archive found in PyPi to account for numpy and nevergrad version incompatibilities)

Note: the requirements are Windows-oriented (e.g. pywin32, tensorflow-intel) and were not rewritten for other platforms. torch and onnxruntime are listed unpinned.

Training: `python train.py` uses the `train-all` config by default; pass `--config-name <name>` (e.g. `train-alex`, `train-george`) to choose another.

## Export to the generic npz format

`src/export_npz.py` converts the controller or hand dataset (via the existing `dataset/controller.py` / `dataset/hands.py`
loaders) to `X` (N, T, C) float32 + `y` (N,) int64 in an `.npz` with a `.meta.json`, the interchange format used by
`efficient-inference/ofa_transformer`. Each recording is linearly resampled to `--length` steps (or zero-padded / cropped
with `--pad`); the datasets have no subject id, so none is written. Run it from `src/`:

```
python export_npz.py --loader controller --dataset-dir <folder with *.bin> --out out/controller.npz
python export_npz.py --loader hand --dataset-dir <folder with *.bin> --out out/hand.npz --handedness right --length 64
```
