Install requirements.txt
Install deps/hydra-nevergrad-sweeper-1.2.0.tar (an edited version of the archive found in PyPi to account for numpy and nevergrad version incompatibilities)

Note: the requirements are Windows-oriented (e.g. pywin32, tensorflow-intel) and were not rewritten for other platforms. torch and onnxruntime are listed unpinned.

Training: `python train.py` uses the `train-all` config by default; pass `--config-name <name>` (e.g. `train-alex`, `train-george`) to choose another.
