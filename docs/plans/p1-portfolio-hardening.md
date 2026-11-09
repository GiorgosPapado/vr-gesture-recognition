## Plan: P1 portfolio hardening

Form: full (four workstreams, one real design decision on how parity is tested, ML work with unverifiable results).

### Objective
Turn the repo from an archive into a verifiable case study: clear README with only verified facts, a test that exposes the preprocessing mismatch between Python, JS and C#, a de-duplicated tree with a model registry, and a small neural baseline next to the existing 0.40 sklearn baseline.

### Reuse Audit
- Exists: `dataset_v2/` (loader, `splits.csv`, bootstrap code in `baselines.py`, ruff config, pytest + synthetic fixtures in `tests/conftest.py`), `export/torch/preproc/controller_transform.py` (torch reference of controller preprocessing), CI workflow (ruff + pytest on `dataset_v2`).
- Exists, reuse: session-cluster bootstrap and paired difference in `baselines.py` (import, do not copy); `render_baselines_md.py` pattern for generated docs.
- Missing: any JS/C# test, any torch code under `dataset_v2`, MODELS.md, architecture diagram, Node/three shim.
- Environment: no `CLAUDE.md`. Local Python 3.12 without torch; Node at `C:\Program Files\nodejs` (not on bash PATH); dotnet present. CI Python is 3.10 with numpy, scikit-learn, pytest, ruff only.

### File Structure
```
README.md                                   modify  (results, diagram, remove TODOs that can be answered)
MODELS.md                                   new     (sha256, size, client, status per .onnx)
docs/ARCHITECTURE.md                        new     (mermaid pipeline + preprocessing comparison table)
parity/                                     new
  README.md                                          what is compared, how to run
  reference_py.py                                    Python/NumPy transform of one recording (spec from controller_transform.py)
  js_runner.mjs                                      runs webxr/gesture_engine.js preprocessing in Node
  test_parity.py                                     compares outputs, xfail with reason where known to differ
  fixtures/sample.json                               one recording taken from webxr/gesture_data_test.json
dataset_v2/deep_baseline.py                 new     (torch 1D-CNN and GRU, optional import)
dataset_v2/tests/test_deep_baseline.py      new     (pytest.importorskip("torch"); tiny synthetic run)
training/components/ml/src/
  onnx.py            -> export_onnx_cli.py  rename  (shadows `onnx`)
  test.py            -> evaluate_cv.py      rename  (shadows stdlib `test`)
  bench.py                                  modify  (path to renamed script)
  run_evaluation.bat                        modify
  .vscode/launch.json                       modify  (2 + 1 program refs)
  evaluation.py                             delete  (empty)
  train.py                                  modify  (stop swallowing exceptions)
  utils/quaternion.py + dataset/quaternion.py  merge into utils/quaternion.py, delete the commented-out copy
unity-webxr-bridge/gesture_data_{train,test}.json   delete (identical to webxr/); README note
.github/workflows/ci.yml                    modify  (parity job, optional torch job)
```
Not touched: Unity package, ONNX files, vendored sweeper, any YAML deduplication beyond the two byte-identical pairs (separate task).

### Design / Data Flow
**Parity test (decision).** Running C# outside Unity is not possible (UnityEngine types). Options: (a) transcribe each client's logic to Python and compare, (b) run the real JS under Node and compare to a Python reference, (c) full C# via dotnet with Unity stubs. Choose (b) for JS plus a documented side-by-side table for C#; the C# logic (`Inference.cs` lines 30-43 in both copies) is a few lines and is compared by reading, recorded in `docs/ARCHITECTURE.md`. Reason: (b) tests real shipped code; (a) only tests our transcription; (c) costs more than it proves.
Compared quantities per client: channel count, reference frame (subtract `pos[0]`, rotate by inverse of `quat[0]`), z flip, timestamp channel, tensor layout. The test asserts what is true today: JS vs Python reference must match or be marked `xfail(strict=True)` with the exact difference, so fixing the skew later flips the test.
**Neural baseline.** Same npz, same `splits.csv`, same 16-or-64-step features as `baselines.py`; model chosen on val macro-F1 only; test scored once; bootstrap CIs from existing function; paired difference against logreg. Output `dataset_v2/deep_baseline_results.json` only when actually run.

### Acceptance Criteria
- [ ] No name shadows a package: `git ls-files | grep -E "/(onnx|test)\.py$"` returns nothing; `grep -rn "onnx.py\|test.py" training README.md` finds no stale reference.
- [ ] Empty file removed: `git ls-files -s | grep e69de29` lists none under `training/`.
- [ ] Duplicate JSON removed and `cmp` of the remaining copy still equals the original blob (`git diff --stat` shows deletions only for the bridge copies).
- [ ] `grep -rn "quaternion" training --include=*.py -l` shows one module; every import resolves: `python -c "import ast,sys"` over the files plus `python -m compileall training/components/ml/src` exits 0.
- [ ] `train.py` re-raises unexpected exceptions, and returns `-inf` only for a narrow, logged class (documented in the code comment). Check: a unit test or manual run with a bad config exits non-zero.
- [ ] `MODELS.md` lists every `*.onnx` in `git ls-files` with sha256 (`sha256sum`) and size; values match command output; unknown fields (classes, accuracy, training run) say "unknown", not guessed.
- [ ] `node parity/js_runner.mjs parity/fixtures/sample.json` prints the JS-preprocessed tensor shape and values; `python -m pytest parity -q` passes (with strict xfail for documented differences).
- [ ] `docs/ARCHITECTURE.md` has a mermaid diagram and a preprocessing comparison table whose Unity/bridge/JS cells each cite a file and line.
- [ ] `ruff check dataset_v2 parity` and `python -m pytest -q dataset_v2` pass on Python 3.10 without torch (deep tests skipped, not failed).
- [ ] With torch installed: `python -m pytest -q dataset_v2/tests/test_deep_baseline.py` passes on synthetic data.
- [ ] README results table contains only numbers present in `dataset_v2/baselines_results.json`; any neural row is present only if `deep_baseline_results.json` exists, produced by a run whose command and npz hash are recorded. Check: script or manual diff of table against JSON.
- [ ] `git status` shows no `.npz`, recordings or raw data added; nothing committed.

### Risks
- Renames break Hydra/launch paths: grep all references (list above) and run `python -m compileall`; do not execute the Hydra jobs, they need the private datasets.
- Parity fixture is controller data, while `dataset_v2` is hand data; the shipped ONNX input contract is undocumented. Mitigation: compare preprocessing only, not model output; label MODELS.md fields unknown.
- JS imports `../build/three.module.js`, absent in repo. Mitigation: Node runner maps the import to `three` from npm (dev-only dependency in `parity/package.json`, pinned) via a loader or a copy of the transform function; if the module cannot be imported unchanged, record that as a finding rather than rewriting the engine.
- Neural result cannot be produced without recordings (kept local by design). Mitigation: ship code, tests and a "not yet run" line; user runs it locally, then we paste real numbers.
- Changing `train.py` error handling alters Nevergrad sweep behaviour. Mitigation: keep `-inf` for sklearn fit failures only, with the exception type logged.
- Deleting `evaluation.py`/`model1.onnx`: confirm unreferenced first (grep found no reference to `model1`; `evaluation.py` is referenced only as README text).

### Reviews
- `ml-systems-reviewer` on the neural baseline and `train.py` change (leakage, split use, test scored once, seeding).
- `platform-engineer` on CI/Node/Windows path changes and the renames.

### Implementation Order
1. **Cleanup** (no behaviour change): renames, delete empty/duplicate files, merge quaternion, update refs. Done when criteria 1-4 pass.
2. **Registry and docs skeleton**: MODELS.md (from `sha256sum`), `train.py` exception handling. Criteria 5-6.
3. **Parity harness**: fixture, Python reference, Node runner, test, ARCHITECTURE table. Read first: `controller_transform.py`, `webxr/gesture_engine.js`, both `MLNET…Inference.cs`. Criteria 7-8.
4. **Neural baseline**: `deep_baseline.py`, tests, CI optional torch job. Read first: `baselines.py`, `loader.py`. Criteria 9-10.
5. **README**: results table, diagram link, resolve TODOs that phases 1-4 answered; leave the rest as TODO. Criteria 11-12.

### Open Questions
1. Do you have the recordings locally, and may I run `baselines`/`deep_baseline` on them? Without that, the README can only show the existing sklearn numbers.
2. Which ONNX is the shipped model: the 1.0 MB `unity/` one or the 231 KB one used by JS? I cannot tell from the repo, so MODELS.md will say unknown unless you answer.
3. Is the JS skew (z flip, timestamp channel) intentional for the model it loads? The test will record it as a difference either way.
4. Allow a dev-only `parity/package.json` (three.js pinned)? Alternative is no JS execution and a table only.
5. Should the bridge copies of the two JSON files be deleted, or kept for Unity-side relative paths?
6. Assumption: the plan is saved in `docs/plans/` because `docs/` exists.

Approve (or answer the open questions) and I will hand phase 1 to the `developer` agent.
