# Baselines (real run)

Session-grouped split (seed_used 3; sessions {'train': 39, 'val': 9, 'test': 9}; recordings {'train': 5185, 'val': 993, 'test': 1004}; 7,182 in-vocabulary recordings, the loose-root files form one pseudo-session `(root)`). Features: 64-step linear resample stored in the npz, subsampled to 16 steps, left+right hand local joint positions and HMD-relative root position (150 channels -> 2,400 features), standardised on train. Hyper-parameter chosen on val macro-F1, test scored once per model. Logistic regression hit the 300-iteration limit (ConvergenceWarning), so it is under-trained. Note the loose `Teleportation.json` and `Zoom.json` contain `class_name` LeftInfinity (see DATA_CARD).

| model | selected | val macro-F1 | test accuracy | test macro-F1 |
|---|---|---|---|---|
| logreg | {'C': 0.1} | 0.420 | 0.371 | 0.371 |
| lda | {'shrinkage': 0.1} | 0.388 | 0.355 | 0.359 |

Chance: majority class (train) test accuracy 0.030; uniform 1/24 = 0.042.

## Per-class test recall

| class | logreg | lda |
|---|---|---|
| LeftClick | 0.00 | 0.38 |
| LeftClose | 0.31 | 0.06 |
| LeftGrab | 0.04 | 0.07 |
| LeftHome | 0.17 | 0.17 |
| LeftInfinity | 0.50 | 0.41 |
| LeftOpenPalm | 0.05 | 0.05 |
| LeftPush | 0.83 | 0.48 |
| LeftRotate | 0.40 | 0.44 |
| LeftPoint | 0.00 | 0.00 |
| LeftScroll | 0.76 | 0.56 |
| LeftBack | 0.35 | 0.28 |
| RightClick | 0.13 | 0.18 |
| RightClose | 0.41 | 0.41 |
| RightGrab | 0.06 | 0.02 |
| RightHome | 0.02 | 0.06 |
| RightInfinity | 0.78 | 0.65 |
| RightOpenPalm | 0.20 | 0.31 |
| RightPush | 0.98 | 0.87 |
| RightRotate | 0.10 | 0.28 |
| RightPoint | 0.21 | 0.31 |
| RightScroll | 0.57 | 0.43 |
| RightBack | 0.32 | 0.32 |
| Teleportation | 0.87 | 1.00 |
| Zoom | 0.78 | 0.68 |
