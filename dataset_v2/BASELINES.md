# Baselines (real run)

Session-grouped split (seed_used 3; {'train': 40, 'val': 8, 'test': 8} sessions; {'train': 4994, 'val': 1115, 'test': 1049} recordings). Features: 64-step linear resample stored in the npz, subsampled to 16 steps, left+right hand local joint positions and HMD-relative root position (150 channels -> 2,400 features), standardised on train. Hyper-parameter chosen on val macro-F1, test scored once per model. Logistic regression hit the 300-iteration limit (ConvergenceWarning), so it is under-trained.

| model | selected | val macro-F1 | test accuracy | test macro-F1 |
|---|---|---|---|---|
| logreg | {'C': 0.1} | 0.467 | 0.421 | 0.425 |
| lda | {'shrinkage': 0.5} | 0.426 | 0.376 | 0.357 |

Chance: majority class (train) test accuracy 0.039; uniform 1/24 = 0.042.

## Per-class test recall

| class | logreg | lda |
|---|---|---|
| LeftClick | 0.28 | 0.38 |
| LeftClose | 0.42 | 0.27 |
| LeftGrab | 0.21 | 0.26 |
| LeftHome | 0.00 | 0.02 |
| LeftInfinity | 0.50 | 0.32 |
| LeftOpenPalm | 0.33 | 0.35 |
| LeftPush | 0.88 | 0.73 |
| LeftRotate | 0.23 | 0.21 |
| LeftPoint | 0.23 | 0.21 |
| LeftScroll | 0.60 | 0.76 |
| LeftBack | 0.49 | 0.49 |
| RightClick | 0.28 | 0.28 |
| RightClose | 0.34 | 0.16 |
| RightGrab | 0.32 | 0.20 |
| RightHome | 0.00 | 0.00 |
| RightInfinity | 0.65 | 0.44 |
| RightOpenPalm | 0.03 | 0.00 |
| RightPush | 0.57 | 0.00 |
| RightRotate | 0.64 | 0.80 |
| RightPoint | 0.12 | 0.34 |
| RightScroll | 0.67 | 0.46 |
| RightBack | 0.43 | 0.43 |
| Teleportation | 0.90 | 1.00 |
| Zoom | 1.00 | 0.89 |
