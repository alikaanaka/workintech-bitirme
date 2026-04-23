# CatBoost CV5 Metrics

Threshold: `0.30`

| Metric | CatBoost (threshold=0.30) |
|---|---:|
| Validation AUC (CV mean±std) | 0.802947 +/- 0.001139 |
| Holdout AUC | 0.803056 |
| Gini (CV mean±std) | 0.605893 +/- 0.002279 |
| KS (CV mean±std) | 0.459606 +/- 0.001127 |
| PR-AUC (CV mean±std) | 0.332560 +/- 0.002205 |
| Brier Score (CV mean±std) | 0.063802 +/- 0.000070 |
| Precision (CV mean±std) | 0.474967 +/- 0.008661 |
| Recall (CV mean±std) | 0.209768 +/- 0.003681 |
| F1-Score (CV mean±std) | 0.291002 +/- 0.004881 |

## Holdout Confusion Matrix
- `[[110698, 2377], [7902, 2028]]`

## Fold Details
- Fold 1: AUC=0.802432, PR-AUC=0.334460, Precision=0.473444, Recall=0.208711, F1=0.289708
- Fold 2: AUC=0.803873, PR-AUC=0.331371, Precision=0.474702, Recall=0.210222, F1=0.291398
- Fold 3: AUC=0.801277, PR-AUC=0.329070, Precision=0.461668, Recall=0.207704, F1=0.286508
- Fold 4: AUC=0.804521, PR-AUC=0.332677, Precision=0.476107, Recall=0.205690, F1=0.287271
- Fold 5: AUC=0.802630, PR-AUC=0.335223, Precision=0.488914, Recall=0.216516, F1=0.300122

Saved model: `/Users/elifcubukcu/Downloads/workintech-bitirme/models_saved/catboost_cv5_threshold030_20260420.pkl`