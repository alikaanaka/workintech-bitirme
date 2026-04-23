# LightGBM CV5 Metrics (Train Feature Only)

Threshold: `0.30`
Scale pos weight (final train split): `11.387116`

| Metric | Value |
|---|---:|
| Validation AUC (CV mean +/- std) | 0.883946 +/- 0.001786 |
| Holdout AUC | 0.882818 |
| Gini (CV mean +/- std) | 0.767892 +/- 0.003572 |
| KS (CV mean +/- std) | 0.629468 +/- 0.003853 |
| PR-AUC (CV mean +/- std) | 0.456115 +/- 0.005209 |
| Brier Score (CV mean +/- std) | 0.113363 +/- 0.000600 |
| Precision (CV mean +/- std) | 0.192320 +/- 0.001154 |
| Recall (CV mean +/- std) | 0.907100 +/- 0.001649 |
| F1-Score (CV mean +/- std) | 0.317354 +/- 0.001606 |

## Holdout Confusion Matrix
- `[[75457, 37618], [965, 8965]]`

## Precision-Recall Chart
- Chart file: `/Users/elifcubukcu/Downloads/workintech-bitirme/reports/lgbm_train_feature_precision_recall_curve.png`

## Precision-Recall by Threshold (Holdout)
| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.10 | 0.108628 | 0.978550 | 0.195548 |
| 0.20 | 0.146360 | 0.945519 | 0.253483 |
| 0.30 | 0.192452 | 0.902820 | 0.317272 |
| 0.40 | 0.246549 | 0.848943 | 0.382122 |
| 0.50 | 0.313584 | 0.768580 | 0.445430 |
| 0.60 | 0.388759 | 0.634542 | 0.482133 |
| 0.70 | 0.487531 | 0.458711 | 0.472682 |
| 0.80 | 0.592679 | 0.231521 | 0.332971 |
| 0.90 | 0.741472 | 0.041591 | 0.078764 |

## Fold Details
- Fold 1: AUC=0.885344, PR-AUC=0.461879, Precision=0.194227, Recall=0.907981, F1=0.320002
- Fold 2: AUC=0.885479, PR-AUC=0.451252, Precision=0.192060, Recall=0.909869, F1=0.317171
- Fold 3: AUC=0.880600, PR-AUC=0.450217, Precision=0.191357, Recall=0.905211, F1=0.315928
- Fold 4: AUC=0.884579, PR-AUC=0.454642, Precision=0.192924, Recall=0.906093, F1=0.318116
- Fold 5: AUC=0.883728, PR-AUC=0.462584, Precision=0.191032, Recall=0.906344, F1=0.315554

Saved model: `/Users/elifcubukcu/Downloads/workintech-bitirme/models_saved/lgbm_train_feature_cv5_threshold030_20260420.pkl`