# XGBoost CV5 Metrics (Train Feature Only)

Threshold: `0.30`
Scale pos weight (final train split): `11.387116`

| Metric | Value |
|---|---:|
| Validation AUC (CV mean +/- std) | 0.859756 +/- 0.001977 |
| Holdout AUC | 0.859936 |
| Gini (CV mean +/- std) | 0.719511 +/- 0.003955 |
| KS (CV mean +/- std) | 0.570399 +/- 0.002946 |
| PR-AUC (CV mean +/- std) | 0.399177 +/- 0.004248 |
| Brier Score (CV mean +/- std) | 0.133177 +/- 0.000413 |
| Precision (CV mean +/- std) | 0.165490 +/- 0.000548 |
| Recall (CV mean +/- std) | 0.910473 +/- 0.003405 |
| F1-Score (CV mean +/- std) | 0.280073 +/- 0.000865 |

## Holdout Confusion Matrix
- `[[67946, 45129], [881, 9049]]`

## Precision-Recall Chart
- Chart file: `/Users/elifcubukcu/Downloads/workintech-bitirme/reports/xgboost_train_feature_precision_recall_curve.png`

## Precision-Recall by Threshold (Holdout)
| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.10 | 0.098516 | 0.986002 | 0.179134 |
| 0.20 | 0.128536 | 0.957805 | 0.226655 |
| 0.30 | 0.167024 | 0.911279 | 0.282305 |
| 0.40 | 0.212253 | 0.841893 | 0.339031 |
| 0.50 | 0.265581 | 0.738973 | 0.390735 |
| 0.60 | 0.330550 | 0.601913 | 0.426746 |
| 0.70 | 0.413400 | 0.434340 | 0.423611 |
| 0.80 | 0.523509 | 0.242195 | 0.331176 |
| 0.90 | 0.700357 | 0.059315 | 0.109368 |

## Fold Details
- Fold 1: AUC=0.858773, PR-AUC=0.402770, Precision=0.165964, Recall=0.906344, F1=0.280554
- Fold 2: AUC=0.862337, PR-AUC=0.403394, Precision=0.166271, Recall=0.915785, F1=0.281442
- Fold 3: AUC=0.856506, PR-AUC=0.392841, Precision=0.164799, Recall=0.907226, F1=0.278930
- Fold 4: AUC=0.860587, PR-AUC=0.395414, Precision=0.165339, Recall=0.911631, F1=0.279911
- Fold 5: AUC=0.860576, PR-AUC=0.401464, Precision=0.165078, Recall=0.911380, F1=0.279526

Saved model: `/Users/elifcubukcu/Downloads/workintech-bitirme/models_saved/xgboost_train_feature_cv5_threshold030_20260420.pkl`