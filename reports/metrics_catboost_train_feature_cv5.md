# CatBoost CV5 Metrics (Train Feature Only)

Threshold: `0.30`
Scale pos weight (final train split): `11.387116`

| Metric | Value |
|---|---:|
| Validation AUC (CV mean +/- std) | 0.811428 +/- 0.002035 |
| Holdout AUC | 0.811567 |
| Gini (CV mean +/- std) | 0.622856 +/- 0.004071 |
| KS (CV mean +/- std) | 0.474151 +/- 0.003606 |
| PR-AUC (CV mean +/- std) | 0.302859 +/- 0.003382 |
| Brier Score (CV mean +/- std) | 0.170431 +/- 0.000525 |
| Precision (CV mean +/- std) | 0.132221 +/- 0.000516 |
| Recall (CV mean +/- std) | 0.918051 +/- 0.004023 |
| F1-Score (CV mean +/- std) | 0.231151 +/- 0.000909 |

## Holdout Confusion Matrix
- `[[53650, 59425], [831, 9099]]`

## Precision-Recall Chart
- Chart file: `/Users/elifcubukcu/Downloads/workintech-bitirme/reports/catboost_train_feature_precision_recall_curve.png`

## Precision-Recall by Threshold (Holdout)
| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.10 | 0.087627 | 0.996274 | 0.161086 |
| 0.20 | 0.106195 | 0.969285 | 0.191418 |
| 0.30 | 0.132786 | 0.916314 | 0.231958 |
| 0.40 | 0.164498 | 0.829909 | 0.274572 |
| 0.50 | 0.204831 | 0.719839 | 0.318915 |
| 0.60 | 0.251256 | 0.574018 | 0.349522 |
| 0.70 | 0.313782 | 0.405136 | 0.353655 |
| 0.80 | 0.394817 | 0.208661 | 0.273027 |
| 0.90 | 0.556678 | 0.049950 | 0.091674 |

## Fold Details
- Fold 1: AUC=0.811680, PR-AUC=0.306805, Precision=0.132634, Recall=0.918555, F1=0.231798
- Fold 2: AUC=0.812411, PR-AUC=0.301965, Precision=0.132640, Recall=0.921450, F1=0.231899
- Fold 3: AUC=0.807814, PR-AUC=0.297134, Precision=0.131449, Recall=0.911883, F1=0.229775
- Fold 4: AUC=0.813995, PR-AUC=0.302701, Precision=0.132631, Recall=0.922961, F1=0.231934
- Fold 5: AUC=0.811240, PR-AUC=0.305689, Precision=0.131751, Recall=0.915408, F1=0.230349

Saved model: `/Users/elifcubukcu/Downloads/workintech-bitirme/models_saved/catboost_train_feature_cv5_threshold030_20260420.pkl`