# Metrics Comparison V2

Threshold: `0.30`

| Metric | v1 @threshold | v2 @threshold |
|---|---:|---:|
| Validation AUC | 0.887870 | 0.887870 |
| Holdout AUC | 0.893654 | 0.893654 |
| Gini | 0.787307 | 0.787307 |
| KS | 0.650290 | 0.650290 |
| PR-AUC | 0.501409 | 0.501409 |
| Brier Score | 0.103303 | 0.103303 |
| Precision | 0.206786 | 0.206786 |
| Recall | 0.899698 | 0.899698 |
| F1-Score | 0.336282 | 0.336282 |

## Confusion Matrices (Holdout)
- v1: `[[78805, 34270], [996, 8934]]`
- v2: `[[78805, 34270], [996, 8934]]`

Saved model: `C:\Users\meaki\Desktop\WORKINTECH-FINAL-PROJECT\models_saved\lgbm_v2_threshold030_20260420.pkl`

## CatBoost CV5 @ 0.30 (New Run)

Saved model: `models_saved/catboost_cv5_threshold030_20260420.pkl`

| Metric | v1 @0.30 | v2 @0.30 | CatBoost CV5 Validation (mean +/- std) | CatBoost Holdout |
|---|---:|---:|---:|---:|
| Validation AUC | 0.887870 | 0.887870 | 0.802947 +/- 0.001139 | - |
| Holdout AUC | 0.893654 | 0.893654 | - | 0.803056 |
| Gini | 0.787307 | 0.787307 | 0.605893 +/- 0.002279 | - |
| KS | 0.650290 | 0.650290 | 0.459606 +/- 0.001127 | - |
| PR-AUC | 0.501409 | 0.501409 | 0.332560 +/- 0.002205 | - |
| Brier Score | 0.103303 | 0.103303 | 0.063802 +/- 0.000070 | - |
| Precision | 0.206786 | 0.206786 | 0.474967 +/- 0.008661 | - |
| Recall | 0.899698 | 0.899698 | 0.209768 +/- 0.003681 | - |
| F1-Score | 0.336282 | 0.336282 | 0.291002 +/- 0.004881 | - |

CatBoost Holdout Confusion Matrix: `[[110698, 2377], [7902, 2028]]`

## CatBoost CV5 @ 0.30 (Train Feature Only)

Saved model: `models_saved/catboost_train_feature_cv5_threshold030_20260420.pkl`

| Metric | CatBoost (train+prev features) | CatBoost (train_feature only, CV mean +/- std) | CatBoost (train_feature only, holdout) |
|---|---:|---:|---:|
| Validation AUC | 0.802947 +/- 0.001139 | 0.811428 +/- 0.002035 | - |
| Holdout AUC | 0.803056 | - | 0.811567 |
| Gini | 0.605893 +/- 0.002279 | 0.622856 +/- 0.004071 | - |
| KS | 0.459606 +/- 0.001127 | 0.474151 +/- 0.003606 | - |
| PR-AUC | 0.332560 +/- 0.002205 | 0.302859 +/- 0.003382 | - |
| Brier Score | 0.063802 +/- 0.000070 | 0.170431 +/- 0.000525 | - |
| Precision | 0.474967 +/- 0.008661 | 0.132221 +/- 0.000516 | - |
| Recall | 0.209768 +/- 0.003681 | 0.918051 +/- 0.004023 | - |
| F1-Score | 0.291002 +/- 0.004881 | 0.231151 +/- 0.000909 | - |

CatBoost (train_feature only) Holdout Confusion Matrix: `[[53650, 59425], [831, 9099]]`

## XGBoost CV5 @ 0.30 (Train Feature Only)

Saved model: `models_saved/xgboost_train_feature_cv5_threshold030_20260420.pkl`
PR chart: `reports/xgboost_train_feature_precision_recall_curve.png`

| Metric | CatBoost (train_feature only, CV mean +/- std) | XGBoost (train_feature only, CV mean +/- std) | XGBoost (train_feature only, holdout) |
|---|---:|---:|---:|
| Validation AUC | 0.811428 +/- 0.002035 | 0.859756 +/- 0.001977 | - |
| Holdout AUC | - | - | 0.859936 |
| Gini | 0.622856 +/- 0.004071 | 0.719511 +/- 0.003955 | - |
| KS | 0.474151 +/- 0.003606 | 0.570399 +/- 0.002946 | - |
| PR-AUC | 0.302859 +/- 0.003382 | 0.399177 +/- 0.004248 | - |
| Brier Score | 0.170431 +/- 0.000525 | 0.133177 +/- 0.000413 | - |
| Precision | 0.132221 +/- 0.000516 | 0.165490 +/- 0.000548 | - |
| Recall | 0.918051 +/- 0.004023 | 0.910473 +/- 0.003405 | - |
| F1-Score | 0.231151 +/- 0.000909 | 0.280073 +/- 0.000865 | - |

XGBoost (train_feature only) Holdout Confusion Matrix: `[[67946, 45129], [881, 9049]]`

## LightGBM CV5 @ 0.30 (Train Feature Only)

Saved model: `models_saved/lgbm_train_feature_cv5_threshold030_20260420.pkl`
PR chart: `reports/lgbm_train_feature_precision_recall_curve.png`

| Metric | XGBoost (train_feature only, CV mean +/- std) | LightGBM (train_feature only, CV mean +/- std) | LightGBM (train_feature only, holdout) |
|---|---:|---:|---:|
| Validation AUC | 0.859756 +/- 0.001977 | 0.883946 +/- 0.001786 | - |
| Holdout AUC | - | - | 0.882818 |
| Gini | 0.719511 +/- 0.003955 | 0.767892 +/- 0.003572 | - |
| KS | 0.570399 +/- 0.002946 | 0.629468 +/- 0.003853 | - |
| PR-AUC | 0.399177 +/- 0.004248 | 0.456115 +/- 0.005209 | - |
| Brier Score | 0.133177 +/- 0.000413 | 0.113363 +/- 0.000600 | - |
| Precision | 0.165490 +/- 0.000548 | 0.192320 +/- 0.001154 | - |
| Recall | 0.910473 +/- 0.003405 | 0.907100 +/- 0.001649 | - |
| F1-Score | 0.280073 +/- 0.000865 | 0.317354 +/- 0.001606 | - |

LightGBM (train_feature only) Holdout Confusion Matrix: `[[75457, 37618], [965, 8965]]`