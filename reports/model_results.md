# Model Results

Bu dokuman, Phase 2 LightGBM baseline calistirmasindan elde edilen sonuclari saklar.

## Run Configuration

- Pipeline: `python run.py --mode train-lgbm`
- Split stratejisi: tek split (`train/validation/holdout`)
- Test size: `0.2` (holdout)
- Validation size (train icinden): `0.2`
- Cross-validation: yok (Phase 2 geregi)
- Grid search / hyperparameter arama: yok (Phase 2 geregi)

## LightGBM Parameters

- objective: `binary`
- metric: `auc`
- boosting_type: `gbdt`
- learning_rate: `0.05`
- num_leaves: `64`
- min_child_samples: `100`
- reg_alpha: `0.1`
- reg_lambda: `0.1`
- feature_fraction: `0.8`
- bagging_fraction: `0.8`
- bagging_freq: `5`
- n_jobs: `-1`
- seed: `42`
- num_boost_round: `1000`
- early_stopping_rounds: `50`
- decision_threshold: `0.22`

## Metrics Snapshot

- Validation AUC: `0.887870`
- Holdout Test AUC: `0.893654`
- Gini: `0.787307`
- KS: `0.650290`
- PR-AUC: `0.501409`
- F1@0.22: `0.281332`
- Brier Score: `0.103303`
- Best Iteration: `1000`

## Generated Artifacts

- Model: `models_saved/lgbm_model.pkl`
- Top features: `data/artifacts/feature_lists/top50_features.json`
- Metrics summary: `reports/metrics_comparison.md`

## Notes

- Bu sonuclar baseline seviyesindedir.
- Tek split yaklasimi hizli ve sade olsa da metrik varyansini CV kadar iyi yansitmaz.

## CatBoost (CV5, Threshold=0.30)

- Pipeline: `python run.py --mode train-catboost-cv5 --threshold 0.30`
- Cross-validation: `5-fold stratified` (train split uzerinde)
- Holdout: `0.2` (ayni split stratejisi)
- Saved model: `models_saved/catboost_cv5_threshold030_20260420.pkl`
- Detailed report: `reports/metrics_catboost_cv5.md`

### CV Validation Metrics (mean +/- std)

- Validation AUC: `0.802947 +/- 0.001139`
- Gini: `0.605893 +/- 0.002279`
- KS: `0.459606 +/- 0.001127`
- PR-AUC: `0.332560 +/- 0.002205`
- Brier Score: `0.063802 +/- 0.000070`
- Precision: `0.474967 +/- 0.008661`
- Recall: `0.209768 +/- 0.003681`
- F1-Score: `0.291002 +/- 0.004881`

### Holdout Metrics

- Holdout AUC: `0.803056`
- Confusion Matrix: `[[110698, 2377], [7902, 2028]]`

## CatBoost (CV5, Threshold=0.30, Train Feature Only)

- Pipeline: `python run.py --mode train-catboost-train-feature-cv5 --threshold 0.30`
- Data source: `data/train_feature.csv` (previous_application birlestirmesi yok)
- Cross-validation: `5-fold stratified` (train split uzerinde)
- Holdout: `0.2`
- Class weight: `scale_pos_weight` kullanildi (`final train split: 11.387116`)
- Saved model: `models_saved/catboost_train_feature_cv5_threshold030_20260420.pkl`
- Detailed report: `reports/metrics_catboost_train_feature_cv5.md`

### CV Validation Metrics (mean +/- std)

- Validation AUC: `0.811428 +/- 0.002035`
- Gini: `0.622856 +/- 0.004071`
- KS: `0.474151 +/- 0.003606`
- PR-AUC: `0.302859 +/- 0.003382`
- Brier Score: `0.170431 +/- 0.000525`
- Precision: `0.132221 +/- 0.000516`
- Recall: `0.918051 +/- 0.004023`
- F1-Score: `0.231151 +/- 0.000909`

### Holdout Metrics

- Holdout AUC: `0.811567`
- Confusion Matrix: `[[53650, 59425], [831, 9099]]`

## XGBoost (CV5, Threshold=0.30, Train Feature Only)

- Pipeline: `python run.py --mode train-xgboost-train-feature-cv5 --threshold 0.30`
- Data source: `data/train_feature.csv` (previous_application birlestirmesi yok)
- Cross-validation: `5-fold stratified` (train split uzerinde)
- Holdout: `0.2`
- Class weight: `scale_pos_weight` kullanildi (`final train split: 11.387116`)
- Saved model: `models_saved/xgboost_train_feature_cv5_threshold030_20260420.pkl`
- Detailed report: `reports/metrics_xgboost_train_feature_cv5.md`
- Precision-recall chart: `reports/xgboost_train_feature_precision_recall_curve.png`

### CV Validation Metrics (mean +/- std)

- Validation AUC: `0.859756 +/- 0.001977`
- Gini: `0.719511 +/- 0.003955`
- KS: `0.570399 +/- 0.002946`
- PR-AUC: `0.399177 +/- 0.004248`
- Brier Score: `0.133177 +/- 0.000413`
- Precision: `0.165490 +/- 0.000548`
- Recall: `0.910473 +/- 0.003405`
- F1-Score: `0.280073 +/- 0.000865`

### Holdout Metrics

- Holdout AUC: `0.859936`
- Confusion Matrix: `[[67946, 45129], [881, 9049]]`

## LightGBM (CV5, Threshold=0.30, Train Feature Only)

- Pipeline: `python run.py --mode train-lgbm-train-feature-cv5 --threshold 0.30`
- Data source: `data/train_feature.csv` (previous_application birlestirmesi yok)
- Cross-validation: `5-fold stratified` (train split uzerinde)
- Holdout: `0.2`
- Class weight: `scale_pos_weight` kullanildi (`final train split: 11.387116`)
- Saved model: `models_saved/lgbm_train_feature_cv5_threshold030_20260420.pkl`
- Detailed report: `reports/metrics_lgbm_train_feature_cv5.md`
- Precision-recall chart: `reports/lgbm_train_feature_precision_recall_curve.png`

### CV Validation Metrics (mean +/- std)

- Validation AUC: `0.883946 +/- 0.001786`
- Gini: `0.767892 +/- 0.003572`
- KS: `0.629468 +/- 0.003853`
- PR-AUC: `0.456115 +/- 0.005209`
- Brier Score: `0.113363 +/- 0.000600`
- Precision: `0.192320 +/- 0.001154`
- Recall: `0.907100 +/- 0.001649`
- F1-Score: `0.317354 +/- 0.001606`

### Holdout Metrics

- Holdout AUC: `0.882818`
- Confusion Matrix: `[[75457, 37618], [965, 8965]]`
