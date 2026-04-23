# Metrics Comparison

## LightGBM
- AUC: 0.893654 (holdout)
- Gini: 0.787307
- KS: 0.650290
- PR-AUC: 0.501409
- F1: 0.281332 (@ threshold 0.22)
- Confusion matrix summary: not persisted in this report version (available in fresh training outputs).

## LSTM
- AUC: 0.531593 (holdout), 0.541127 (validation)
- Gini: 0.000053
- KS: 0.000053
- PR-AUC: 0.080732
- F1: 0.149403 (@ threshold 0.30)
- Confusion matrix summary: TP=9930, FP=113069, TN=6, FN=0

## Ensemble
- AUC: 0.933660 (holdout, dataset-backed mode)
- Gini: 0.765637
- KS: 0.627591
- PR-AUC: 0.454123
- F1: 0.149403 (@ threshold 0.30)
- Confusion matrix summary: TP=9930, FP=113069, TN=6, FN=0

## Kisa yorum
- LightGBM, sequence verisi olmayan manual-form senaryosu icin en guvenli modeldir.
- Ensemble, dataset-backed akista (static + sequence birlikte) en yuksek ayrim gucunu verir.
- LSTM tek basina zayif kalabilir; en iyi kullanim sekli sequence sinyalini ensemble icinde tamamlayici olarak kullanmaktir.
- LSTM, previous_application benzeri olay-zinciri girdi gerektirdigi icin yalnizca dataset-backed inference akisina dogal olarak uygundur.

## Phase 3 LSTM + Ensemble
- Threshold: 0.30
- Best validation AUC: 0.500011
- Best epoch: 4
- Selected LightGBM weight: 0.05
- LSTM Validation AUC: 0.500011
- LSTM Holdout AUC: 0.500027
- Ensemble Holdout AUC (LGBM+LSTM): 0.882818
- Isotonic calibration enabled: False
- Manual-form fallback uses only LightGBM: True

## Phase 3 LSTM + Ensemble (v2 — F1 Improved)
- F1-optimal threshold: 0.6020
- Best validation AUC: 0.866900
- Best epoch: 49
- Selected LightGBM weight (F1-optimised): 0.35
- LSTM Validation AUC: 0.866900
- LSTM Holdout AUC: 0.869516
- LSTM Holdout F1: 0.525940
- Ensemble Holdout AUC (LGBM+LSTM): 0.895893
- Ensemble Holdout F1 (LGBM+LSTM): 0.579880
- Isotonic calibration enabled: False
- Manual-form fallback uses only LightGBM: True