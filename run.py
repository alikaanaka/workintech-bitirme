"""CLI entrypoint for Home Credit Risk pipeline."""

import argparse

from src.config import config
from src.features.feature_pipeline import run_feature_pipeline
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Home Credit Risk System runner")
    parser.add_argument(
        "--mode",
        choices=[
            "preprocess",
            "train-lgbm",
            "train-lgbm-v2",
            "train-catboost-cv5",
            "train-catboost-train-feature-cv5",
            "train-xgboost-train-feature-cv5",
            "train-lgbm-train-feature-cv5",
            "train-lstm",
            "evaluate",
            "serve",
        ],
        required=True,
        help="Execution mode",
    )
    parser.add_argument("--host", default="0.0.0.0", help="API server host (serve mode)")
    parser.add_argument("--port", type=int, default=8000, help="API server port (serve mode)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.LGBM_V2_DEFAULT_THRESHOLD,
        help="Decision threshold for train-lgbm-v2 mode",
    )
    return parser.parse_args()


def main() -> None:
    """Run selected mode."""
    args = parse_args()
    set_global_seed(config.RANDOM_SEED)

    if args.mode == "preprocess":
        final_df = run_feature_pipeline()
        logger.info(
            "Preprocess completed | rows=%s cols=%s target_col=%s key_col=%s",
            len(final_df),
            len(final_df.columns),
            config.TARGET_COLUMN,
            config.KEY_COLUMN,
        )
    elif args.mode == "train-lgbm":
        from src.training.train_lgbm import train_lgbm_baseline

        result = train_lgbm_baseline()
        logger.info(
            "LightGBM training completed | holdout_auc=%.6f top_features=%s model=%s",
            result["holdout_metrics"]["auc_roc"],
            result["top_features_count"],
            result["model_path"],
        )
    elif args.mode == "evaluate":
        report_path = config.REPORTS_DIR / config.METRICS_REPORT_FILE_NAME
        if report_path.exists():
            logger.info("Metrics report is available at %s", report_path)
        else:
            logger.warning("Metrics report not found. Run --mode train-lgbm first.")
    elif args.mode == "train-lgbm-v2":
        from src.training.train_lgbm_v2 import train_lgbm_v2

        result = train_lgbm_v2(threshold=args.threshold)
        logger.info(
            "LightGBM v2 training completed | threshold=%.2f model=%s",
            result["threshold"],
            result["model_path"],
        )
    elif args.mode == "train-catboost-cv5":
        from src.training.train_catboost_cv import train_catboost_cv5

        result = train_catboost_cv5(threshold=args.threshold)
        logger.info(
            "CatBoost CV5 training completed | threshold=%.2f model=%s report=%s",
            result["threshold"],
            result["model_path"],
            result["report_path"],
        )
    elif args.mode == "train-catboost-train-feature-cv5":
        from src.training.train_catboost_train_feature_cv import train_catboost_train_feature_cv5

        result = train_catboost_train_feature_cv5(threshold=args.threshold)
        logger.info(
            "CatBoost train_feature CV5 completed | threshold=%.2f model=%s report=%s",
            result["threshold"],
            result["model_path"],
            result["report_path"],
        )
    elif args.mode == "train-xgboost-train-feature-cv5":
        from src.training.train_xgboost_train_feature_cv import train_xgboost_train_feature_cv5

        result = train_xgboost_train_feature_cv5(threshold=args.threshold)
        logger.info(
            "XGBoost train_feature CV5 completed | threshold=%.2f model=%s report=%s",
            result["threshold"],
            result["model_path"],
            result["report_path"],
        )
    elif args.mode == "train-lgbm-train-feature-cv5":
        from src.training.train_lgbm_train_feature_cv import train_lgbm_train_feature_cv5

        result = train_lgbm_train_feature_cv5(threshold=args.threshold)
        logger.info(
            "LightGBM train_feature CV5 completed | threshold=%.2f model=%s report=%s",
            result["threshold"],
            result["model_path"],
            result["report_path"],
        )
    elif args.mode == "train-lstm":
        from src.training.train_lstm import train_lstm_oof_cv5

        result = train_lstm_oof_cv5(threshold=args.threshold)
        logger.info(
            "LSTM OOF CV5 completed | oof_auc=%.6f holdout_auc=%.6f holdout_f1=%.6f",
            result["oof_auc_overall"],
            result["holdout_metrics"]["auc_roc"],
            result["holdout_metrics"]["f1_at_threshold"],
        )
    elif args.mode == "serve":
        import uvicorn

        logger.info("Starting API server on %s:%s", args.host, args.port)
        uvicorn.run("api.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
