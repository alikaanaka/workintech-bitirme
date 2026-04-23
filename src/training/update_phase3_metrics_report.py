"""Update Phase 3 LSTM/Ensemble metrics in metrics_comparison.md."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

from src.config import config


def _compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | str]:
    auc = float(roc_auc_score(y_true, y_score))
    gini = float(2.0 * auc - 1.0)
    ks = float(ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic)
    pr_auc = float(average_precision_score(y_true, y_score))
    y_pred = (y_score >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc": auc,
        "gini": gini,
        "ks": ks,
        "pr_auc": pr_auc,
        "f1": f1,
        "cm_summary": f"TP={int(tp)}, FP={int(fp)}, TN={int(tn)}, FN={int(fn)}",
    }


def _replace_in_section(lines: list[str], section: str, key_prefix: str, replacement: str) -> None:
    in_section = False
    for idx, line in enumerate(lines):
        if line.strip() == f"## {section}":
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section and line.startswith(key_prefix):
            lines[idx] = replacement
            return
    raise ValueError(f"Expected report line not found for section={section}, key={key_prefix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=config.MODELS_SAVED_DIR / "phase3_eval_artifacts.npz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=config.REPORTS_DIR / "metrics_comparison.md",
    )
    args = parser.parse_args()

    if not args.artifacts.exists():
        raise FileNotFoundError(
            f"Required artifact is missing: {args.artifacts}. "
            "Run src/training/train_lstm.py to generate phase3_eval_artifacts.npz first."
        )
    if not args.report.exists():
        raise FileNotFoundError(f"Report file is missing: {args.report}")

    data = np.load(args.artifacts)
    required_keys = ["y_holdout", "lstm_holdout_score", "ensemble_holdout_score"]
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise KeyError(f"Artifact file is incomplete: missing keys {missing_keys} in {args.artifacts}")

    y_holdout = data["y_holdout"].astype(int)
    lstm_holdout = data["lstm_holdout_score"].astype(float)
    ensemble_holdout = data["ensemble_holdout_score"].astype(float)

    lstm_metrics = _compute_metrics(y_holdout, lstm_holdout, args.threshold)
    ensemble_metrics = _compute_metrics(y_holdout, ensemble_holdout, args.threshold)

    lines = args.report.read_text(encoding="utf-8").splitlines()

    _replace_in_section(lines, "LSTM", "- Gini:", f"- Gini: {lstm_metrics['gini']:.6f}")
    _replace_in_section(lines, "LSTM", "- KS:", f"- KS: {lstm_metrics['ks']:.6f}")
    _replace_in_section(lines, "LSTM", "- PR-AUC:", f"- PR-AUC: {lstm_metrics['pr_auc']:.6f}")
    _replace_in_section(lines, "LSTM", "- F1:", f"- F1: {lstm_metrics['f1']:.6f} (@ threshold {args.threshold:.2f})")
    _replace_in_section(
        lines,
        "LSTM",
        "- Confusion matrix summary:",
        f"- Confusion matrix summary: {lstm_metrics['cm_summary']}",
    )

    _replace_in_section(lines, "Ensemble", "- Gini:", f"- Gini: {ensemble_metrics['gini']:.6f}")
    _replace_in_section(lines, "Ensemble", "- KS:", f"- KS: {ensemble_metrics['ks']:.6f}")
    _replace_in_section(lines, "Ensemble", "- PR-AUC:", f"- PR-AUC: {ensemble_metrics['pr_auc']:.6f}")
    _replace_in_section(
        lines,
        "Ensemble",
        "- F1:",
        f"- F1: {ensemble_metrics['f1']:.6f} (@ threshold {args.threshold:.2f})",
    )
    _replace_in_section(
        lines,
        "Ensemble",
        "- Confusion matrix summary:",
        f"- Confusion matrix summary: {ensemble_metrics['cm_summary']}",
    )

    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Phase 3 metrics updated successfully.")
    print(f"LSTM holdout AUC={lstm_metrics['auc']:.6f}, Gini={lstm_metrics['gini']:.6f}, KS={lstm_metrics['ks']:.6f}")
    print(
        f"Ensemble holdout AUC={ensemble_metrics['auc']:.6f}, "
        f"Gini={ensemble_metrics['gini']:.6f}, KS={ensemble_metrics['ks']:.6f}"
    )


if __name__ == "__main__":
    main()
