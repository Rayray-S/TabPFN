"""Fraud fine-tuning ablation template for TabPFN.

This script is intentionally data-agnostic:
you must provide `X_train, y_train, X_test, y_test` with a binary label.

AUC/NDCG-oriented ablation dimensions:
- loss: CE vs weighted CE vs focal
- early-stopping metric: prioritize roc_auc
- inference calibration toggles: average_before_softmax / balance_probabilities
- optimizer choice: adamw / adam / rmsprop
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier


# ========================= MOD 1 =========================
# Added ndcg_at_k + ndcg@1% computation, and switched summary ranking
# from AUPRC to AUC first (then NDCG@1%).
# =========================================================
def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Compute NDCG@k for binary relevance labels."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    if k <= 0:
        return 0.0

    order = np.argsort(-y_score)
    y_sorted = y_true[order][:k]

    dcg = 0.0
    for i, rel in enumerate(y_sorted):
        dcg += rel / math.log2(i + 2)

    ideal = np.sort(y_true)[::-1][:k]
    idcg = 0.0
    for i, rel in enumerate(ideal):
        idcg += rel / math.log2(i + 2)

    return float(dcg / idcg) if idcg > 0 else 0.0


@dataclass(frozen=True)
class Variant:
    name: str
    loss_type: Literal["ce", "weighted_ce", "focal"]
    eval_metric: Literal["roc_auc", "log_loss", "auprc", "precision_at_1pct"]
    average_before_softmax: bool | None = None
    balance_probabilities: bool | None = None
    optimizer_name: Literal["adamw", "adam", "rmsprop"] = "adamw"
    focal_gamma: float = 2.0
    weight_clamp_max: float = 50.0


def precision_at_ratio(y_true: np.ndarray, y_score: np.ndarray, ratio: float = 0.01):
    """Precision@Top-k where k=ceil(n*ratio) on a single global ranking."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    n = len(y_true)
    k = max(1, int(math.ceil(n * ratio)))
    order = np.argsort(-y_score)
    top_idx = order[:k]
    y_top = y_true[top_idx]
    capture = int(y_top.sum())
    total_pos = int(y_true.sum())
    prec = capture / k if k > 0 else 0.0
    sens = capture / total_pos if total_pos > 0 else 0.0
    ndcg = ndcg_at_k(y_true, y_score, k)
    return prec, sens, ndcg, k


def evaluate_binary_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Compute AUC, AUPRC, Precision@1%, Sensitivity@1%, NDCG@1%, F1."""
    y_true = np.asarray(y_true).astype(int)
    # TabPFN binary proba convention: class "1" is the second column.
    if y_proba.ndim != 2 or y_proba.shape[1] < 2:
        raise ValueError("Expected y_proba with shape (n_samples, 2).")
    scores = y_proba[:, 1]

    auc = float(roc_auc_score(y_true, scores))
    auprc = float(average_precision_score(y_true, scores))
    precision_at_1pct, sensitivity_at_1pct, ndcg_1pct, _ = precision_at_ratio(
        y_true, scores, ratio=0.01
    )

    # Optional: F1 using a fixed threshold (you may tune this separately).
    y_pred = (scores >= 0.5).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "AUC": auc,
        "NDCG@1%": ndcg_1pct,
        "AUPRC": auprc,
        "Precision@1%": precision_at_1pct,
        "Sensitivity@1%": sensitivity_at_1pct,
        "F1": f1,
    }


def run_ablation(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    *,
    variants: list[Variant],
    num_epochs: int = 30,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    random_state: int = 0,
    n_estimators_finetune: int = 2,
    n_estimators_validation: int = 2,
    n_estimators_final_inference: int = 2,
):
    """Run ablation variants and print compact results.

    Ranking in summary is AUC-first, then NDCG@1%.
    """
    results: list[dict[str, float | str]] = []

    for v in variants:
        print(f"\n==== Variant: {v.name} ====")
        clf = FinetunedTabPFNClassifier(
            device="cuda",
            epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            random_state=random_state,
            n_estimators_finetune=n_estimators_finetune,
            n_estimators_validation=n_estimators_validation,
            n_estimators_final_inference=n_estimators_final_inference,
            # ========================= MOD 2 =========================
            # Explicitly pass optimizer/loss knobs that were added in
            # forked finetuned classifier implementation.
            # =========================================================
            loss_type=v.loss_type,
            focal_gamma=v.focal_gamma,
            weight_clamp_max=v.weight_clamp_max,
            eval_metric=v.eval_metric,
            average_before_softmax=v.average_before_softmax,
            balance_probabilities=v.balance_probabilities,
            optimizer_name=v.optimizer_name,
        )
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)
        metrics = evaluate_binary_metrics(np.asarray(y_test), y_proba)
        row = {"Variant": v.name, **metrics}
        results.append(row)  # type: ignore[arg-type]
        print(metrics)

    # ========================= MOD 3 =========================
    # AUC-first ranking (primary), NDCG@1% secondary for tie-break.
    # =========================================================
    results_sorted = sorted(
        results,
        key=lambda r: (float(r.get("AUC", 0.0)), float(r.get("NDCG@1%", 0.0))),
        reverse=True,
    )
    print("\n==== Summary (sorted by AUC, then NDCG@1%) ====")
    for row in results_sorted:
        print(row)


def main() -> None:
    # ========================= MOD 4 =========================
    # AUC/NDCG-oriented default variants.
    # You should replace with your real data loading and call run_ablation.
    # =========================================================
    variants = [
        Variant(name="baseline_ce_auc", loss_type="ce", eval_metric="roc_auc"),
        Variant(
            name="weighted_ce_auc",
            loss_type="weighted_ce",
            eval_metric="roc_auc",
            weight_clamp_max=20.0,
        ),
        Variant(
            name="focal_auc_g1.5",
            loss_type="focal",
            eval_metric="roc_auc",
            focal_gamma=1.5,
            weight_clamp_max=20.0,
        ),
    ]

    raise NotImplementedError(
        "Please provide X_train, y_train, X_test, y_test (binary labels 0/1), "
        "then call run_ablation(..., variants=variants)."
    )


if __name__ == "__main__":
    main()
