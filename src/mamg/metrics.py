"""Classification metrics. Headline: macro-F1 and per-class recall (invariant 7).
Plain accuracy is deliberately not computed."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, int), np.asarray(y_pred, int)), 1)
    return cm


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[str],
                    malignant: Sequence[str]) -> dict[str, float]:
    """Per-class precision/recall/F1 over ALL classes (absent class -> 0, matching
    sklearn's zero_division=0 with labels=range(C)), macro-F1, balanced accuracy, and
    two malignant summaries:
      malignant_recall_macro : mean recall of BCC, MEL, SCC
      malignant_sensitivity  : fraction of malignant images predicted as ANY malignant class
    """
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    C = len(classes)
    cm = confusion(y_true, y_pred, C)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1).astype(float), cm.sum(0).astype(float)
    recall = np.divide(tp, support, out=np.zeros(C), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(C), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(C), where=denom > 0)
    out = {"n": int(len(y_true)), "macro_f1": float(f1.mean()),
           "balanced_accuracy": float(recall[support > 0].mean()) if (support > 0).any() else 0.0}
    for i, c in enumerate(classes):
        out[f"recall_{c}"] = float(recall[i])
        out[f"f1_{c}"] = float(f1[i])
    mal = np.array([classes.index(c) for c in malignant])
    out["malignant_recall_macro"] = float(recall[mal].mean())
    is_mal_true = np.isin(y_true, mal)
    out["malignant_sensitivity"] = (float(np.isin(y_pred[is_mal_true], mal).mean())
                                    if is_mal_true.any() else 0.0)
    return out


HEADLINE = ("macro_f1", "balanced_accuracy", "malignant_recall_macro", "malignant_sensitivity")
