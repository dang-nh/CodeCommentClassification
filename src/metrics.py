import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc
)
from typing import Dict, List, Tuple


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    threshold: float = 0.5
) -> Dict[str, float]:
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    metrics = {}
    
    metrics['micro_precision'] = precision_score(y_true, y_pred_binary, average='micro', zero_division=0)
    metrics['micro_recall'] = recall_score(y_true, y_pred_binary, average='micro', zero_division=0)
    metrics['micro_f1'] = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    
    metrics['macro_precision'] = precision_score(y_true, y_pred_binary, average='macro', zero_division=0)
    metrics['macro_recall'] = recall_score(y_true, y_pred_binary, average='macro', zero_division=0)
    metrics['macro_f1'] = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    
    per_label_precision = precision_score(y_true, y_pred_binary, average=None, zero_division=0)
    per_label_recall = recall_score(y_true, y_pred_binary, average=None, zero_division=0)
    per_label_f1 = f1_score(y_true, y_pred_binary, average=None, zero_division=0)
    
    for i, label_name in enumerate(label_names):
        metrics[f'{label_name}_precision'] = per_label_precision[i]
        metrics[f'{label_name}_recall'] = per_label_recall[i]
        metrics[f'{label_name}_f1'] = per_label_f1[i]
    
    return metrics


def compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, List[float]]:
    per_label_auc = []
    
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            per_label_auc.append(0.0)
            continue
        
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        pr_auc = auc(recall, precision)
        per_label_auc.append(pr_auc)
    
    macro_pr_auc = np.mean(per_label_auc)
    return macro_pr_auc, per_label_auc


def find_best_threshold_per_label(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    thresholds: np.ndarray = None
) -> np.ndarray:
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)
    
    num_labels = y_true.shape[1]
    best_thresholds = np.zeros(num_labels)
    
    for i in range(num_labels):
        if y_true[:, i].sum() == 0:
            best_thresholds[i] = 0.5
            continue
        
        best_f1 = 0
        best_t = 0.5
        
        for t in thresholds:
            y_pred = (y_scores[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        
        best_thresholds[i] = best_t
    
    return best_thresholds
