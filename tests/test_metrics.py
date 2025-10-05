import numpy as np
from src.metrics import compute_metrics, compute_pr_auc, find_best_threshold_per_label


def test_compute_metrics():
    y_true = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1]
    ])
    
    y_pred = np.array([
        [0.9, 0.1, 0.8],
        [0.2, 0.9, 0.1],
        [0.8, 0.7, 0.3],
        [0.1, 0.2, 0.9]
    ])
    
    label_names = ['label1', 'label2', 'label3']
    
    metrics = compute_metrics(y_true, y_pred, label_names, threshold=0.5)
    
    assert 'micro_f1' in metrics, "Should compute micro-F1"
    assert 'macro_f1' in metrics, "Should compute macro-F1"
    assert 0 <= metrics['micro_f1'] <= 1, "Micro-F1 should be between 0 and 1"
    assert 0 <= metrics['macro_f1'] <= 1, "Macro-F1 should be between 0 and 1"
    
    for label in label_names:
        assert f'{label}_f1' in metrics, f"Should compute F1 for {label}"
    
    print("✓ Compute metrics test passed")


def test_compute_pr_auc():
    y_true = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1]
    ])
    
    y_scores = np.array([
        [0.9, 0.1, 0.8],
        [0.2, 0.9, 0.1],
        [0.8, 0.7, 0.3],
        [0.1, 0.2, 0.9]
    ])
    
    macro_pr_auc, per_label_auc = compute_pr_auc(y_true, y_scores)
    
    assert 0 <= macro_pr_auc <= 1, "Macro PR-AUC should be between 0 and 1"
    assert len(per_label_auc) == y_true.shape[1], "Should compute PR-AUC for each label"
    
    for auc_val in per_label_auc:
        assert 0 <= auc_val <= 1, "Per-label PR-AUC should be between 0 and 1"
    
    print("✓ Compute PR-AUC test passed")


def test_find_best_threshold():
    y_true = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1]
    ])
    
    y_scores = np.array([
        [0.9, 0.1, 0.8],
        [0.2, 0.9, 0.1],
        [0.8, 0.7, 0.3],
        [0.1, 0.2, 0.9]
    ])
    
    thresholds = find_best_threshold_per_label(y_true, y_scores)
    
    assert len(thresholds) == y_true.shape[1], "Should find threshold for each label"
    
    for t in thresholds:
        assert 0 < t < 1, "Thresholds should be between 0 and 1"
    
    print("✓ Find best threshold test passed")


if __name__ == '__main__':
    test_compute_metrics()
    test_compute_pr_auc()
    test_find_best_threshold()
    print("\nAll metric tests passed!")
