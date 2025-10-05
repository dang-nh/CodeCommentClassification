import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

from src.data import load_raw_data
from src.labels import LabelManager
from src.metrics import compute_metrics, compute_pr_auc
from src.utils import load_config, load_json, set_seed


def main():
    parser = argparse.ArgumentParser(description='TF-IDF baseline for multi-label classification')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--fold', type=int, default=0, help='Fold number to train')
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(123)
    
    print(f"Loading data...")
    df = load_raw_data(config['data']['raw_path'])
    splits = load_json(config['data']['split_file'])
    
    label_manager = LabelManager(splits['label_names'])
    
    fold_data = splits['folds'][args.fold]
    train_indices = fold_data['train_indices']
    val_indices = fold_data['val_indices']
    
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    
    train_labels = label_manager.encode([
        label_manager.parse_label_string(s) for s in train_df['labels']
    ])
    val_labels = label_manager.encode([
        label_manager.parse_label_string(s) for s in val_df['labels']
    ])
    
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    print(f"Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=config.get('max_features', 10000),
        ngram_range=tuple(config.get('ngram_range', [1, 2])),
        min_df=config.get('min_df', 2),
        max_df=config.get('max_df', 0.95)
    )
    
    X_train = vectorizer.fit_transform(train_df['sentence'])
    X_val = vectorizer.transform(val_df['sentence'])
    
    print(f"Feature dimension: {X_train.shape[1]}")
    
    if config.get('model_type', 'svm') == 'svm':
        print(f"Training Linear SVM...")
        base_clf = LinearSVC(
            C=config.get('svm_params', {}).get('C', 1.0),
            max_iter=config.get('svm_params', {}).get('max_iter', 1000)
        )
    else:
        print(f"Training Logistic Regression...")
        base_clf = LogisticRegression(
            C=config.get('logreg_params', {}).get('C', 1.0),
            max_iter=config.get('logreg_params', {}).get('max_iter', 1000),
            solver=config.get('logreg_params', {}).get('solver', 'lbfgs')
        )
    
    clf = OneVsRestClassifier(base_clf, n_jobs=-1)
    clf.fit(X_train, train_labels)
    
    print(f"Evaluating...")
    val_preds = clf.predict_proba(X_val)
    
    if isinstance(val_preds, list):
        val_preds = np.column_stack(val_preds)
    
    metrics = compute_metrics(val_labels, val_preds, label_manager.label_names)
    pr_auc, _ = compute_pr_auc(val_labels, val_preds)
    
    print(f"\nResults:")
    print(f"  Val Micro-F1: {metrics['micro_f1']:.4f}")
    print(f"  Val Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"  Val PR-AUC: {pr_auc:.4f}")


if __name__ == '__main__':
    main()
