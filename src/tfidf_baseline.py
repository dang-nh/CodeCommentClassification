import argparse
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

from src.data import load_raw_data
from src.labels import LabelManager
from src.metrics import compute_metrics, compute_pr_auc
from src.utils import load_config, load_json, set_seed, save_json


def main():
    parser = argparse.ArgumentParser(description='TF-IDF baseline for multi-label classification')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--classifier', type=str, default=None, choices=['svm', 'logreg'], 
                        help='Classifier type (overrides config)')
    parser.add_argument('--fold', type=int, default=0, help='Fold number to train')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--use_test', action='store_true', help='Evaluate on test set instead of validation')
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(123)
    
    classifier_type = args.classifier if args.classifier else config.get('model_type', 'svm')
    
    if args.out_dir:
        output_dir = args.out_dir
    else:
        output_dir = os.path.join(config['logging']['output_dir'], f'tfidf_{classifier_type}_fold_{args.fold}')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"TF-IDF + {classifier_type.upper()} Baseline")
    print("=" * 60)
    print()
    
    print(f"Loading data from {config['data']['raw_path']}...")
    df = load_raw_data(config['data']['raw_path'])
    splits = load_json(config['data']['split_file'])
    
    label_manager = LabelManager(splits['label_names'])
    print(f"Number of labels: {label_manager.get_num_labels()}")
    print(f"Labels: {label_manager.label_names}")
    print()
    
    if args.use_test:
        train_indices = []
        for fold_data in splits['folds']:
            train_indices.extend(fold_data['train_indices'])
            train_indices.extend(fold_data['val_indices'])
        test_indices = splits['test_indices']
        
        train_df = df.iloc[train_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)
        
        train_labels = label_manager.encode([
            label_manager.parse_label_string(s) for s in train_df['labels']
        ])
        test_labels = label_manager.encode([
            label_manager.parse_label_string(s) for s in test_df['labels']
        ])
        
        eval_df = test_df
        eval_labels = test_labels
        print(f"Using test set: Train={len(train_df)}, Test={len(test_df)}")
    else:
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
        
        eval_df = val_df
        eval_labels = val_labels
        print(f"Using fold {args.fold}: Train={len(train_df)}, Val={len(val_df)}")
    
    print()
    print(f"Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=config.get('max_features', 10000),
        ngram_range=tuple(config.get('ngram_range', [1, 2])),
        min_df=config.get('min_df', 2),
        max_df=config.get('max_df', 0.95),
        lowercase=True,
        stop_words='english'
    )
    
    X_train = vectorizer.fit_transform(train_df['sentence'])
    X_eval = vectorizer.transform(eval_df['sentence'])
    
    print(f"Feature dimension: {X_train.shape[1]}")
    print()
    
    if classifier_type == 'svm':
        print(f"Training Linear SVM (One-vs-Rest)...")
        base_clf = LinearSVC(
            C=config.get('svm_params', {}).get('C', 1.0),
            max_iter=config.get('svm_params', {}).get('max_iter', 1000),
            random_state=123
        )
    else:
        print(f"Training Logistic Regression (One-vs-Rest)...")
        base_clf = LogisticRegression(
            C=config.get('logreg_params', {}).get('C', 1.0),
            max_iter=config.get('logreg_params', {}).get('max_iter', 1000),
            solver=config.get('logreg_params', {}).get('solver', 'lbfgs'),
            random_state=123
        )
    
    clf = OneVsRestClassifier(base_clf, n_jobs=-1)
    clf.fit(X_train, train_labels)
    print("Training complete!")
    print()
    
    print(f"Evaluating on {'test' if args.use_test else 'validation'} set...")
    
    try:
        eval_preds = clf.predict_proba(X_eval)
        if isinstance(eval_preds, list):
            eval_preds = np.column_stack(eval_preds)
    except AttributeError:
        eval_preds = clf.decision_function(X_eval)
        from scipy.special import expit
        eval_preds = expit(eval_preds)
    
    metrics = compute_metrics(eval_labels, eval_preds, label_manager.label_names)
    pr_auc, per_label_auc = compute_pr_auc(eval_labels, eval_preds)
    
    print()
    print("=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"  Micro-F1:    {metrics['micro_f1']:.4f}")
    print(f"  Macro-F1:    {metrics['macro_f1']:.4f}")
    print(f"  Macro PR-AUC: {pr_auc:.4f}")
    print()
    
    per_label_metrics = []
    for i, label_name in enumerate(label_manager.label_names):
        per_label_metrics.append({
            'label': label_name,
            'precision': metrics[f'{label_name}_precision'],
            'recall': metrics[f'{label_name}_recall'],
            'f1': metrics[f'{label_name}_f1'],
            'pr_auc': per_label_auc[i],
            'support': int(eval_labels[:, i].sum())
        })
    
    per_label_df = pd.DataFrame(per_label_metrics)
    per_label_csv = os.path.join(output_dir, 'metrics_per_label.csv')
    per_label_df.to_csv(per_label_csv, index=False)
    print(f"Saved per-label metrics to: {per_label_csv}")
    
    summary = {
        'classifier': classifier_type,
        'fold': args.fold if not args.use_test else 'all',
        'split': 'test' if args.use_test else 'validation',
        'micro_f1': float(metrics['micro_f1']),
        'macro_f1': float(metrics['macro_f1']),
        'micro_precision': float(metrics['micro_precision']),
        'micro_recall': float(metrics['micro_recall']),
        'macro_precision': float(metrics['macro_precision']),
        'macro_recall': float(metrics['macro_recall']),
        'macro_pr_auc': float(pr_auc),
        'num_train': len(train_df),
        'num_eval': len(eval_df),
        'num_features': X_train.shape[1],
        'num_labels': label_manager.get_num_labels()
    }
    
    summary_json = os.path.join(output_dir, 'metrics_summary.json')
    save_json(summary, summary_json)
    print(f"Saved summary metrics to: {summary_json}")
    
    predictions_df = pd.DataFrame({
        'id': eval_df['id'].values,
        'sentence': eval_df['sentence'].values,
        'true_labels': eval_df['labels'].values
    })
    
    for i, label_name in enumerate(label_manager.label_names):
        predictions_df[f'pred_{label_name}'] = eval_preds[:, i]
    
    predictions_csv = os.path.join(output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_csv, index=False)
    print(f"Saved predictions to: {predictions_csv}")
    
    print()
    print("Top 5 labels by F1:")
    top_labels = per_label_df.nlargest(5, 'f1')[['label', 'f1', 'support']]
    print(top_labels.to_string(index=False))
    
    print()
    print("Bottom 5 labels by F1:")
    bottom_labels = per_label_df.nsmallest(5, 'f1')[['label', 'f1', 'support']]
    print(bottom_labels.to_string(index=False))
    
    print()
    print("=" * 60)
    print("Baseline complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()