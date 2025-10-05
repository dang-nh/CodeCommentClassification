import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc

from src.utils import load_json


def plot_pr_curves(y_true, y_scores, label_names, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    all_precision = []
    all_recall = []
    
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue
        
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, alpha=0.3, linewidth=1)
        all_precision.extend(precision)
        all_recall.extend(recall)
    
    micro_precision, micro_recall, _ = precision_recall_curve(
        y_true.ravel(), y_scores.ravel()
    )
    micro_auc = auc(micro_recall, micro_precision)
    
    plt.plot(
        micro_recall, micro_precision,
        color='red', linewidth=2,
        label=f'Micro-average (AUC={micro_auc:.3f})'
    )
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves (All Labels)', fontsize=14)
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pr_curves_all.png', dpi=300)
    plt.close()
    
    for i, label_name in enumerate(label_names):
        if y_true[:, i].sum() == 0:
            continue
        
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, linewidth=2, label=f'{label_name} (AUC={pr_auc:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve: {label_name}', fontsize=14)
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pr_curve_{label_name}.png', dpi=300)
        plt.close()
    
    print(f"Saved PR curves to {output_dir}")


def plot_label_distribution(y_true, label_names, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    label_counts = y_true.sum(axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(label_names)), label_counts)
    plt.xticks(range(len(label_names)), label_names, rotation=45, ha='right')
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Label Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/label_distribution.png', dpi=300)
    plt.close()
    
    print(f"Saved label distribution plot to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate plots for evaluation')
    parser.add_argument('--preds', type=str, required=True, help='Path to predictions CSV')
    parser.add_argument('--labels', type=str, required=True, help='Path to ground truth labels (.npy)')
    parser.add_argument('--out', type=str, required=True, help='Output directory for plots')
    parser.add_argument('--label_names', type=str, default=None, help='Path to label names JSON')
    args = parser.parse_args()
    
    print(f"Loading predictions from {args.preds}...")
    preds_df = pd.read_csv(args.preds)
    
    print(f"Loading labels from {args.labels}...")
    y_true = np.load(args.labels)
    
    pred_cols = [col for col in preds_df.columns if col.startswith('pred_')]
    label_names = [col.replace('pred_', '') for col in pred_cols]
    
    if args.label_names:
        label_names_from_file = load_json(args.label_names)
        if len(label_names_from_file) == len(label_names):
            label_names = label_names_from_file
    
    y_scores = preds_df[pred_cols].values
    
    print(f"Generating PR curves...")
    plot_pr_curves(y_true, y_scores, label_names, args.out)
    
    print(f"Generating label distribution plot...")
    plot_label_distribution(y_true, label_names, args.out)
    
    print(f"\nAll plots saved to {args.out}")


if __name__ == '__main__':
    main()
