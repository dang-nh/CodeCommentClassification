import argparse
import numpy as np
from src.metrics import find_best_threshold_per_label
from src.utils import save_json, load_json


def main():
    parser = argparse.ArgumentParser(description='Find optimal per-label thresholds')
    parser.add_argument('--preds', type=str, required=True, help='Path to predictions (.npy)')
    parser.add_argument('--labels', type=str, required=True, help='Path to ground truth labels (.npy)')
    parser.add_argument('--out', type=str, required=True, help='Path to output thresholds JSON')
    parser.add_argument('--label_names', type=str, default=None, help='Path to label names JSON')
    args = parser.parse_args()
    
    print(f"Loading predictions from {args.preds}...")
    y_scores = np.load(args.preds)
    
    print(f"Loading labels from {args.labels}...")
    y_true = np.load(args.labels)
    
    print(f"Finding optimal thresholds for {y_true.shape[1]} labels...")
    thresholds = find_best_threshold_per_label(y_true, y_scores)
    
    threshold_dict = {}
    if args.label_names:
        label_names = load_json(args.label_names)
        for i, name in enumerate(label_names):
            threshold_dict[name] = float(thresholds[i])
    else:
        for i in range(len(thresholds)):
            threshold_dict[f'label_{i}'] = float(thresholds[i])
    
    save_json(threshold_dict, args.out)
    print(f"Saved thresholds to {args.out}")
    
    print("\nThreshold summary:")
    print(f"  Min: {thresholds.min():.3f}")
    print(f"  Max: {thresholds.max():.3f}")
    print(f"  Mean: {thresholds.mean():.3f}")
    print(f"  Median: {np.median(thresholds):.3f}")


if __name__ == '__main__':
    main()
