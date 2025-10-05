import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import train_test_split

from src.labels import LabelManager
from src.utils import save_json, set_seed
from src.data import load_raw_data


def group_aware_split(
    df: pd.DataFrame,
    label_matrix: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    
    groups = df['class_id'].values
    unique_groups = np.unique(groups)
    
    group_labels = {}
    for group in unique_groups:
        mask = groups == group
        group_labels[group] = label_matrix[mask].max(axis=0)
    
    group_list = list(unique_groups)
    group_label_matrix = np.array([group_labels[g] for g in group_list])
    
    train_groups, test_groups = train_test_split(
        group_list,
        test_size=test_size,
        random_state=seed,
        stratify=group_label_matrix.sum(axis=1)
    )
    
    train_mask = df['class_id'].isin(train_groups)
    test_mask = df['class_id'].isin(test_groups)
    
    return df[train_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


def group_aware_kfold(
    df: pd.DataFrame,
    label_matrix: np.ndarray,
    n_splits: int = 5,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    set_seed(seed)
    
    groups = df['class_id'].values
    unique_groups = np.unique(groups)
    
    group_labels = {}
    for group in unique_groups:
        mask = groups == group
        group_labels[group] = label_matrix[mask].max(axis=0)
    
    group_list = np.array(list(unique_groups))
    group_label_matrix = np.array([group_labels[g] for g in group_list])
    
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    group_to_indices = defaultdict(list)
    for idx, group in enumerate(groups):
        group_to_indices[group].append(idx)
    
    folds = []
    for train_group_idx, val_group_idx in mskf.split(group_list, group_label_matrix):
        train_groups = group_list[train_group_idx]
        val_groups = group_list[val_group_idx]
        
        train_indices = []
        for group in train_groups:
            train_indices.extend(group_to_indices[group])
        
        val_indices = []
        for group in val_groups:
            val_indices.extend(group_to_indices[group])
        
        folds.append((np.array(train_indices), np.array(val_indices)))
    
    return folds


def main():
    parser = argparse.ArgumentParser(description='Create train/test splits with group-aware stratification')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV')
    parser.add_argument('--out', type=str, required=True, help='Path to output splits JSON')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    df = load_raw_data(args.input)
    print(f"Loaded {len(df)} samples")
    
    label_manager = LabelManager([])
    all_labels = set()
    for label_str in df['labels']:
        labels = label_manager.parse_label_string(label_str)
        all_labels.update(labels)
    
    label_manager = LabelManager(list(all_labels))
    print(f"Found {label_manager.get_num_labels()} unique labels: {label_manager.label_names}")
    
    label_lists = [label_manager.parse_label_string(s) for s in df['labels']]
    label_matrix = label_manager.encode(label_lists)
    
    print(f"\nCreating {100-args.test_size*100:.0f}/{args.test_size*100:.0f} train/test split...")
    train_df, test_df = group_aware_split(df, label_matrix, args.test_size, args.seed)
    print(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")
    
    train_label_lists = [label_manager.parse_label_string(s) for s in train_df['labels']]
    train_label_matrix = label_manager.encode(train_label_lists)
    
    print(f"\nCreating {args.folds}-fold CV splits on train set...")
    folds = group_aware_kfold(train_df, train_label_matrix, args.folds, args.seed)
    
    splits_data = {
        'test_indices': test_df.index.tolist(),
        'folds': [
            {
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist()
            }
            for train_idx, val_idx in folds
        ],
        'label_names': label_manager.label_names
    }
    
    save_json(splits_data, args.out)
    print(f"\nSaved splits to {args.out}")
    
    for i, fold in enumerate(folds):
        train_idx, val_idx = fold
        print(f"Fold {i}: Train={len(train_idx)}, Val={len(val_idx)}")


if __name__ == '__main__':
    main()
