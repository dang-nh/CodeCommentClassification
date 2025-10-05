import numpy as np
import pandas as pd
from src.split import group_aware_split, group_aware_kfold
from src.labels import LabelManager


def test_group_aware_split():
    df = pd.DataFrame({
        'id': range(100),
        'class_id': [f'class_{i//10}' for i in range(100)],
        'sentence': [f'sentence {i}' for i in range(100)],
        'lang': ['JAVA'] * 100,
        'labels': ['label1;label2'] * 100
    })
    
    label_manager = LabelManager(['label1', 'label2'])
    label_lists = [label_manager.parse_label_string(s) for s in df['labels']]
    label_matrix = label_manager.encode(label_lists)
    
    train_df, test_df = group_aware_split(df, label_matrix, test_size=0.2, seed=42)
    
    train_groups = set(train_df['class_id'].unique())
    test_groups = set(test_df['class_id'].unique())
    
    assert len(train_groups.intersection(test_groups)) == 0, "Groups should not overlap between train and test"
    assert len(train_df) + len(test_df) == len(df), "All samples should be in either train or test"
    print("✓ Group-aware split test passed")


def test_group_aware_kfold():
    df = pd.DataFrame({
        'id': range(100),
        'class_id': [f'class_{i//10}' for i in range(100)],
        'sentence': [f'sentence {i}' for i in range(100)],
        'lang': ['JAVA'] * 100,
        'labels': ['label1;label2'] * 100
    })
    
    label_manager = LabelManager(['label1', 'label2'])
    label_lists = [label_manager.parse_label_string(s) for s in df['labels']]
    label_matrix = label_manager.encode(label_lists)
    
    folds = group_aware_kfold(df, label_matrix, n_splits=5, seed=42)
    
    assert len(folds) == 5, "Should create 5 folds"
    
    for i, (train_idx, val_idx) in enumerate(folds):
        train_groups = set(df.iloc[train_idx]['class_id'].unique())
        val_groups = set(df.iloc[val_idx]['class_id'].unique())
        
        assert len(train_groups.intersection(val_groups)) == 0, f"Fold {i}: Groups should not overlap"
        assert len(train_idx) + len(val_idx) == len(df), f"Fold {i}: All samples should be used"
    
    print("✓ Group-aware k-fold test passed")


if __name__ == '__main__':
    test_group_aware_split()
    test_group_aware_kfold()
    print("\nAll split tests passed!")
