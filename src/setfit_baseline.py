import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.data import load_raw_data
from src.labels import LabelManager
from src.metrics import compute_metrics, compute_pr_auc
from src.utils import load_config, load_json, save_json, set_seed, get_device


class SetFitDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class SetFitClassifier(nn.Module):
    def __init__(self, encoder_name, num_labels):
        super(SetFitClassifier, self).__init__()
        self.encoder = SentenceTransformer(encoder_name)
        self.hidden_size = self.encoder.get_sentence_embedding_dimension()
        self.classifier = nn.Linear(self.hidden_size, num_labels)
    
    def forward(self, texts):
        embeddings = self.encoder.encode(
            texts, convert_to_tensor=True, show_progress_bar=False
        )
        logits = self.classifier(embeddings)
        return logits


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for texts, labels in tqdm(dataloader, desc='Training'):
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(texts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


@torch.no_grad()
def eval_epoch(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    for texts, labels in tqdm(dataloader, desc='Evaluating'):
        logits = model(texts)
        probs = torch.sigmoid(logits)
        
        all_preds.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
    
    return np.vstack(all_preds), np.vstack(all_labels)


def main():
    parser = argparse.ArgumentParser(description='SetFit baseline for multi-label classification')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--fold', type=int, default=0, help='Fold number to train')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--use_test', action='store_true', help='Evaluate on test set instead of validation')
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config['train_params']['seed'])
    device = get_device()
    
    if args.out_dir:
        output_dir = args.out_dir
    else:
        output_dir = os.path.join(config['logging']['output_dir'], f'setfit_fold_{args.fold}')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SetFit Baseline")
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
    
    train_dataset = SetFitDataset(
        train_df['sentence'].tolist(),
        torch.tensor(train_labels, dtype=torch.float32)
    )
    
    eval_dataset = SetFitDataset(
        eval_df['sentence'].tolist(),
        torch.tensor(eval_labels, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['train_params']['batch_size'], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config['train_params']['batch_size'] * 2, shuffle=False)
    
    print(f"Building SetFit model: {config['model_name']}")
    model = SetFitClassifier(config['model_name'], label_manager.get_num_labels())
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train_params']['lr'],
        weight_decay=config['train_params']['weight_decay']
    )
    
    print(f"Training for {config['train_params']['epochs']} epochs...")
    print()
    
    best_f1 = 0
    best_preds = None
    
    for epoch in range(config['train_params']['epochs']):
        print(f"Epoch {epoch + 1}/{config['train_params']['epochs']}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        eval_preds, eval_labels_np = eval_epoch(model, eval_loader, device)
        
        metrics = compute_metrics(eval_labels_np, eval_preds, label_manager.label_names)
        pr_auc, _ = compute_pr_auc(eval_labels_np, eval_preds)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Eval Micro-F1: {metrics['micro_f1']:.4f}")
        print(f"  Eval Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"  Eval PR-AUC: {pr_auc:.4f}")
        print()
        
        if metrics['micro_f1'] > best_f1:
            best_f1 = metrics['micro_f1']
            best_preds = eval_preds
    
    print("=" * 60)
    print("Final Results:")
    print("=" * 60)
    
    final_metrics = compute_metrics(eval_labels, best_preds, label_manager.label_names)
    final_pr_auc, per_label_auc = compute_pr_auc(eval_labels, best_preds)
    
    print(f"  Micro-F1:    {final_metrics['micro_f1']:.4f}")
    print(f"  Macro-F1:    {final_metrics['macro_f1']:.4f}")
    print(f"  Macro PR-AUC: {final_pr_auc:.4f}")
    print()
    
    per_label_metrics = []
    for i, label_name in enumerate(label_manager.label_names):
        per_label_metrics.append({
            'label': label_name,
            'precision': final_metrics[f'{label_name}_precision'],
            'recall': final_metrics[f'{label_name}_recall'],
            'f1': final_metrics[f'{label_name}_f1'],
            'pr_auc': per_label_auc[i],
            'support': int(eval_labels[:, i].sum())
        })
    
    per_label_df = pd.DataFrame(per_label_metrics)
    per_label_csv = os.path.join(output_dir, 'metrics_per_label.csv')
    per_label_df.to_csv(per_label_csv, index=False)
    print(f"Saved per-label metrics to: {per_label_csv}")
    
    summary = {
        'model': 'setfit',
        'encoder': config['model_name'],
        'fold': args.fold if not args.use_test else 'all',
        'split': 'test' if args.use_test else 'validation',
        'micro_f1': float(final_metrics['micro_f1']),
        'macro_f1': float(final_metrics['macro_f1']),
        'micro_precision': float(final_metrics['micro_precision']),
        'micro_recall': float(final_metrics['micro_recall']),
        'macro_precision': float(final_metrics['macro_precision']),
        'macro_recall': float(final_metrics['macro_recall']),
        'macro_pr_auc': float(final_pr_auc),
        'num_train': len(train_df),
        'num_eval': len(eval_df),
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
        predictions_df[f'pred_{label_name}'] = best_preds[:, i]
    
    predictions_csv = os.path.join(output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_csv, index=False)
    print(f"Saved predictions to: {predictions_csv}")
    
    print()
    print("=" * 60)
    print("SetFit baseline complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()