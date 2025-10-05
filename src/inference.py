import argparse
import glob
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import load_raw_data, CodeCommentDataset, prepare_tokenizer
from src.labels import LabelManager
from src.models import MultiLabelClassifier
from src.chains import ClassifierChains
from src.metrics import compute_metrics, compute_pr_auc
from src.utils import load_config, load_json, save_json, get_device, set_seed


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    all_preds = []
    
    for batch in tqdm(dataloader, desc='Predicting'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
    
    return np.vstack(all_preds)


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained models')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--ckpts', type=str, required=True, help='Checkpoint path pattern (e.g., runs/fold_*/best.pt)')
    parser.add_argument('--ensemble', type=str, default='mean', choices=['mean', 'median'], help='Ensemble method')
    parser.add_argument('--out', type=str, required=True, help='Output predictions CSV path')
    parser.add_argument('--test', action='store_true', help='Run on test set (default: validation)')
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config['train_params']['seed'])
    device = get_device()
    
    print(f"Loading data...")
    df = load_raw_data(config['data']['raw_path'])
    splits = load_json(config['data']['split_file'])
    
    label_manager = LabelManager(splits['label_names'])
    
    if args.test:
        test_indices = splits['test_indices']
        test_df = df.iloc[test_indices].reset_index(drop=True)
        test_labels = label_manager.encode([
            label_manager.parse_label_string(s) for s in test_df['labels']
        ])
        eval_df = test_df
        eval_labels = test_labels
        print(f"Test samples: {len(test_df)}")
    else:
        fold_data = splits['folds'][0]
        val_indices = fold_data['val_indices']
        val_df = df.iloc[val_indices].reset_index(drop=True)
        val_labels = label_manager.encode([
            label_manager.parse_label_string(s) for s in val_df['labels']
        ])
        eval_df = val_df
        eval_labels = val_labels
        print(f"Validation samples: {len(val_df)}")
    
    tokenizer = prepare_tokenizer(config['tokenizer_name'])
    
    eval_dataset = CodeCommentDataset(
        eval_df['sentence'].tolist(),
        torch.tensor(eval_labels),
        eval_df['lang'].tolist(),
        tokenizer,
        config['max_len']
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['train_params']['batch_size'] * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    ckpt_paths = sorted(glob.glob(args.ckpts))
    print(f"Found {len(ckpt_paths)} checkpoints")
    
    all_fold_preds = []
    
    for ckpt_path in ckpt_paths:
        print(f"\nLoading checkpoint: {ckpt_path}")
        
        model = MultiLabelClassifier(
            config['model_name'],
            label_manager.get_num_labels(),
            use_peft=config['peft']['enabled'],
            peft_config=config['peft'] if config['peft']['enabled'] else None,
            gradient_checkpointing=config['gradient_checkpointing']
        )
        
        if config['chains']['enabled']:
            label_orders = label_manager.generate_label_orders(config['chains']['num_orders'])
            model = ClassifierChains(model, label_manager.get_num_labels(), label_orders)
        
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
        
        preds = predict(model, eval_loader, device)
        all_fold_preds.append(preds)
    
    if args.ensemble == 'mean':
        final_preds = np.mean(all_fold_preds, axis=0)
    else:
        final_preds = np.median(all_fold_preds, axis=0)
    
    print(f"\nComputing metrics...")
    metrics = compute_metrics(eval_labels, final_preds, label_manager.label_names)
    pr_auc, per_label_auc = compute_pr_auc(eval_labels, final_preds)
    
    print(f"\nResults:")
    print(f"  Micro-F1: {metrics['micro_f1']:.4f}")
    print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"  Macro PR-AUC: {pr_auc:.4f}")
    
    results_df = pd.DataFrame({
        'id': eval_df['id'],
        'sentence': eval_df['sentence'],
        'true_labels': eval_df['labels'],
    })
    
    for i, label_name in enumerate(label_manager.label_names):
        results_df[f'pred_{label_name}'] = final_preds[:, i]
    
    results_df.to_csv(args.out, index=False)
    print(f"\nSaved predictions to {args.out}")
    
    metrics_output = {
        'micro_f1': float(metrics['micro_f1']),
        'macro_f1': float(metrics['macro_f1']),
        'macro_pr_auc': float(pr_auc),
        'per_label_metrics': {}
    }
    
    for i, label_name in enumerate(label_manager.label_names):
        metrics_output['per_label_metrics'][label_name] = {
            'precision': float(metrics[f'{label_name}_precision']),
            'recall': float(metrics[f'{label_name}_recall']),
            'f1': float(metrics[f'{label_name}_f1']),
            'pr_auc': float(per_label_auc[i])
        }
    
    metrics_path = args.out.replace('.csv', '_metrics.json')
    save_json(metrics_output, metrics_path)
    print(f"Saved metrics to {metrics_path}")


if __name__ == '__main__':
    main()
