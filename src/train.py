import argparse
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from src.data import load_raw_data, CodeCommentDataset, prepare_tokenizer
from src.labels import LabelManager
from src.models import MultiLabelClassifier
from src.chains import ClassifierChains
from src.losses import AsymmetricLoss, BCEWithLogitsLoss
from src.metrics import compute_metrics, compute_pr_auc
from src.utils import (
    load_config, save_json, load_json, setup_logging,
    set_seed, get_device, count_parameters
)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, grad_accum):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc='Training')
    for i, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss = loss / grad_accum
        loss.backward()
        
        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum
        pbar.set_postfix({'loss': f'{loss.item() * grad_accum:.4f}'})
    
    return total_loss / len(dataloader)


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Evaluating')
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        probs = torch.sigmoid(logits)
        
        all_preds.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        total_loss += loss.item()
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    return total_loss / len(dataloader), all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train multi-label classifier')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--fold', type=int, default=0, help='Fold number to train')
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config['train_params']['seed'])
    device = get_device()
    
    output_dir = os.path.join(config['logging']['output_dir'], f'fold_{args.fold}')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, 'train')
    
    logger.info(f"Loading data...")
    df = load_raw_data(config['data']['raw_path'])
    splits = load_json(config['data']['split_file'])
    
    label_manager = LabelManager(splits['label_names'])
    logger.info(f"Number of labels: {label_manager.get_num_labels()}")
    
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
    
    logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    tokenizer = prepare_tokenizer(config['tokenizer_name'])
    
    train_dataset = CodeCommentDataset(
        train_df['sentence'].tolist(),
        torch.tensor(train_labels),
        train_df['lang'].tolist(),
        tokenizer,
        config['max_len']
    )
    
    val_dataset = CodeCommentDataset(
        val_df['sentence'].tolist(),
        torch.tensor(val_labels),
        val_df['lang'].tolist(),
        tokenizer,
        config['max_len']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_params']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train_params']['batch_size'] * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Building model: {config['model_name']}")
    model = MultiLabelClassifier(
        config['model_name'],
        label_manager.get_num_labels(),
        use_peft=config['peft']['enabled'],
        peft_config=config['peft'] if config['peft']['enabled'] else None,
        gradient_checkpointing=config['gradient_checkpointing']
    )
    
    if config['chains']['enabled']:
        logger.info(f"Enabling classifier chains with {config['chains']['num_orders']} orders")
        label_orders = label_manager.generate_label_orders(config['chains']['num_orders'])
        model = ClassifierChains(model, label_manager.get_num_labels(), label_orders)
    
    model = model.to(device)
    logger.info(f"Trainable parameters: {count_parameters(model):,}")
    
    if config['loss_type'] == 'asl':
        criterion = AsymmetricLoss(**config['loss_params'])
    else:
        criterion = BCEWithLogitsLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train_params']['lr'],
        weight_decay=config['train_params']['weight_decay']
    )
    
    num_training_steps = len(train_loader) * config['train_params']['epochs']
    num_warmup_steps = int(num_training_steps * config['train_params']['warmup'])
    
    if config['train_params']['scheduler'] == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    
    writer = None
    if config['logging']['tensorboard']:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
    
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(config['train_params']['epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['train_params']['epochs']}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, config['train_params']['grad_accum']
        )
        
        val_loss, val_preds, val_labels_np = eval_epoch(model, val_loader, criterion, device)
        
        metrics = compute_metrics(val_labels_np, val_preds, label_manager.label_names)
        pr_auc, _ = compute_pr_auc(val_labels_np, val_preds)
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Val Micro-F1: {metrics['micro_f1']:.4f}")
        logger.info(f"Val Macro-F1: {metrics['macro_f1']:.4f}")
        logger.info(f"Val PR-AUC: {pr_auc:.4f}")
        
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('F1/micro', metrics['micro_f1'], epoch)
            writer.add_scalar('F1/macro', metrics['macro_f1'], epoch)
            writer.add_scalar('PR-AUC/macro', pr_auc, epoch)
        
        if metrics['micro_f1'] > best_f1:
            best_f1 = metrics['micro_f1']
            patience_counter = 0
            
            torch.save(model.state_dict(), os.path.join(output_dir, 'best.pt'))
            np.save(os.path.join(output_dir, 'val_preds.npy'), val_preds)
            np.save(os.path.join(output_dir, 'val_labels.npy'), val_labels_np)
            
            logger.info(f"Saved best model with Micro-F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config['logging']['early_stop']:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
    
    if writer:
        writer.close()
    
    logger.info(f"\nTraining complete. Best Micro-F1: {best_f1:.4f}")


if __name__ == '__main__':
    main()
