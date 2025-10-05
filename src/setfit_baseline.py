import argparse
import numpy as np
import torch
import torch.nn as nn
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
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config['train_params']['seed'])
    device = get_device()
    
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
    
    train_dataset = SetFitDataset(
        train_df['sentence'].tolist(),
        torch.tensor(train_labels, dtype=torch.float32)
    )
    
    val_dataset = SetFitDataset(
        val_df['sentence'].tolist(),
        torch.tensor(val_labels, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['train_params']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train_params']['batch_size'] * 2, shuffle=False)
    
    print(f"Building SetFit model: {config['model_name']}")
    model = SetFitClassifier(config['model_name'], label_manager.get_num_labels())
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train_params']['lr'],
        weight_decay=config['train_params']['weight_decay']
    )
    
    best_f1 = 0
    
    for epoch in range(config['train_params']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['train_params']['epochs']}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_preds, val_labels_np = eval_epoch(model, val_loader, device)
        
        metrics = compute_metrics(val_labels_np, val_preds, label_manager.label_names)
        pr_auc, _ = compute_pr_auc(val_labels_np, val_preds)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Micro-F1: {metrics['micro_f1']:.4f}")
        print(f"Val Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"Val PR-AUC: {pr_auc:.4f}")
        
        if metrics['micro_f1'] > best_f1:
            best_f1 = metrics['micro_f1']
            print(f"New best Micro-F1: {best_f1:.4f}")
    
    print(f"\nSetFit baseline complete. Best Micro-F1: {best_f1:.4f}")


if __name__ == '__main__':
    main()
