import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    f1_score
)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
import json
import yaml
from tqdm import tqdm
import warnings
import random
warnings.filterwarnings('ignore')
import os

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        probs_pos = probs
        probs_neg = 1 - probs
        
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)
        
        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        
        loss_pos = loss_pos * (1 - probs_pos) ** self.gamma_pos
        loss_neg = loss_neg * probs_pos ** self.gamma_neg
        
        loss = -(loss_pos + loss_neg)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * bce_loss
        return loss.mean()


class CodeCommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        model_name,
        num_labels,
        dropout=0.1,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        pooling_strategy='cls'
    ):
        super().__init__()
        self.num_labels = num_labels
        self.use_lora = use_lora
        self.pooling_strategy = pooling_strategy
        
        self.encoder = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.encoder.config.hidden_size
        
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["query", "key", "value", "dense"],
                bias="none"
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            print(f"✅ LoRA enabled: r={lora_r}, alpha={lora_alpha}")
        else:
            trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.encoder.parameters())
            print(f"✅ Full SFT enabled: {trainable_params:,}/{total_params:,} parameters trainable ({trainable_params/total_params*100:.1f}%)")
        
        if pooling_strategy == 'concat':
            classifier_input_size = hidden_size * 3
        else:
            classifier_input_size = hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        if self.use_lora and hasattr(self.encoder, 'get_base_model'):
            base_model = self.encoder.get_base_model()
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        hidden_states = outputs.last_hidden_state
        
        if self.pooling_strategy == 'concat':
            cls_output = hidden_states[:, 0, :]
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            mean_output = sum_hidden / sum_mask
            hidden_masked = hidden_states.masked_fill(mask_expanded == 0, -1e9)
            max_output = torch.max(hidden_masked, dim=1)[0]
            pooled_output = torch.cat([cls_output, mean_output, max_output], dim=-1)
        elif self.pooling_strategy == 'mean':
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled_output = sum_hidden / sum_mask
        else:
            pooled_output = hidden_states[:, 0, :]
        
        logits = self.classifier(pooled_output)
        return logits


def optimize_thresholds(y_true, y_probs, num_thresholds=200):
    thresholds = np.linspace(0.05, 0.95, num_thresholds)
    num_labels = y_true.shape[1]
    best_thresholds = np.ones(num_labels) * 0.5
    best_f1s = np.zeros(num_labels)
    
    for label_idx in range(num_labels):
        label_true = y_true[:, label_idx]
        label_probs = y_probs[:, label_idx]
        
        if label_true.sum() == 0:
            continue
            
        for threshold in thresholds:
            y_pred = (label_probs >= threshold).astype(int)
            f1 = f1_score(label_true, y_pred, zero_division=0)
            if f1 > best_f1s[label_idx]:
                best_f1s[label_idx] = f1
                best_thresholds[label_idx] = threshold
    
    return best_thresholds, best_f1s


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(targets.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    
    return all_probs, all_labels


def compute_metrics(y_true, y_pred, y_probs):
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_samples, recall_samples, f1_samples, _ = precision_recall_fscore_support(
        y_true, y_pred, average='samples', zero_division=0
    )
    
    try:
        roc_auc_micro = roc_auc_score(y_true, y_probs, average='micro')
        roc_auc_macro = roc_auc_score(y_true, y_probs, average='macro')
    except:
        roc_auc_micro = 0.0
        roc_auc_macro = 0.0
    
    return {
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_samples': precision_samples,
        'recall_samples': recall_samples,
        'f1_samples': f1_samples,
        'roc_auc_micro': roc_auc_micro,
        'roc_auc_macro': roc_auc_macro
    }


def train_fold(
    fold_idx,
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    config,
    device
):
    print(f"\n{'='*80}")
    print(f"Training Fold {fold_idx + 1}")
    print(f"{'='*80}")
    
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    
    train_dataset = CodeCommentDataset(
        train_texts, train_labels, tokenizer, config['max_len']
    )
    val_dataset = CodeCommentDataset(
        val_texts, val_labels, tokenizer, config['max_len']
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
    
    model = TransformerClassifier(
        model_name=config['model_name'],
        num_labels=config['num_labels'],
        dropout=config['peft'].get('dropout', 0.1),
        use_lora=config['peft']['enabled'],
        lora_r=config['peft']['r'],
        lora_alpha=config['peft']['alpha'],
        lora_dropout=config['peft']['dropout'],
        pooling_strategy=config.get('pooling_strategy', 'cls')
    ).to(device)
    
    if config['loss_type'] == 'asl':
        criterion = AsymmetricLoss(
            gamma_pos=config['loss_params']['gamma_pos'],
            gamma_neg=config['loss_params']['gamma_neg'],
            clip=config['loss_params']['clip']
        )
    elif config['loss_type'] == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
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
    
    scaler = torch.cuda.amp.GradScaler() if config.get('precision') == 'fp16' else None
    
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(config['train_params']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['train_params']['epochs']}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        val_probs, val_labels = evaluate(model, val_loader, device)
        
        val_preds = (val_probs >= 0.5).astype(int)
        metrics = compute_metrics(val_labels, val_preds, val_probs)
        
        print(f"Val F1 (micro): {metrics['f1_micro']:.4f}")
        print(f"Val F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"Val F1 (samples): {metrics['f1_samples']:.4f}")
        
        if metrics['f1_samples'] > best_f1:
            best_f1 = metrics['f1_samples']
            patience_counter = 0
            best_probs = val_probs
            best_labels = val_labels
        else:
            patience_counter += 1
        
        if patience_counter >= config['logging']['early_stop']:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    best_thresholds, best_f1s = optimize_thresholds(best_labels, best_probs)
    best_preds = (best_probs >= best_thresholds).astype(int)
    final_metrics = compute_metrics(best_labels, best_preds, best_probs)
    
    print(f"\n✅ Fold {fold_idx + 1} Best F1 (samples): {final_metrics['f1_samples']:.4f}")
    
    return {
        'fold': fold_idx,
        'metrics': final_metrics,
        'thresholds': best_thresholds.tolist(),
        'best_f1s': best_f1s.tolist()
    }


def main():
    import sys
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = Path('./configs/dl_best_config.yaml')
    
    print(f"📁 Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['train_params']['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*80)
    print("DEEP LEARNING SOLUTION - TRANSFORMER-BASED MULTI-LABEL CLASSIFICATION")
    print("="*80)
    
    df = pd.read_csv(config['data']['raw_path'])
    
    label_columns = [
        'summary', 'usage', 'expand', 'parameters', 'deprecation',
        'ownership', 'pointer', 'rational', 'intent', 'example',
        'responsibilities', 'collaborators', 'classreferences',
        'keymessages', 'keyimplementationpoints', 'developmentnotes'
    ]
    
    all_labels = df['labels'].str.split(';').apply(lambda x: [l.strip() for l in x])
    
    label_matrix = np.zeros((len(df), len(label_columns)))
    for idx, labels in enumerate(all_labels):
        for label in labels:
            if label in label_columns:
                label_idx = label_columns.index(label)
                label_matrix[idx, label_idx] = 1
    
    texts = df['sentence'].values
    labels = label_matrix
    
    print(f"\nDataset: {len(texts)} samples, {len(label_columns)} labels")
    print(f"Label distribution:")
    for i, label in enumerate(label_columns):
        count = labels[:, i].sum()
        print(f"  {label}: {int(count)} ({count/len(labels)*100:.1f}%)")
    
    use_single_split = config.get('use_single_split', False)
    
    if use_single_split:
        print("\n🔹 Using SINGLE stratified train/test split (80/20)")
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=config['train_params']['seed']
        )
        train_idx, val_idx = next(msss.split(texts, labels))
        
        train_texts = texts[train_idx]
        train_labels = labels[train_idx]
        val_texts = texts[val_idx]
        val_labels = labels[val_idx]
        
        print(f"Train set: {len(train_texts)} samples")
        print(f"Test set:  {len(val_texts)} samples")
        print(f"Train labels: {len(train_labels)}")
        print(f"Test labels: {len(val_labels)}")
        print(f"\nTrain label distribution:")
        for i, label in enumerate(label_columns):
            train_count = train_labels[:, i].sum()
            val_count = val_labels[:, i].sum()
            print(f"  {label:30s}: train={int(train_count):4d} ({train_count/len(train_labels)*100:5.1f}%)  test={int(val_count):4d} ({val_count/len(val_labels)*100:5.1f}%)")
        
        result = train_fold(
            0,
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            config,
            device
        )
        fold_results = [result]
    else:
        print("\n🔹 Using 5-FOLD Cross-Validation")
        n_splits = 5
        mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(mskf.split(texts, labels)):
            train_texts = texts[train_idx]
            train_labels = labels[train_idx]
            val_texts = texts[val_idx]
            val_labels = labels[val_idx]
            
            result = train_fold(
                fold_idx,
                train_texts,
                train_labels,
                val_texts,
                val_labels,
                config,
                device
            )
            fold_results.append(result)
    
    output_dir = Path(config['logging']['output_dir'])
    # output_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(config['logging']['output_dir'], exist_ok=True)
    
    print("\n" + "="*80)
    if use_single_split:
        print("FINAL RESULTS - SINGLE TRAIN/TEST SPLIT")
    else:
        print("FINAL RESULTS - 5-FOLD CROSS-VALIDATION")
    print("="*80)
    
    if use_single_split:
        metrics = fold_results[0]['metrics']
        print(f"\n📊 Test Set Performance:")
        print(f"  F1 (micro):   {metrics['f1_micro']:.4f}")
        print(f"  F1 (macro):   {metrics['f1_macro']:.4f}")
        print(f"  F1 (samples): {metrics['f1_samples']:.4f}")
        print(f"  Precision:    {metrics['precision_samples']:.4f}")
        print(f"  Recall:       {metrics['recall_samples']:.4f}")
        print(f"  ROC-AUC:      {metrics['roc_auc_macro']:.4f}")
        
        with open(output_dir / 'results.json', 'w') as f:
            json.dump({
                'single_split_result': fold_results[0],
                'metrics': metrics
            }, f, indent=2)
    else:
        avg_metrics = {}
        for key in fold_results[0]['metrics'].keys():
            values = [r['metrics'][key] for r in fold_results]
            avg_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        print(f"\n📊 Average Performance:")
        print(f"  F1 (micro):   {avg_metrics['f1_micro']['mean']:.4f} ± {avg_metrics['f1_micro']['std']:.4f}")
        print(f"  F1 (macro):   {avg_metrics['f1_macro']['mean']:.4f} ± {avg_metrics['f1_macro']['std']:.4f}")
        print(f"  F1 (samples): {avg_metrics['f1_samples']['mean']:.4f} ± {avg_metrics['f1_samples']['std']:.4f}")
        print(f"  Precision:    {avg_metrics['precision_samples']['mean']:.4f} ± {avg_metrics['precision_samples']['std']:.4f}")
        print(f"  Recall:       {avg_metrics['recall_samples']['mean']:.4f} ± {avg_metrics['recall_samples']['std']:.4f}")
        print(f"  ROC-AUC:      {avg_metrics['roc_auc_macro']['mean']:.4f} ± {avg_metrics['roc_auc_macro']['std']:.4f}")
        
        with open(output_dir / 'results.json', 'w') as f:
            json.dump({
                'fold_results': fold_results,
                'average_metrics': {k: v['mean'] for k, v in avg_metrics.items()},
                'std_metrics': {k: v['std'] for k, v in avg_metrics.items()}
            }, f, indent=2)
    
    print(f"\n✅ Results saved to {output_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()

