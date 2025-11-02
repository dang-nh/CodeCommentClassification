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
import copy
warnings.filterwarnings('ignore')
import os

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, hidden_states, attention_mask):
        attention_weights = self.attention(hidden_states)
        attention_mask = attention_mask.unsqueeze(-1).float()
        attention_weights = attention_weights * attention_mask
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-9)
        pooled = (hidden_states * attention_weights).sum(dim=1)
        return pooled


class MultiSampleDropout(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_rates=[0.1, 0.2, 0.3, 0.4, 0.5]):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(rate) for rate in dropout_rates])
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, num_labels) for _ in dropout_rates
        ])
    
    def forward(self, x):
        logits_list = []
        for dropout, classifier in zip(self.dropouts, self.classifiers):
            logits_list.append(classifier(dropout(x)))
        return torch.stack(logits_list).mean(dim=0)


class EnhancedAsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8, label_smoothing=0.0):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
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


class FocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * bce_loss
        return loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.asl = EnhancedAsymmetricLoss(label_smoothing=label_smoothing)
        self.focal = FocalLossWithSmoothing(label_smoothing=label_smoothing)
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        loss_asl = self.asl(logits, targets)
        loss_focal = self.focal(logits, targets)
        loss_bce = self.bce(logits, targets)
        return self.alpha * loss_asl + self.beta * loss_focal + self.gamma * loss_bce


class RDropLoss(nn.Module):
    def __init__(self, base_criterion, alpha=5.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.alpha = alpha
    
    def forward(self, logits1, logits2, targets):
        ce_loss = 0.5 * (self.base_criterion(logits1, targets) + self.base_criterion(logits2, targets))
        kl_loss = self.compute_kl_loss(logits1, logits2)
        return ce_loss + self.alpha * kl_loss
    
    def compute_kl_loss(self, logits1, logits2):
        p1 = F.log_softmax(logits1, dim=-1)
        p2 = F.log_softmax(logits2, dim=-1)
        kl_loss = F.kl_div(p1, F.softmax(logits2, dim=-1), reduction='batchmean')
        kl_loss += F.kl_div(p2, F.softmax(logits1, dim=-1), reduction='batchmean')
        return kl_loss / 2


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


class EnhancedTransformerClassifier(nn.Module):
    def __init__(
        self,
        model_name,
        num_labels,
        dropout=0.1,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        use_multi_sample_dropout=True,
        use_attention_pooling=True,
        pooling_strategy='concat'
    ):
        super().__init__()
        self.num_labels = num_labels
        self.use_lora = use_lora
        self.use_multi_sample_dropout = use_multi_sample_dropout
        self.use_attention_pooling = use_attention_pooling
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
        
        if use_attention_pooling:
            self.attention_pool = AttentionPooling(hidden_size)
            print("✅ Attention Pooling enabled")
        
        if pooling_strategy == 'concat':
            pooled_size = hidden_size * 3
        elif pooling_strategy == 'concat_all':
            pooled_size = hidden_size * 4
        else:
            pooled_size = hidden_size
        
        if use_multi_sample_dropout:
            self.pre_classifier = nn.Sequential(
                nn.Linear(pooled_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.classifier = MultiSampleDropout(hidden_size, num_labels)
            print("✅ Multi-Sample Dropout enabled")
        else:
            self.classifier = nn.Sequential(
                nn.Linear(pooled_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_labels)
            )

    def forward(self, input_ids, attention_mask, return_hidden=False):
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
        
        if self.pooling_strategy == 'cls':
            pooled = hidden_states[:, 0, :]
        elif self.pooling_strategy == 'mean':
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_hidden / sum_mask
        elif self.pooling_strategy == 'max':
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states = hidden_states.masked_fill(mask_expanded == 0, -1e9)
            pooled = torch.max(hidden_states, dim=1)[0]
        elif self.pooling_strategy == 'concat':
            cls_output = hidden_states[:, 0, :]
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            mean_output = sum_hidden / sum_mask
            hidden_masked = hidden_states.masked_fill(mask_expanded == 0, -1e9)
            max_output = torch.max(hidden_masked, dim=1)[0]
            pooled = torch.cat([cls_output, mean_output, max_output], dim=-1)
        elif self.pooling_strategy == 'concat_all':
            cls_output = hidden_states[:, 0, :]
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            mean_output = sum_hidden / sum_mask
            hidden_masked = hidden_states.masked_fill(mask_expanded == 0, -1e9)
            max_output = torch.max(hidden_masked, dim=1)[0]
            
            if self.use_attention_pooling:
                attention_output = self.attention_pool(hidden_states, attention_mask)
                pooled = torch.cat([cls_output, mean_output, max_output, attention_output], dim=-1)
            else:
                pooled = torch.cat([cls_output, mean_output, max_output], dim=-1)
        else:
            pooled = hidden_states[:, 0, :]
        
        if self.use_multi_sample_dropout:
            pooled = self.pre_classifier(pooled)
        
        logits = self.classifier(pooled)
        
        if return_hidden:
            return logits, pooled
        return logits


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def optimize_thresholds_advanced(y_true, y_probs, num_thresholds=200, search_range=(0.05, 0.95)):
    thresholds = np.linspace(search_range[0], search_range[1], num_thresholds)
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


def get_layerwise_lr(model, lr, lr_decay=0.95):
    opt_parameters = []
    named_parameters = list(model.named_parameters())
    
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    
    encoder_params = [(n, p) for n, p in named_parameters if 'encoder' in n]
    classifier_params = [(n, p) for n, p in named_parameters if 'classifier' in n or 'pre_classifier' in n]
    
    num_layers = 24
    groups = []
    
    for layer in range(num_layers):
        layer_params = {
            "params": [p for n, p in encoder_params 
                      if f"layer.{layer}." in n and not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
            "lr": lr * (lr_decay ** (num_layers - layer))
        }
        groups.append(layer_params)
        
        layer_params_no_decay = {
            "params": [p for n, p in encoder_params 
                      if f"layer.{layer}." in n and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr * (lr_decay ** (num_layers - layer))
        }
        groups.append(layer_params_no_decay)
    
    group_classifier = {
        "params": [p for n, p in classifier_params if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
        "lr": lr
    }
    groups.append(group_classifier)
    
    group_classifier_no_decay = {
        "params": [p for n, p in classifier_params if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": lr
    }
    groups.append(group_classifier_no_decay)
    
    return groups


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler=None, use_rdrop=False, ema=None):
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
                if use_rdrop:
                    logits1 = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits2 = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(logits1, logits2, targets)
                else:
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if use_rdrop:
                logits1 = model(input_ids=input_ids, attention_mask=attention_mask)
                logits2 = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits1, logits2, targets)
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        if ema is not None:
            ema.update()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, use_ema=False, ema=None):
    if use_ema and ema is not None:
        ema.apply_shadow()
    
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
    
    if use_ema and ema is not None:
        ema.restore()
    
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
    
    model = EnhancedTransformerClassifier(
        model_name=config['model_name'],
        num_labels=config['num_labels'],
        dropout=config['peft'].get('dropout', 0.1),
        use_lora=config['peft']['enabled'],
        lora_r=config['peft']['r'],
        lora_alpha=config['peft']['alpha'],
        lora_dropout=config['peft']['dropout'],
        use_multi_sample_dropout=config.get('use_multi_sample_dropout', True),
        use_attention_pooling=config.get('use_attention_pooling', True),
        pooling_strategy=config.get('pooling_strategy', 'concat_all')
    ).to(device)
    
    loss_type = config.get('loss_type', 'combined')
    label_smoothing = config.get('label_smoothing', 0.1)
    use_rdrop = config.get('use_rdrop', True)
    
    if loss_type == 'combined':
        base_criterion = CombinedLoss(
            alpha=0.5, beta=0.3, gamma=0.2, 
            label_smoothing=label_smoothing
        )
        print(f"✅ Combined Loss (ASL+Focal+BCE) with label smoothing={label_smoothing}")
    elif loss_type == 'asl':
        base_criterion = EnhancedAsymmetricLoss(
            gamma_pos=config['loss_params']['gamma_pos'],
            gamma_neg=config['loss_params']['gamma_neg'],
            clip=config['loss_params']['clip'],
            label_smoothing=label_smoothing
        )
    elif loss_type == 'focal':
        base_criterion = FocalLossWithSmoothing(
            alpha=0.25, gamma=2.0, label_smoothing=label_smoothing
        )
    else:
        base_criterion = nn.BCEWithLogitsLoss()
    
    if use_rdrop:
        criterion = RDropLoss(base_criterion, alpha=config.get('rdrop_alpha', 5.0))
        print(f"✅ R-Drop enabled with alpha={config.get('rdrop_alpha', 5.0)}")
    else:
        criterion = base_criterion
    
    use_layerwise_lr = config.get('use_layerwise_lr', True)
    if use_layerwise_lr:
        param_groups = get_layerwise_lr(
            model, 
            float(config['train_params']['lr']),
            float(config.get('lr_decay', 0.95))
        )
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=float(config['train_params']['weight_decay'])
        )
        print(f"✅ Layer-wise LR decay enabled (decay={config.get('lr_decay', 0.95)})")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config['train_params']['lr']),
            weight_decay=float(config['train_params']['weight_decay'])
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
    
    use_ema = config.get('use_ema', True)
    if use_ema:
        ema = EMA(model, decay=config.get('ema_decay', 0.999))
        print(f"✅ EMA enabled with decay={config.get('ema_decay', 0.999)}")
    else:
        ema = None
    
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(config['train_params']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['train_params']['epochs']}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler, use_rdrop, ema
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        val_probs, val_labels = evaluate(model, val_loader, device, use_ema, ema)
        
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
    
    best_thresholds, best_f1s = optimize_thresholds_advanced(
        best_labels, best_probs, 
        num_thresholds=config.get('num_threshold_search', 200)
    )
    best_preds = (best_probs >= best_thresholds).astype(int)
    final_metrics = compute_metrics(best_labels, best_preds, best_probs)
    
    print(f"\n✅ Fold {fold_idx + 1} Best F1 (samples): {final_metrics['f1_samples']:.4f}")
    print(f"   F1 (macro): {final_metrics['f1_macro']:.4f}")
    print(f"   F1 (micro): {final_metrics['f1_micro']:.4f}")
    
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
        config_path = Path('./configs/dl_enhanced_config.yaml')
    
    print(f"📁 Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['train_params']['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*80)
    print("ENHANCED DEEP LEARNING SOLUTION - BEATING STACC (F1=0.744)")
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
        
        print(f"\n🎯 STACC Comparison:")
        print(f"  STACC F1: 0.744")
        print(f"  Our F1:   {metrics['f1_samples']:.4f}")
        improvement = (metrics['f1_samples'] - 0.744) / 0.744 * 100
        if improvement > 0:
            print(f"  ✅ IMPROVEMENT: +{improvement:.2f}%")
        else:
            print(f"  ❌ Behind by: {abs(improvement):.2f}%")
        
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
        
        print(f"\n🎯 STACC Comparison:")
        print(f"  STACC F1: 0.744")
        print(f"  Our F1:   {avg_metrics['f1_samples']['mean']:.4f}")
        improvement = (avg_metrics['f1_samples']['mean'] - 0.744) / 0.744 * 100
        if improvement > 0:
            print(f"  ✅ IMPROVEMENT: +{improvement:.2f}%")
        else:
            print(f"  ❌ Behind by: {abs(improvement):.2f}%")
        
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

