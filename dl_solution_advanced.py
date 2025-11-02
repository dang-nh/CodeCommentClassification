import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    TrainerCallback
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
import warnings
import random
import os
warnings.filterwarnings('ignore')

try:
    import nlpaug.augmenter.word as naw
    NLPAUG_AVAILABLE = True
except ImportError:
    NLPAUG_AVAILABLE = False
    print("⚠️ nlpaug not available. Data augmentation disabled.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DataAugmenter:
    def __init__(self, aug_p=0.3):
        self.aug_p = aug_p
        self.synonym_aug = None
        
        if NLPAUG_AVAILABLE:
            try:
                self.synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.1)
            except:
                pass
    
    def augment(self, text):
        if random.random() > self.aug_p or self.synonym_aug is None:
            return text
        
        try:
            augmented = self.synonym_aug.augment(text)
            return augmented
        except:
            return text


class AsymmetricLossWithClassWeights(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, class_weights=None, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.class_weights = class_weights

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
        
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)
        
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * bce_loss
        
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)
        
        return loss.mean()


class CodeCommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512, augment=False, augmenter=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        self.augmenter = augmenter

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        if self.augment and self.augmenter:
            text = self.augmenter.augment(text)
        
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


class MultiSampleDropout(nn.Module):
    def __init__(self, hidden_size, num_labels, num_samples=5, dropout=0.1):
        super().__init__()
        self.num_samples = num_samples
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_samples)])
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, num_labels) for _ in range(num_samples)
        ])
    
    def forward(self, x):
        logits_list = []
        for dropout, classifier in zip(self.dropouts, self.classifiers):
            logits_list.append(classifier(dropout(x)))
        return torch.mean(torch.stack(logits_list), dim=0)


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
        use_multisample_dropout=True,
        label_smoothing=0.0
    ):
        super().__init__()
        self.num_labels = num_labels
        self.use_lora = use_lora
        self.label_smoothing = label_smoothing
        
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
        
        if use_multisample_dropout:
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.GELU(),
                MultiSampleDropout(hidden_size // 2, num_labels, num_samples=5, dropout=dropout)
            )
        else:
            self.classifier = nn.Sequential(
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
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits


class FGM:
    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def compute_class_weights(labels):
    num_labels = labels.shape[1]
    weights = []
    for i in range(num_labels):
        pos_count = labels[:, i].sum()
        neg_count = len(labels) - pos_count
        if pos_count > 0:
            weight = neg_count / pos_count
        else:
            weight = 1.0
        weights.append(weight)
    weights = np.array(weights)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.5, 5.0)
    return torch.FloatTensor(weights)


def optimize_thresholds(y_true, y_probs, num_thresholds=100):
    thresholds = np.linspace(0.1, 0.9, num_thresholds)
    num_labels = y_true.shape[1]
    best_thresholds = np.ones(num_labels) * 0.5
    best_f1s = np.zeros(num_labels)
    
    for label_idx in range(num_labels):
        for threshold in thresholds:
            y_pred = (y_probs[:, label_idx] >= threshold).astype(int)
            f1 = f1_score(y_true[:, label_idx], y_pred, zero_division=0)
            if f1 > best_f1s[label_idx]:
                best_f1s[label_idx] = f1
                best_thresholds[label_idx] = threshold
    
    return best_thresholds, best_f1s


class CustomTrainer(Trainer):
    def __init__(self, *args, custom_loss_fn=None, use_fgm=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss_fn = custom_loss_fn
        self.use_fgm = use_fgm
        self.fgm = FGM(self.model) if use_fgm else None
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.get('logits')
        
        if self.custom_loss_fn is not None:
            loss = self.custom_loss_fn(logits, labels)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        return (loss, {"logits": logits}) if return_outputs else loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        loss = self.compute_loss(model, inputs)
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        if self.use_fp16_legacy_mixed_precision:
            self.scaler.scale(loss).backward()
        elif self.deepspeed:
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        
        if self.use_fgm and self.fgm:
            self.fgm.attack()
            adv_loss = self.compute_loss(model, inputs)
            if self.args.gradient_accumulation_steps > 1:
                adv_loss = adv_loss / self.args.gradient_accumulation_steps
            if self.use_fp16_legacy_mixed_precision:
                self.scaler.scale(adv_loss).backward()
            elif self.deepspeed:
                self.deepspeed.backward(adv_loss)
            else:
                adv_loss.backward()
            self.fgm.restore()
        
        return loss.detach()


def compute_metrics_fn(eval_pred: EvalPrediction, threshold=0.5):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    
    metrics = compute_metrics(labels, preds, probs)
    return metrics


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
    
    augmenter = DataAugmenter(aug_p=config.get('augment_p', 0.3)) if config.get('use_augmentation', False) else None
    
    train_dataset = CodeCommentDataset(
        train_texts, train_labels, tokenizer, config['max_len'],
        augment=config.get('use_augmentation', False),
        augmenter=augmenter
    )
    val_dataset = CodeCommentDataset(
        val_texts, val_labels, tokenizer, config['max_len']
    )
    
    model = TransformerClassifier(
        model_name=config['model_name'],
        num_labels=config['num_labels'],
        dropout=config['peft'].get('dropout', 0.1),
        use_lora=config['peft']['enabled'],
        lora_r=config['peft']['r'],
        lora_alpha=config['peft']['alpha'],
        lora_dropout=config['peft']['dropout'],
        use_multisample_dropout=config.get('use_multisample_dropout', True),
        label_smoothing=config.get('label_smoothing', 0.0)
    )
    
    class_weights = None
    if config.get('use_class_weights', False):
        class_weights = compute_class_weights(train_labels).to(device)
        print(f"✅ Class weights enabled: min={class_weights.min():.2f}, max={class_weights.max():.2f}")
    
    if config['loss_type'] == 'asl':
        criterion = AsymmetricLossWithClassWeights(
            gamma_pos=config['loss_params']['gamma_pos'],
            gamma_neg=config['loss_params']['gamma_neg'],
            clip=config['loss_params']['clip'],
            class_weights=class_weights
        )
    elif config['loss_type'] == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0, class_weights=class_weights)
    else:
        criterion = None
    
    output_dir = Path('./runs/dl_advanced') / f'fold_{fold_idx}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fp16 = config.get('precision') == 'fp16'
    bf16 = config.get('precision') == 'bf16'
    use_fgm = config.get('use_fgm', False)
    
    deepspeed_config = config.get('deepspeed')
    if deepspeed_config and isinstance(deepspeed_config, str):
        if not os.path.exists(deepspeed_config):
            deepspeed_config = None
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config['train_params']['epochs'],
        per_device_train_batch_size=config['train_params']['batch_size'],
        per_device_eval_batch_size=config['train_params']['batch_size'] * 2,
        learning_rate=config['train_params']['lr'],
        weight_decay=config['train_params']['weight_decay'],
        warmup_ratio=config['train_params']['warmup'],
        lr_scheduler_type=config['train_params']['scheduler'],
        fp16=fp16,
        bf16=bf16,
        logging_dir=str(output_dir / 'logs'),
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f1_samples',
        greater_is_better=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=config['train_params'].get('grad_accum', 1),
        max_grad_norm=1.0,
        report_to=['tensorboard'],
        deepspeed=deepspeed_config,
        seed=config['train_params']['seed']
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
        custom_loss_fn=criterion,
        use_fgm=use_fgm
    )
    
    trainer.train()
    
    predictions = trainer.predict(val_dataset)
    val_probs = 1 / (1 + np.exp(-predictions.predictions))
    val_labels = predictions.label_ids
    
    best_thresholds, best_f1s = optimize_thresholds(val_labels, val_probs)
    best_preds = (val_probs >= best_thresholds).astype(int)
    final_metrics = compute_metrics(val_labels, best_preds, val_probs)
    
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
        config_path = Path('./configs/dl_advanced_config.yaml')
    
    print(f"📁 Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['train_params']['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*80)
    print("ADVANCED DEEP LEARNING SOLUTION")
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
    
    output_dir = Path('./runs/dl_advanced')
    output_dir.mkdir(parents=True, exist_ok=True)
    
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

