import os
import gc
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    f1_score
)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

# PEFT / LoRA (tùy chọn)
try:
    from peft import get_peft_model, LoraConfig, TaskType
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

warnings.filterwarnings("ignore")


# ===================== Utils =====================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_yaml(path: Path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def norm_lang(v):
    s = str(v).strip().lower()
    if "java" in s:
        return "java"
    if ("python" in s) or (s in ("py", "py3", "python3")):
        return "python"
    if ("pharo" in s) or ("smalltalk" in s):
        return "pharo"
    return "other"


def map_labels_for_row(labels_str, lang_norm, expand_to_19):
    labs = str(labels_str or "").split(";")
    labs = [x.strip().lower() for x in labs if x.strip()]
    if not expand_to_19:
        return labs
    mapped = []
    for lab in labs:
        if lab in ("summary", "usage", "expand"):
            if lang_norm in ("java", "python"):
                mapped.append(f"{lab}_{lang_norm}")
            else:
                # ít gặp ở pharo; giữ nguyên nếu có
                mapped.append(lab)
        else:
            mapped.append(lab)
    return mapped


def build_label_matrix(df, label_list, expand_to_19=True):
    name_to_idx = {name: i for i, name in enumerate(label_list)}
    L = len(label_list)
    mat = np.zeros((len(df), L), dtype=np.float32)
    for i, row in df.iterrows():
        lang = norm_lang(row.get("lang", ""))
        labs = map_labels_for_row(row.get("labels", ""), lang, expand_to_19)
        for lab in labs:
            if lab in name_to_idx:
                mat[i, name_to_idx[lab]] = 1.0
    return mat


def move_state_to_cpu(state_dict):
    out = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ===================== Losses =====================

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


# ===================== Dataset =====================

class CodeCommentDataset(Dataset):
    def __init__(self, df, label_cols, tokenizer, max_len=128, text_cfg=None, to_19=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_cols = label_cols
        self.text_cfg = text_cfg or {}
        self.to_19 = to_19

        self.name_to_idx = {name: i for i, name in enumerate(self.label_cols)}
        self.label_matrix = np.zeros((len(self.df), len(self.label_cols)), dtype=np.float32)

        for i, row in self.df.iterrows():
            lang = norm_lang(row.get("lang", ""))
            labs = map_labels_for_row(row.get("labels", ""), lang, self.to_19)
            for lab in labs:
                if lab in self.name_to_idx:
                    self.label_matrix[i, self.name_to_idx[lab]] = 1.0

    def __len__(self):
        return len(self.df)

    def _build_text(self, row):
        sentence = str(row["sentence"])
        lang = norm_lang(row.get("lang", ""))
        class_id = str(row.get("class_id", "UNKNOWN"))
        include_lang = self.text_cfg.get("include_lang", False)
        include_class_id = self.text_cfg.get("include_class_id", False)
        template = self.text_cfg.get("template", "[{lang}] {class_id} | {sentence}")
        if include_lang or include_class_id:
            return template.format(lang=lang.upper(), class_id=class_id, sentence=sentence)
        return sentence

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = self._build_text(row)
        labels = self.label_matrix[idx]

        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


# ===================== Model =====================

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules=None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.use_lora = use_lora and HAS_PEFT

        self.encoder = AutoModel.from_pretrained(model_name)

        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            try:
                self.encoder.gradient_checkpointing_enable()
            except Exception:
                pass

        if self.use_lora:
            target_modules = lora_target_modules or ["query", "key", "value", "dense"]
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,  # encoder
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            print(f"✅ LoRA enabled (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}) on {target_modules}")
        else:
            trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.encoder.parameters())
            print(f"✅ Full SFT: {trainable:,}/{total:,} trainable ({trainable/total*100:.1f}%)")

        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        # Mean pooling theo mask
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / denom

        logits = self.classifier(pooled)
        return logits


# ===================== Metrics & Thresholds =====================

def safe_roc_auc(y_true, y_probs, average="macro"):
    try:
        valid = []
        for j in range(y_true.shape[1]):
            col = y_true[:, j]
            if col.min() != col.max():
                valid.append(j)
        if not valid:
            return np.nan
        return roc_auc_score(y_true[:, valid], y_probs[:, valid], average=average)
    except Exception:
        return np.nan


def compute_metrics(y_true, y_pred, y_probs):
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_samples, recall_samples, f1_samples, _ = precision_recall_fscore_support(
        y_true, y_pred, average="samples", zero_division=0
    )

    try:
        roc_auc_micro = safe_roc_auc(y_true, y_probs, average="micro")
        roc_auc_macro = safe_roc_auc(y_true, y_probs, average="macro")
        pr_auc_micro = average_precision_score(y_true, y_probs, average="micro")
        pr_auc_macro = average_precision_score(y_true, y_probs, average="macro")
    except Exception:
        roc_auc_micro = roc_auc_macro = pr_auc_micro = pr_auc_macro = np.nan

    return {
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_samples": precision_samples,
        "recall_samples": recall_samples,
        "f1_samples": f1_samples,
        "roc_auc_micro": roc_auc_micro,
        "roc_auc_macro": roc_auc_macro,
        "pr_auc_micro": pr_auc_micro,
        "pr_auc_macro": pr_auc_macro,
    }


def optimize_thresholds_per_label(y_true, y_probs, num_thresholds=100):
    thresholds = np.linspace(0.1, 0.9, num_thresholds)
    L = y_true.shape[1]
    best_t = np.ones(L) * 0.5
    for j in range(L):
        yt = y_true[:, j]
        yp = y_probs[:, j]
        best_f, bt = -1.0, 0.5
        for t in thresholds:
            pred = (yp >= t).astype(int)
            f = f1_score(yt, pred, zero_division=0)
            if f > best_f:
                best_f, bt = f, t
        best_t[j] = bt
    return best_t


def optimize_thresholds_samples(y_true, y_probs, base=None, iters=2, grid=50):
    """
    Coordinate-descent tối ưu F1(samples) toàn cục.
    - base: khởi tạo từ per-label thresholds (nên có).
    """
    L = y_true.shape[1]
    thr = (base.copy() if base is not None else np.ones(L) * 0.5).astype(np.float32)

    def f1s(th):
        pred = (y_probs >= th).astype(int)
        return f1_score(y_true, pred, average="samples", zero_division=0)

    if base is None:
        ts = np.linspace(0.05, 0.95, grid)
        for j in range(L):
            best_f, bt = -1.0, 0.5
            for t in ts:
                pred = (y_probs[:, j] >= t).astype(int)
                f = f1_score(y_true[:, j], pred, zero_division=0)
                if f > best_f:
                    best_f, bt = f, t
            thr[j] = bt

    ts = np.linspace(0.05, 0.95, grid)
    for _ in range(iters):
        improved = False
        for j in range(L):
            cur_best = f1s(thr)
            cur_t = thr[j]
            best_f, bt = cur_best, cur_t
            for t in ts:
                tmp = thr.copy()
                tmp[j] = t
                f = f1s(tmp)
                if f > best_f:
                    best_f, bt = f, t
            if bt != cur_t:
                thr[j] = bt
                improved = True
        if not improved:
            break
    return thr


# ===================== Train/Eval loops =====================

def make_scheduler(optimizer, total_updates, warmup_ratio, name="cosine"):
    warmup_steps = int(total_updates * float(warmup_ratio))
    if name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    else:
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)


def get_amp_setup(precision):
    use_amp = precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))
    return use_amp, amp_dtype, scaler


def make_sampler_if_needed(train_labels, cfg):
    samp_cfg = cfg.get("sampler", {})
    if not samp_cfg.get("enabled", False):
        return None
    typ = samp_cfg.get("type", "weighted")
    if typ != "weighted":
        return None

    eps = float(samp_cfg.get("smooth_eps", 1e-3))
    freq = train_labels.sum(0) + eps
    inv = 1.0 / freq  # nhãn hiếm -> trọng số lớn
    sample_w = (train_labels * inv).sum(1)
    # nếu mẫu không có nhãn (hiếm): gán trọng số tối thiểu
    min_w = float(inv.min()) * 0.5
    sample_w = np.where(sample_w > 0, sample_w, min_w)
    sample_w = torch.tensor(sample_w, dtype=torch.float32)
    return WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, precision="fp32",
                    grad_accum=1, max_grad_norm=1.0):
    model.train()
    running = 0.0
    step_in_accum = 0
    use_amp, amp_dtype, scaler = get_amp_setup(precision)

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, targets) / grad_accum
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, targets) / grad_accum

        if precision == "fp16":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        step_in_accum += 1
        if step_in_accum % grad_accum == 0:
            if precision == "fp16":
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            step_in_accum = 0

        running += loss.item() * grad_accum

    return running / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, precision="fp32"):
    model.eval()
    probs_all, labels_all = [], []
    use_amp, amp_dtype, _ = get_amp_setup(precision)

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

        probs = torch.sigmoid(logits)
        probs_all.append(probs.cpu().numpy())
        labels_all.append(targets.cpu().numpy())

    probs_all = np.vstack(probs_all)
    labels_all = np.vstack(labels_all)
    return probs_all, labels_all


def build_dataloaders(train_df, val_df, label_cols, tokenizer, config):
    text_cfg = config.get("text_features", {})
    to_19 = bool(config["data"].get("expand_to_19_classes", False))

    train_bs = int(config["train_params"]["batch_size"])
    eval_bs = int(config["train_params"].get("eval_batch_size", train_bs))  # NEW: tách batch val

    train_ds = CodeCommentDataset(train_df, label_cols, tokenizer, max_len=config["max_len"], text_cfg=text_cfg, to_19=to_19)
    val_ds = CodeCommentDataset(val_df, label_cols, tokenizer, max_len=config["max_len"], text_cfg=text_cfg, to_19=to_19)

    sampler = make_sampler_if_needed(train_ds.label_matrix, config)
    if sampler is not None:
        train_loader = DataLoader(
            train_ds, batch_size=train_bs,
            sampler=sampler, num_workers=config["train_params"].get("num_workers", 4),
            pin_memory=True, drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=train_bs,
            shuffle=True, num_workers=config["train_params"].get("num_workers", 4),
            pin_memory=True, drop_last=False,
        )

    val_loader = DataLoader(
        val_ds, batch_size=eval_bs,          # NEW: không *2 nữa
        shuffle=False, num_workers=config["train_params"].get("num_workers", 4),
        pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader, train_ds


def make_loss(config, device, train_label_matrix=None):
    lt = config.get("loss_type", "asl").lower()
    if lt == "asl":
        p = config.get("loss_params", {})
        return AsymmetricLoss(gamma_pos=p.get("gamma_pos", 0), gamma_neg=p.get("gamma_neg", 4), clip=p.get("clip", 0.05))
    elif lt == "focal":
        return FocalLoss(alpha=0.25, gamma=2.0)
    else:
        # BCE with pos_weight
        assert train_label_matrix is not None, "Need train_label_matrix for BCE pos_weight."
        pos = train_label_matrix.sum(0)
        neg = len(train_label_matrix) - pos
        pos_weight = torch.tensor((neg / (pos + 1e-6)).astype(np.float32), device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_fold(
    fold_idx,
    train_df,
    val_df,
    label_cols,
    config,
    device
):
    print(f"\n{'='*80}\nTraining Fold {fold_idx+1}\n{'='*80}")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    train_loader, val_loader, train_ds = build_dataloaders(train_df, val_df, label_cols, tokenizer, config)

    model = TransformerClassifier(
        model_name=config["model_name"],
        num_labels=len(label_cols),
        dropout=config["peft"].get("dropout", 0.1),
        use_lora=config["peft"].get("enabled", False),
        lora_r=config["peft"].get("r", 16),
        lora_alpha=config["peft"].get("alpha", 32),
        lora_dropout=config["peft"].get("dropout", 0.1),
        lora_target_modules=config["peft"].get("target_modules", None),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
    ).to(device)

    criterion = make_loss(config, device, train_label_matrix=train_ds.label_matrix)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train_params"]["lr"],
        weight_decay=config["train_params"]["weight_decay"],
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(1, int(config["train_params"]["grad_accum"])))
    total_updates = steps_per_epoch * int(config["train_params"]["epochs"])
    scheduler = make_scheduler(
        optimizer, total_updates=total_updates,
        warmup_ratio=config["train_params"]["warmup"],
        name=config["train_params"]["scheduler"],
    )

    precision = config.get("precision", "fp32")
    best_state = None
    best_f1 = -1.0
    patience = 0
    early_stop = int(config["logging"].get("early_stop", 6))
    do_thr_search = bool(config["eval"].get("threshold_search", True))

    for epoch in range(config["train_params"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['train_params']['epochs']}")
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            precision=precision, grad_accum=int(config["train_params"]["grad_accum"])
        )
        print(f"Train Loss: {tr_loss:.4f}")

        val_probs, val_labels = evaluate(model, val_loader, device, precision=precision)

        if do_thr_search:
            base_thr = optimize_thresholds_per_label(val_labels, val_probs)
            thr = optimize_thresholds_samples(val_labels, val_probs, base=base_thr, iters=2)
        else:
            thr = np.ones(len(label_cols)) * 0.5

        val_preds = (val_probs >= thr).astype(int)
        metrics = compute_metrics(val_labels, val_preds, val_probs)
        print(
            f"Val F1(samples): {metrics['f1_samples']:.4f} | "
            f"F1(micro): {metrics['f1_micro']:.4f} | "
            f"F1(macro): {metrics['f1_macro']:.4f} | "
            f"PR-AUC(macro): {metrics['pr_auc_macro'] if isinstance(metrics['pr_auc_macro'], float) else 'nan'}"
        )

        if metrics["f1_samples"] > best_f1:
            best_f1 = metrics["f1_samples"]
            # Lưu state_dict về CPU để giảm chiếm VRAM khi giữ best_state trong RAM
            best_state = {
                "model": move_state_to_cpu(model.state_dict()),
                "thresholds": thr.tolist(),
                "metrics": metrics
            }
            patience = 0
        else:
            patience += 1

        if patience >= early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\n✅ Fold {fold_idx+1} Best F1(samples): {best_state['metrics']['f1_samples']:.4f}")

    # Giải phóng RAM/GPU trong phạm vi fold
    del train_loader, val_loader, train_ds, tokenizer, optimizer, scheduler, criterion, model
    free_cuda()

    return best_state


def train_full(df, label_cols, config, device, save_path: Path, med_thresholds: np.ndarray):
    print("\n================ FULL TRAIN ================")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    # train full + holdout nhỏ không có → chỉ để huấn luyện; ngưỡng dùng median thresholds từ CV
    text_cfg = config.get("text_features", {})
    full_ds = CodeCommentDataset(
        df, label_cols, tokenizer, max_len=config["max_len"], text_cfg=text_cfg,
        to_19=bool(config["data"].get("expand_to_19_classes", False))
    )
    sampler = make_sampler_if_needed(full_ds.label_matrix, config)
    full_loader = DataLoader(
        full_ds,
        batch_size=int(config["train_params"]["batch_size"]),
        sampler=sampler if sampler is not None else None,
        shuffle=(sampler is None),
        num_workers=config["train_params"].get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )

    model = TransformerClassifier(
        model_name=config["model_name"],
        num_labels=len(label_cols),
        dropout=config["peft"].get("dropout", 0.1),
        use_lora=config["peft"].get("enabled", False),
        lora_r=config["peft"].get("r", 16),
        lora_alpha=config["peft"].get("alpha", 32),
        lora_dropout=config["peft"].get("dropout", 0.1),
        lora_target_modules=config["peft"].get("target_modules", None),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
    ).to(device)

    criterion = make_loss(config, device, train_label_matrix=full_ds.label_matrix)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train_params"]["lr"], weight_decay=config["train_params"]["weight_decay"])

    steps_per_epoch = math.ceil(len(full_loader) / max(1, int(config["train_params"]["grad_accum"])))
    total_updates = steps_per_epoch * int(config.get("final_training", {}).get("epochs", config["train_params"]["epochs"]))
    scheduler = make_scheduler(
        optimizer, total_updates=total_updates,
        warmup_ratio=config.get("final_training", {}).get("warmup", config["train_params"]["warmup"]),
        name=config["train_params"]["scheduler"],
    )

    precision = config.get("precision", "fp32")
    epochs = int(config.get("final_training", {}).get("epochs", config["train_params"]["epochs"]))
    for epoch in range(epochs):
        print(f"\n[Full Train] Epoch {epoch+1}/{epochs}")
        tr_loss = train_one_epoch(
            model, full_loader, optimizer, scheduler, criterion, device,
            precision=precision, grad_accum=int(config["train_params"]["grad_accum"])
        )
        print(f"[Full Train] Loss: {tr_loss:.4f}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": move_state_to_cpu(model.state_dict()), "thresholds": med_thresholds.tolist()}, save_path)
    print(f"✅ Saved final full-data model to: {save_path}")

    # cleanup
    del full_loader, full_ds, tokenizer, optimizer, scheduler, criterion, model
    free_cuda()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/train_config_19_cv.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    print(f"📁 Loading config from: {cfg_path}")
    config = read_yaml(cfg_path)

    set_seed(int(config["train_params"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_path = Path(config["data"]["raw_path"])
    if not data_path.exists():
        alt = Path("sentences.csv")
        if alt.exists():
            data_path = alt
    df = pd.read_csv(data_path)

    assert "sentence" in df.columns and "labels" in df.columns, "CSV cần cột 'sentence' và 'labels'"

    # Label list (19 nhãn) + kiểm tra
    label_cols = config["data"].get("label_list", None)
    assert label_cols and isinstance(label_cols, list), "Bạn cần 'data.label_list' trong config."
    assert len(label_cols) == int(config["num_labels"]), "num_labels không khớp với số nhãn trong label_list."

    # Chuẩn hoá lang để chia fold đúng
    df = df.copy()
    df["__lang_norm__"] = df.get("lang", "").apply(norm_lang)

    # Ma trận nhãn cho split (đã ánh xạ 19)
    labels_mat = build_label_matrix(df, label_cols, expand_to_19=bool(config["data"].get("expand_to_19_classes", False)))

    results_dir = Path(config["logging"]["output_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    fold_states = []
    if bool(config.get("use_single_split", False)):
        print("\n🔹 Using SINGLE stratified train/test split (80/20)")
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=int(config["train_params"]["seed"]))
        X = df["sentence"].values
        (tr_idx, va_idx) = next(msss.split(X, labels_mat))

        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)
        best_state = train_fold(0, train_df, val_df, label_cols, config, device)
        fold_states.append(best_state)

        if config["logging"].get("save_best", True):
            torch.save(best_state["model"], results_dir / f"best_model_fold0.pt")
            with open(results_dir / f"best_thresholds_fold0.json", "w") as f:
                json.dump(best_state["thresholds"], f, indent=2)

        with open(results_dir / "results.json", "w") as f:
            json.dump({"split": "single", "metrics": best_state["metrics"], "thresholds": best_state["thresholds"], "label_list": label_cols}, f, indent=2)

        m = best_state["metrics"]
        print("\n📊 Test Set Performance:")
        print(f"  F1 (micro):   {m['f1_micro']:.4f}")
        print(f"  F1 (macro):   {m['f1_macro']:.4f}")
        print(f"  F1 (samples): {m['f1_samples']:.4f}")
        print(f"  Precision:    {m['precision_samples']:.4f}")
        print(f"  Recall:       {m['recall_samples']:.4f}")
        print(f"  ROC-AUC(macro): {m['roc_auc_macro']}")
        print(f"  PR-AUC(macro):  {m['pr_auc_macro']}")
    else:
        print("\n🔹 Using 5-FOLD Cross-Validation")
        n_splits = int(config.get("cv", {}).get("n_splits", 5))
        start_fold = int(config.get("cv", {}).get("start_fold", 0))  # NEW: resume từ fold bất kỳ (0-based)
        skip_if_exists = bool(config.get("cv", {}).get("skip_if_checkpoint_exists", False))  # NEW

        mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(config["train_params"]["seed"]))
        X = df["sentence"].values

        for fold_idx, (tr_idx, va_idx) in enumerate(mskf.split(X, labels_mat)):
            if fold_idx < start_fold:
                print(f"⏩ Skip fold {fold_idx} (start_fold={start_fold})")
                continue

            ckpt_path = results_dir / f"best_model_fold{fold_idx}.pt"
            thr_path  = results_dir / f"best_thresholds_fold{fold_idx}.json"
            if skip_if_exists and ckpt_path.exists() and thr_path.exists():
                print(f"⏩ Skip fold {fold_idx} (checkpoint exists)")
                continue

            train_df = df.iloc[tr_idx].reset_index(drop=True)
            val_df = df.iloc[va_idx].reset_index(drop=True)

            best_state = train_fold(fold_idx, train_df, val_df, label_cols, config, device)
            fold_states.append(best_state)

            if config["logging"].get("save_best", True):
                torch.save(best_state["model"], ckpt_path)
                with open(thr_path, "w") as f:
                    json.dump(best_state["thresholds"], f, indent=2)

            # giải phóng sau mỗi fold ngay tại đây cũng được (best_state chỉ giữ CPU tensors)
            free_cuda()

        if not fold_states:
            print("\nℹ️ Không có fold nào được train trong lượt này (có thể do start_fold/skip_if_checkpoint_exists).")
            return

        keys = fold_states[0]["metrics"].keys()
        avg = {k: float(np.mean([st["metrics"][k] for st in fold_states])) for k in keys}
        std = {k: float(np.std([st["metrics"][k] for st in fold_states])) for k in keys}

        # Median thresholds across folds (khuyên dùng cho deploy)
        thr_stack = np.vstack([np.array(st["thresholds"]) for st in fold_states])
        thr_median = np.median(thr_stack, axis=0)

        summary = {
            "split": "5fold",
            "fold_metrics": [st["metrics"] for st in fold_states],
            "avg_metrics": avg,
            "std_metrics": std,
            "median_thresholds": thr_median.tolist(),
            "label_list": label_cols,
        }
        with open(results_dir / "results.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\n📊 Average Performance (trained folds):")
        print(f"  F1 (micro):   {avg['f1_micro']:.4f} ± {std['f1_micro']:.4f}")
        print(f"  F1 (macro):   {avg['f1_macro']:.4f} ± {std['f1_macro']:.4f}")
        print(f"  F1 (samples): {avg['f1_samples']:.4f} ± {std['f1_samples']:.4f}")
        print(f"  Precision:    {avg['precision_samples']:.4f} ± {std['precision_samples']:.4f}")
        print(f"  Recall:       {avg['recall_samples']:.4f} ± {std['recall_samples']:.4f}")
        print(f"  ROC-AUC(macro): {avg['roc_auc_macro']}  |  PR-AUC(macro): {avg['pr_auc_macro']}")

        # (Tuỳ chọn) train lại full-data với median thresholds
        if bool(config.get("final_training", {}).get("enabled", False)):
            save_path = Path(config["final_training"]["save_path"])
            train_full(df, label_cols, config, device, save_path, med_thresholds=thr_median)

    print(f"\n✅ Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
