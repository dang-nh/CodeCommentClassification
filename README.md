# Code Comment Classification (Multi-Label, 16/19 classes)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-red.svg)](#)
[![HF Transformers](https://img.shields.io/badge/transformers-4.x-ff69b4.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

A compact, reproducible recipe for **multi-label code comment classification**.
We frame the task as *one* multi-label problem (not 19 separate binary tasks), and we optimize the **sample-level F1** directly via per-label threshold calibration.

**Highlights**

* Single **multi-label** model (19 outputs) with **RoBERTa-large** backbone
* **Masked mean pooling** (instead of `[CLS]`) + **context prompt**: `"[LANG] class_id | sentence"`
* **Asymmetric Loss (ASL)** for imbalance + optional **WeightedRandomSampler**
* **Per-label threshold search** (grid + coordinate descent) **in-loop**; **early stopping by F1(samples)**
* **5-fold** Multilabel Stratified CV; **median thresholds** across folds for deployment
* Trains comfortably on a single **NVIDIA L40S 46GB** (bf16/fp32)

---

## TL;DR Results (19 labels)

Cross-validated (mean ± std across trained folds):

| Metric           |               Score |
| ---------------- | ------------------: |
| **F1 (samples)** | **0.8289 ± 0.0049** |
| F1 (micro)       |     0.8138 ± 0.0043 |
| F1 (macro)       |     0.7943 ± 0.0103 |
| PR-AUC (macro)   |     0.8193 ± 0.0131 |
| ROC-AUC (macro)  |     0.9430 ± 0.0047 |

> **Protocol note**: NLBSE’23 reports *Avg F1 over 19 one-vs-rest binary tasks* (macro over labels).
> We train *a multi-label model* and report **F1 (samples)** (macro over sentences).
> To "shake hands" with the NLBSE protocol, we also report **F1 (macro)** (table above).

---

## Why Multi-Label (and not 19 models)?

In practical implementations, a multi-label model:

* **Learning joint representations** between co-occurring labels → good for rare labels, reducing variance
* **Cheap inference**: 1 forward for 19 labels (instead of 19 models)
* **Simple ML-Ops**: 1 checkpoint + 1 set of thresholds (median across folds)
* **On-target optimization**: threshold is found to **maximize global F1(samples)**

If you must compare according to the NLBSE metric, you can retrain the **19 binary classifiers** and calculate “AvgF1 + OC”; In products, multi-label is a more reasonable choice.

---

## Repository Structure

```
CodeCommentClassification/
├── train.py                       # Main training (multi-label)
├── configs/
│   └── train_config.yaml          # L40S-ready 19-class config (edit as needed)
├── data/
│   ├── raw/
│   │   └── sentences.csv          # Input CSV (required)
│   └── processed/                 # (optional) cached splits/labels
├── runs/
│   └── roberta_large_ccc_19cv_fold_validation/
│       ├── best_model_fold*.pt
│       ├── best_thresholds_fold*.json
│       ├── final_model.pt         # (if final_training.enabled: true)
│       └── results.json
├── requirements.txt
└── README.md
```

---

## Data & Labeling

* **Input CSV**: `data/raw/sentences.csv` must contain at least:

  * `sentence` (string)
  * `labels` (semicolon-separated label names)
  * `lang` (e.g., `Java`, `Python`, `Pharo`) — used for 16→19 mapping
  * (optional) `class_id` — used in context prompt

* **16 → 19 mapping** (language-aware expansion):

  ```
  summary  -> summary_java, summary_python
  usage    -> usage_java,   usage_python
  expand   -> expand_java,  expand_python
  ```

  Other labels remain language-agnostic.
  Toggle with `data.expand_to_19_classes` in config and provide `data.label_list` accordingly.

---

## Installation

```bash
# clone
git clone https://github.com/dang-nh/CodeCommentClassification.git
cd CodeCommentClassification

# Conda env (recommended)
conda create -n code-comment python=3.10 -y
conda activate code-comment

# Install deps
pip install -r requirements.txt
```

**GPU / Precision**

* L40S 46GB: `precision: "bf16"` (recommended) or `"fp32"`.
* If using FP16: enable `precision: "fp16"` (with AMP scaler), but bf16 is more stable on Hopper/ADA.

---

## Configuration (key fields)

File: `configs/train_config.yaml` (for example: 19 labels, L40S-friendly)

```yaml
model_name: "roberta-large"
tokenizer_name: "roberta-large"
num_labels: 19
max_len: 128

precision: "bf16"               # bf16 (L40S) | fp16 | fp32
gradient_checkpointing: false

use_single_split: false         # 5-fold CV when false

cv:
  n_splits: 5

text_features:
  include_lang: true
  include_class_id: true
  template: "[{lang}] {class_id} | {sentence}"

peft:
  enabled: true                 # true = enable LoRA; false = full fine-tune
  r: 32
  alpha: 64
  dropout: 0.05
  target_modules: ["query", "key", "value", "dense"]

loss_type: "asl"                # asl | focal | bce
loss_params:
  gamma_pos: 0
  gamma_neg: 4
  clip: 0.05

sampler:
  enabled: true                 # weighted instance sampling (helps rare labels)
  type: "weighted"
  smooth_eps: 1e-3

train_params:
  batch_size: 160               # fits on L40S with RoBERTa-large, seq=128
  grad_accum: 1
  epochs: 100
  lr: 0.00005
  scheduler: "cosine"
  warmup: 0.10
  weight_decay: 0.01
  seed: 42
  num_workers: 4

data:
  raw_path: "data/raw/sentences.csv"
  processed_path: "data/processed/"
  expand_to_19_classes: true
  label_list:
    - summary_java
    - pointer
    - deprecation
    - rational
    - ownership
    - usage_java
    - expand_java
    - summary_python
    - parameters
    - usage_python
    - developmentnotes
    - expand_python
    - keymessages
    - intent
    - classreferences
    - example
    - keyimplementationpoints
    - responsibilities
    - collaborators

eval:
  metrics: ["precision", "recall", "f1", "roc_auc", "pr_auc"]
  threshold_search: true

logging:
  output_dir: "runs/roberta_large_ccc_19cv_fold_validation"
  save_best: true
  early_stop: 10
  tensorboard: false

final_training:
  enabled: true
  epochs: 50
  warmup: 0.10
  save_path: "runs/roberta_large_ccc_19cv_fold_validation/final_model.pt"
```

> **LoRA note**: `peft.enabled: true` → turn on LoRA (less VRAM, faster). `false` → full fine-tune.

---

## Training

### 5-Fold Cross-Validation (default)

```bash
python train.py configs/train_config.yaml
```

Artifacts per fold will appear under `runs/.../`:

* `best_model_fold{k}.pt`
* `best_thresholds_fold{k}.json`
* `results.json` (CV summary: means, stds, thresholds)

### Single train/test split (80/20)

```yaml
# in config
use_single_split: true
```

```bash
python train.py configs/train_config.yaml
```

### Final training on full data (optional)

Enable `final_training.enabled: true` in config.
The script will train on full data and save `final_model.pt` with the **median** thresholds from CV.

---

## Reproducing Our Numbers

1. Use the provided config (19 labels, L40S friendly).
2. Run 5-fold CV: `python train.py configs/train_config.yaml`
3. Inspect `runs/.../results.json`:

   * `avg_metrics` ≈ the TL;DR table above
   * `median_thresholds` used for deployment

> Note: results depend on random split; we use seed = 42.

---

## Inference (loading model + thresholds)

Sample pseudo-code (minimal) to load checkpoint + thresholds and infer:

```python
import torch, json
import numpy as np
from transformers import AutoTokenizer
from train import TransformerClassifier, norm_lang  # from this repo

ckpt = "runs/roberta_large_ccc_19cv_fold_validation/best_model_fold0.pt"
thr_path = "runs/roberta_large_ccc_19cv_fold_validation/best_thresholds_fold0.json"
with open(thr_path) as f: thresholds = np.array(json.load(f))

label_list = [ ... ]  # same order as in config

tok = AutoTokenizer.from_pretrained("roberta-large")
model = TransformerClassifier(
    model_name="roberta-large",
    num_labels=len(label_list),
    dropout=0.05,
    use_lora=True, lora_r=32, lora_alpha=64, lora_dropout=0.05,
    gradient_checkpointing=False
)
state = torch.load(ckpt, map_location="cpu")
model.load_state_dict(state)  # if you saved only model.state_dict()
model.eval()

def build_text(sentence, lang="java", class_id="UNK"):
    return f"[{lang.upper()}] {class_id} | {sentence}"

x = build_text("This method returns the user ID.", lang="python", class_id="User.getId")
enc = tok(x, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
with torch.no_grad():
    logits = model(**enc)
    probs = torch.sigmoid(logits).numpy().squeeze()
pred = (probs >= thresholds).astype(int)
pred_labels = [lbl for lbl, b in zip(label_list, pred) if b==1]
print(pred_labels)
```

> In production, it's recommended to use **median thresholds** from CV to stabilize precision/recall.

---

## Evaluation Protocols (Important)

* **Ours**: report **F1 (samples)** (macro over sentences) for a multi-label model.
* **NLBSE’23**: report **Avg F1** (macro over labels) of **19 independent binary tasks**.
* Because these numbers are not directly comparable; we also report **F1 (macro)** for fairer reference.

---

## Troubleshooting

* **OOM (out-of-memory)**

  * Use `precision: "bf16"` (L40S) or `"fp16"`
  * Decrease `train_params.batch_size` (e.g. 128 → 96 → 64)
  * Enable `gradient_checkpointing: true`
  * Decrease `max_len: 128 → 96`
  * Enable LoRA: `peft.enabled: true` (greatly reduces VRAM)

* **F1(samples) low, macro high** → balance thresholds using the `median_thresholds` table; increase `sampler.enabled: true`.

* **Poor F1(samples), high F1(macro)** → keep ASL, enable sampler; consider decreasing `gamma_neg` (ASL) or try `loss_type: "bce"` with `pos_weight`.

---

## Extending / Ablations

* **Backbones**: try `roberta-base`, `deberta-v3-base/large` (need to adjust batch size).
* **Pooling**: compare `[CLS]` vs **masked mean** (currently default).
* **Objectives**: `asl` vs `bce` (with `pos_weight`) vs `focal`.
* **Prompting**: enable/disable `text_features` (LANG, class_id).
* **LoRA vs Full FT**: `peft.enabled: true/false`.

---

## Citation

```bibtex
@misc{dang2025ccc,
  title        = {Multi-Label Code Comment Classification using Transformer Models},
  author       = {Dang, N.H. and Duc, N.T.M},
  year         = {2025},
  note         = {GitHub: dang-nh/CodeCommentClassification}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

### Acknowledgments

* NLBSE dataset & baselines: Rani et al., Indika et al.
* STACC (SetFit/SBERT): Al-Kaswan et al.
* CodeT5, RoBERTa, HuggingFace, and the broader open-source community.
