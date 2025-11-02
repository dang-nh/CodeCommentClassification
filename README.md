# Code Comment Classification

This repository contains the implementation for multi-label code comment classification using transformer-based deep learning models. The final solution achieves **82.9% F1-score (samples)** using RoBERTa-large with LoRA fine-tuning and advanced training techniques.

## Performance Highlights

| Metric | Score |
|--------|-------|
| **F1-score (samples)** | **82.9%** |
| F1-score (micro) | 81.4% |
| F1-score (macro) | 79.4% |
| Precision (samples) | 82.0% |
| Recall (samples) | 86.6% |
| ROC-AUC (macro) | 94.3% |
| PR-AUC (macro) | 81.9% |

## Methodology

Our solution uses a **RoBERTa-large** transformer model fine-tuned with **LoRA (Low-Rank Adaptation)** for efficient parameter-efficient fine-tuning. The model is trained on 19 label categories (expanded from 16 base categories by separating language-specific labels).

### Key Features

- **Model Architecture**: RoBERTa-large with LoRA fine-tuning (r=32, alpha=64)
- **Loss Function**: Asymmetric Loss (ASL) to handle class imbalance
- **Threshold Optimization**: Per-label threshold optimization for F1-score maximization
- **Cross-Validation**: 5-fold multilabel stratified cross-validation
- **Training**: Weighted sampling, cosine learning rate scheduling with warmup

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Conda environment manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd CodeCommentClassification
   ```

2. **Create and activate the conda environment:**
   ```bash
   conda create -n code-comment python=3.10
   conda activate code-comment
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Training

**To train the model:**
```bash
conda activate code-comment
CUDA_VISIBLE_DEVICES=0 python train.py configs/train_config.yaml
```

The training script supports:
- **5-fold cross-validation** (default)
- **Single train/test split** (set `use_single_split: true` in config)
- **Resume training** from specific folds (set `cv.start_fold` in config)
- **Checkpoint skipping** (set `cv.skip_if_checkpoint_exists: true`)

**Expected Runtime**: Approximately 10-15 hours on a single GPU (depending on hardware).

**Output**: Results will be saved in `runs/roberta_large_ccc_19cv_fold_validation/`:
- `best_model_fold{N}.pt`: Best model checkpoint for each fold
- `best_thresholds_fold{N}.json`: Optimized thresholds for each fold
- `final_model.pt`: Model trained on full dataset (if `final_training.enabled: true`)
- `results.json`: Comprehensive metrics and summary

## Project Structure

```
CodeCommentClassification/
├── train.py                          # Main training script
├── configs/
│   └── train_config.yaml            # Training configuration
├── data/
│   ├── raw/
│   │   └── sentences.csv            # Main dataset
│   └── processed/
│       └── splits.json              # Processed splits (if any)
├── runs/
│   └── roberta_large_ccc_19cv_fold_validation/  # Final results
│       ├── best_model_fold*.pt
│       ├── best_thresholds_fold*.json
│       ├── final_model.pt
│       └── results.json
├── STACC/                           # STACC baseline (git subtree)
└── requirements.txt
```

## Configuration

The training configuration is defined in `configs/train_config.yaml`. Key parameters:

- **Model**: `roberta-large`
- **LoRA**: Enabled (r=32, alpha=64)
- **Batch Size**: 64
- **Learning Rate**: 5e-5
- **Epochs**: 100 (with early stopping)
- **Loss**: Asymmetric Loss (ASL)
- **Scheduler**: Cosine with 10% warmup

## Results

The final model achieves:
- **F1-score (samples)**: 82.9% ± 0.49%
- **F1-score (micro)**: 81.4% ± 0.43%
- **F1-score (macro)**: 79.4% ± 1.03%

See `runs/roberta_large_ccc_19cv_fold_validation/results.json` for detailed metrics.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{code-comment-classification-2024,
    title={Multi-Label Code Comment Classification using Transformer Models},
    author={Your Name},
    year={2024}
}
```

## License

This project is licensed under the MIT License.
