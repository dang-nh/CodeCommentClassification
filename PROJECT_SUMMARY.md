# Code Comment Classification - Project Summary

## Overview

Complete, production-ready implementation of multi-label code comment classification using ModernBERT-base with LoRA and Asymmetric Loss (ASL). Designed to beat 2023 baselines while staying GPU-friendly (24GB VRAM max).

## Key Features

### Core Model
- **Architecture**: ModernBERT-base (149M params) with LoRA (r=8, alpha=16)
- **Loss Function**: Asymmetric Loss (ASL) with gamma_pos=0, gamma_neg=4, clip=0.05
- **Optimization**: AdamW with cosine schedule, 10% warmup, lr=2e-4
- **Precision**: bfloat16 with gradient checkpointing
- **Memory**: ~18-20GB VRAM with batch_size=48, max_len=128

### Advanced Features
- **Language-aware tokenization**: Prepends `[JAVA]`, `[PY]`, or `[PHARO]` tokens
- **Group-aware stratification**: Ensures no class_id leakage across splits
- **Per-label threshold tuning**: Maximizes F1 for each label independently
- **Classifier chains** (optional): Sequential label prediction with probability feedback
- **5-fold ensemble**: Averages logits from 5 models for robust predictions

### Baselines
1. **SetFit**: Sentence-Transformers (MiniLM) + linear head
2. **TF-IDF + Linear SVM**: Fast classical baseline

## File Structure

```
code-comment-classification/
├── README.md                    # Main documentation
├── QUICKSTART.md               # Quick start guide
├── PROJECT_SUMMARY.md          # This file
├── requirements.txt            # Python dependencies
├── Makefile                    # Build automation
├── .gitignore                  # Git ignore rules
├── run_full_pipeline.sh        # Complete pipeline script
│
├── configs/                    # Configuration files
│   ├── default.yaml           # Base configuration
│   ├── lora_modernbert.yaml   # Main model config
│   ├── setfit.yaml            # SetFit baseline config
│   └── tfidf.yaml             # TF-IDF baseline config
│
├── data/                       # Data directory
│   ├── raw/                   # Raw data (sentences.csv)
│   └── processed/             # Processed splits and labels
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── utils.py               # Utilities (seeding, logging, config)
│   ├── labels.py              # Label management and encoding
│   ├── losses.py              # ASL and BCE loss implementations
│   ├── data.py                # Dataset loading and tokenization
│   ├── split.py               # Group-aware stratified splitting
│   ├── models.py              # ModernBERT + LoRA model
│   ├── chains.py              # Classifier chains wrapper
│   ├── train.py               # Training loop
│   ├── thresholding.py        # Per-label threshold tuning
│   ├── metrics.py             # Evaluation metrics
│   ├── inference.py           # Ensemble inference
│   ├── plotting.py            # PR curves and visualizations
│   ├── setfit_baseline.py     # SetFit baseline
│   └── tfidf_baseline.py      # TF-IDF baseline
│
├── experiments/                # Experiment scripts
│   ├── run_lora.sh            # Train 5 folds with LoRA
│   ├── run_setfit.sh          # Run SetFit baseline
│   ├── run_tfidf.sh           # Run TF-IDF baseline
│   └── tune_thresholds.sh     # Tune thresholds for all folds
│
├── tests/                      # Unit tests
│   ├── test_asl.py            # ASL loss tests
│   ├── test_splits.py         # Splitting tests
│   └── test_metrics.py        # Metrics tests
│
├── runs/                       # Training outputs (created at runtime)
└── plots/                      # Evaluation plots (created at runtime)
```

## Usage

### Complete Pipeline (One Command)

```bash
./run_full_pipeline.sh
```

### Step-by-Step

```bash
# 1. Create splits
python -m src.split --input data/raw/sentences.csv --out data/processed/splits.json --test_size 0.2 --folds 5

# 2. Train 5 folds
for fold in {0..4}; do
    python -m src.train --config configs/lora_modernbert.yaml --fold $fold
done

# 3. Tune thresholds
for fold in {0..4}; do
    python -m src.thresholding --preds runs/fold_${fold}/val_preds.npy --labels runs/fold_${fold}/val_labels.npy --out runs/fold_${fold}/thresholds.json
done

# 4. Run ensemble inference
python -m src.inference --config configs/lora_modernbert.yaml --ckpts "runs/fold_*/best.pt" --ensemble mean --out runs/test_preds.csv --test

# 5. Generate plots
python -m src.plotting --preds runs/test_preds.csv --labels data/processed/test_labels.npy --out plots/
```

## Output Files

After running the complete pipeline:

- `data/processed/splits.json` - Train/test splits with fold indices
- `runs/fold_*/best.pt` - Best model checkpoint for each fold
- `runs/fold_*/val_preds.npy` - Validation predictions for threshold tuning
- `runs/fold_*/thresholds.json` - Optimal per-label thresholds
- `runs/test_preds.csv` - Final ensemble predictions on test set
- `runs/test_preds_metrics.json` - Comprehensive evaluation metrics
- `plots/pr_curves_all.png` - PR curves for all labels
- `plots/pr_curve_{label}.png` - Individual PR curve per label
- `plots/label_distribution.png` - Label frequency distribution

## Metrics

The system computes:

- **Per-label**: Precision, Recall, F1, PR-AUC
- **Aggregate**: Micro-F1, Macro-F1, Micro-PR-AUC, Macro-PR-AUC
- **Visualizations**: PR curves, label distributions

## Configuration

All hyperparameters are in YAML files:

### Model Configuration
- `model_name`: Encoder model (default: `answerdotai/ModernBERT-base`)
- `num_labels`: Number of output labels (default: 19)
- `max_len`: Maximum sequence length (default: 128)
- `precision`: Training precision (`bfloat16`, `fp16`, or `fp32`)
- `gradient_checkpointing`: Enable to save memory (default: true)

### LoRA Configuration
- `peft.enabled`: Enable LoRA (default: true)
- `peft.r`: LoRA rank (default: 8)
- `peft.alpha`: LoRA alpha (default: 16)
- `peft.dropout`: LoRA dropout (default: 0.05)

### Loss Configuration
- `loss_type`: Loss function (`asl` or `bce`)
- `loss_params.gamma_pos`: ASL positive focusing (default: 0)
- `loss_params.gamma_neg`: ASL negative focusing (default: 4)
- `loss_params.clip`: ASL probability clipping (default: 0.05)

### Training Configuration
- `train_params.batch_size`: Batch size (default: 48)
- `train_params.grad_accum`: Gradient accumulation steps (default: 1)
- `train_params.epochs`: Training epochs (default: 10)
- `train_params.lr`: Learning rate (default: 2e-4)
- `train_params.scheduler`: LR scheduler (`cosine` or `linear`)
- `train_params.warmup`: Warmup ratio (default: 0.1)
- `train_params.weight_decay`: Weight decay (default: 0.01)

### Classifier Chains (Optional)
- `chains.enabled`: Enable classifier chains (default: false)
- `chains.num_orders`: Number of label orderings (default: 3)

## Technical Details

### Data Splitting
- Uses iterative stratified K-fold for multi-label data
- Ensures no `class_id` appears in both train and validation
- Maintains label distribution across folds

### Loss Function
- Asymmetric Loss (ASL) addresses class imbalance
- Focuses on hard negatives (gamma_neg=4)
- Clips negative probabilities to prevent overconfidence

### Threshold Tuning
- Searches thresholds in [0.1, 0.9] with 81 steps
- Maximizes F1 score per label independently
- Applied during inference for optimal predictions

### Ensemble
- Averages logits (not probabilities) from 5 folds
- More stable than probability averaging
- Supports mean or median aggregation

## Performance Expectations

With default settings on ~19 labels:

| Method | Micro-F1 | Macro-F1 | Training Time (5 folds) |
|--------|----------|----------|-------------------------|
| ModernBERT + LoRA + ASL | >0.85 | >0.75 | ~3-5 hours |
| SetFit (MiniLM) | ~0.75-0.80 | ~0.65-0.70 | ~30-60 min |
| TF-IDF + SVM | ~0.65-0.70 | ~0.55-0.60 | ~5-10 min |

## Memory Optimization

If running out of memory:

1. Reduce `batch_size` (e.g., 32 or 24)
2. Increase `grad_accum` to maintain effective batch size
3. Reduce `max_len` (e.g., 96 or 64)
4. Use `fp16` instead of `bfloat16` (slightly less stable)
5. Disable `gradient_checkpointing` if you have enough memory

## Reproducibility

- All random seeds are set (123, 456, 789)
- Deterministic CUDA operations enabled
- Fixed split indices saved in `splits.json`
- Model checkpoints include full state

## Testing

Run unit tests:

```bash
python -m tests.test_asl        # Test ASL loss
python -m tests.test_splits     # Test splitting logic
python -m tests.test_metrics    # Test evaluation metrics
```

## Dependencies

Core dependencies:
- `torch==2.1.0` - Deep learning framework
- `transformers==4.36.0` - Hugging Face models
- `peft==0.7.1` - Parameter-efficient fine-tuning
- `scikit-learn==1.3.2` - Classical ML and metrics
- `iterative-stratification==0.1.7` - Multi-label stratification
- `sentence-transformers==2.2.2` - SetFit baseline

See `requirements.txt` for complete list.

## Future Enhancements

Potential improvements:
1. Enable classifier chains for sequential label prediction
2. Add focal loss as alternative to ASL
3. Implement label correlation analysis
4. Add support for more encoders (RoBERTa, ELECTRA)
5. Add hyperparameter search with Optuna
6. Implement active learning for label-scarce scenarios

## License

MIT License (see project root for details)

## Citation

If you use this code, please cite:

```bibtex
@software{code_comment_classification,
  title={Code Comment Classification with ModernBERT and LoRA},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/CodeCommentClassification}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
