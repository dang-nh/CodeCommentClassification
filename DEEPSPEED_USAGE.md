# DeepSpeed Training Guide

This document describes how to use the updated training scripts with DeepSpeed optimization.

## Changes Overview

Both `dl_solution.py` and `dl_solution_advanced.py` have been updated to use:
- **Hugging Face Trainer API** instead of custom training loops
- **DeepSpeed integration** for distributed training and memory optimization

## Key Updates

### 1. Trainer API Integration
- Replaced custom `train_epoch()` and `evaluate()` functions with `CustomTrainer` class
- Uses `TrainingArguments` for configuration
- Automatic mixed precision training (FP16/BF16)
- Built-in logging, checkpointing, and early stopping

### 2. DeepSpeed Support
- Three pre-configured DeepSpeed configs:
  - `configs/ds_config_zero1.json` - ZeRO Stage 1 (optimizer state partitioning)
  - `configs/ds_config_zero2.json` - ZeRO Stage 2 (optimizer + gradient partitioning)
  - `configs/ds_config_zero3.json` - ZeRO Stage 3 (optimizer + gradient + parameter partitioning)

### 3. Custom Features Preserved
- **Custom loss functions** (Asymmetric Loss, Focal Loss)
- **LoRA/PEFT support**
- **Threshold optimization** for multi-label classification
- **FGM adversarial training** (in advanced version)
- **Data augmentation** (in advanced version)
- **Multi-sample dropout** (in advanced version)

## Installation

Install DeepSpeed:
```bash
pip install deepspeed
```

Or with all dependencies:
```bash
pip install deepspeed[1bit,autotuning,cpu_adam,cpu_lion]
```

## Usage

### Basic Training (Without DeepSpeed)

```bash
python dl_solution.py configs/dl_graphcodebert.yaml
```

### Training with DeepSpeed (Single GPU)

1. Add DeepSpeed config to your YAML file:
```yaml
deepspeed: "configs/ds_config_zero2.json"
```

2. Run with DeepSpeed launcher:
```bash
deepspeed --num_gpus=1 dl_solution.py configs/dl_graphcodebert_deepspeed.yaml
```

### Training with DeepSpeed (Multi-GPU)

```bash
deepspeed --num_gpus=4 dl_solution.py configs/dl_graphcodebert_deepspeed.yaml
```

### Training with DeepSpeed (Multi-Node)

Create a hostfile `hostfile.txt`:
```
worker1 slots=4
worker2 slots=4
```

Run:
```bash
deepspeed --hostfile=hostfile.txt \
          --master_addr=worker1 \
          --master_port=29500 \
          dl_solution.py configs/dl_graphcodebert_deepspeed.yaml
```

## Configuration

### YAML Config File

Add the `deepspeed` parameter to your config:

```yaml
model_name: "microsoft/graphcodebert-base"
tokenizer_name: "microsoft/graphcodebert-base"
num_labels: 16
max_len: 128
precision: "fp16"  # or "bf16" for bfloat16
deepspeed: "configs/ds_config_zero2.json"  # Add this line

train_params:
  batch_size: 32
  grad_accum: 4
  epochs: 10
  lr: 0.0002
  scheduler: "cosine"
  warmup: 0.1
  weight_decay: 0.01
  seed: 42
```

### DeepSpeed Config Selection

Choose based on your GPU memory:

| Config | Memory Savings | Speed | Use Case |
|--------|---------------|-------|----------|
| ZeRO-1 | Low (~7.5x) | Fastest | Multiple GPUs, large memory |
| ZeRO-2 | Medium (~8x) | Fast | Limited GPU memory |
| ZeRO-3 | High (~15x+) | Slower | Very limited memory, very large models |

### DeepSpeed Config Customization

Edit the JSON files to customize:

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",  // Offload to CPU to save GPU memory
      "pin_memory": true
    },
    "offload_param": {  // Only in ZeRO-3
      "device": "cpu",
      "pin_memory": true
    }
  },
  "gradient_accumulation_steps": "auto",  // Syncs with TrainingArguments
  "train_micro_batch_size_per_gpu": "auto",
  "fp16": {
    "enabled": true
  }
}
```

## Advanced Features

### FGM Adversarial Training (Advanced Version)

```yaml
use_fgm: true  # Enable FGM adversarial training
```

The `CustomTrainer` in `dl_solution_advanced.py` integrates FGM with DeepSpeed:

```python
trainer = CustomTrainer(
    model=model,
    args=training_args,
    custom_loss_fn=criterion,
    use_fgm=True  # Enable adversarial training
)
```

### Data Augmentation (Advanced Version)

```yaml
use_augmentation: true
augment_p: 0.3  # Augmentation probability
```

### Multi-Sample Dropout (Advanced Version)

```yaml
use_multisample_dropout: true
```

### Class Weights (Advanced Version)

```yaml
use_class_weights: true  # Balance imbalanced labels
```

## Performance Tips

### 1. Batch Size Optimization
With DeepSpeed, you can use larger effective batch sizes:
```yaml
train_params:
  batch_size: 16  # Per-device batch size
  grad_accum: 8   # Gradient accumulation steps
  # Effective batch size = 16 * 8 * num_gpus
```

### 2. Memory Optimization
- Use ZeRO-2 or ZeRO-3 for models that don't fit in GPU memory
- Enable CPU offloading in DeepSpeed config
- Reduce `max_len` or `batch_size` if OOM occurs

### 3. Speed Optimization
- Use ZeRO-1 if memory allows
- Disable CPU offloading if you have enough GPU memory
- Use `bf16` instead of `fp16` on Ampere+ GPUs (A100, RTX 3090+)

### 4. Gradient Accumulation
Increase `grad_accum` to simulate larger batch sizes:
```yaml
train_params:
  batch_size: 8
  grad_accum: 16  # Effective batch = 8 * 16 = 128
```

## Monitoring

### TensorBoard
Monitor training progress:
```bash
tensorboard --logdir runs/dl_solution/fold_0/logs
```

### DeepSpeed Profiling
Enable profiling in DeepSpeed config:
```json
{
  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1
  }
}
```

## Troubleshooting

### CUDA Out of Memory
1. Reduce `batch_size`
2. Increase `grad_accum`
3. Use ZeRO-3 instead of ZeRO-2
4. Enable CPU offloading
5. Reduce `max_len`

### Slow Training
1. Use ZeRO-1 instead of ZeRO-3
2. Disable CPU offloading
3. Reduce `gradient_accumulation_steps`
4. Use multiple GPUs

### DeepSpeed Not Found
```bash
pip install deepspeed
# Or for specific CUDA version:
DS_BUILD_OPS=1 pip install deepspeed
```

### Version Incompatibility
Ensure compatible versions:
```bash
pip install transformers>=4.30.0 deepspeed>=0.9.0 torch>=2.0.0
```

## Example Commands

### Train GraphCodeBERT with ZeRO-2 on 2 GPUs
```bash
deepspeed --num_gpus=2 \
          dl_solution.py \
          configs/dl_graphcodebert_deepspeed.yaml
```

### Train Advanced Model with ZeRO-3
```bash
deepspeed --num_gpus=4 \
          dl_solution_advanced.py \
          configs/dl_advanced_config.yaml
```

### Resume Training
```bash
deepspeed --num_gpus=2 \
          dl_solution.py \
          configs/dl_graphcodebert_deepspeed.yaml \
          --resume_from_checkpoint runs/dl_solution/fold_0/checkpoint-1000
```

## Key Differences from Original Code

| Feature | Original | Updated |
|---------|----------|---------|
| Training Loop | Custom | Trainer API |
| Mixed Precision | Manual GradScaler | Built-in FP16/BF16 |
| Distributed Training | Not supported | DeepSpeed ZeRO |
| Checkpointing | Manual | Automatic |
| Logging | Manual | TensorBoard integrated |
| Early Stopping | Manual | Built-in |
| Gradient Clipping | Manual | Automatic |
| Learning Rate Schedule | Manual | Automatic |

## References

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [Hugging Face Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [DeepSpeed Integration with Transformers](https://huggingface.co/docs/transformers/main_classes/deepspeed)

