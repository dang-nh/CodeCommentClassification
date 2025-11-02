# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Verify Setup
```bash
python test_trainer_setup.py
```

### Step 2: Run Training (Without DeepSpeed)
```bash
python dl_solution.py configs/dl_graphcodebert.yaml
```

### Step 3: Run with DeepSpeed (Multi-GPU)
```bash
deepspeed --num_gpus=2 dl_solution.py configs/dl_graphcodebert_deepspeed.yaml
```

---

## 📖 Common Commands

### Training Commands

#### Basic Training
```bash
python dl_solution.py configs/dl_graphcodebert.yaml
```

#### Advanced Training (with FGM)
```bash
python dl_solution_advanced.py configs/dl_advanced_config.yaml
```

#### Single GPU with DeepSpeed
```bash
deepspeed --num_gpus=1 dl_solution.py configs/dl_graphcodebert_deepspeed.yaml
```

#### Multi-GPU Training
```bash
deepspeed --num_gpus=4 dl_solution.py configs/dl_graphcodebert_deepspeed.yaml
```

### Monitoring

#### TensorBoard
```bash
tensorboard --logdir runs/dl_solution/fold_0/logs
```

#### Watch Training
```bash
watch -n 1 nvidia-smi
```

---

## 🔧 Quick Configuration

### Add DeepSpeed to Any Config

Edit your YAML file:
```yaml
model_name: "microsoft/graphcodebert-base"
num_labels: 16
deepspeed: "configs/ds_config_zero2.json"  # ← Add this line

train_params:
  batch_size: 32
  epochs: 10
  ...
```

### Choose DeepSpeed Stage

| Config | Use When | Memory Savings |
|--------|----------|----------------|
| `ds_config_zero1.json` | Multiple GPUs, plenty of memory | Low (~7x) |
| `ds_config_zero2.json` | Limited GPU memory | Medium (~8x) |
| `ds_config_zero3.json` | Very limited memory | High (~15x+) |

---

## 💡 Quick Fixes

### Out of Memory?
1. Reduce batch size:
   ```yaml
   train_params:
     batch_size: 16  # Was 32
   ```

2. Or use ZeRO-3:
   ```yaml
   deepspeed: "configs/ds_config_zero3.json"
   ```

### Training Too Slow?
1. Use ZeRO-1:
   ```yaml
   deepspeed: "configs/ds_config_zero1.json"
   ```

2. Or remove DeepSpeed:
   ```yaml
   # deepspeed: "configs/ds_config_zero2.json"  # Comment out
   ```

### Need More GPUs?
```bash
deepspeed --num_gpus=8 dl_solution.py config.yaml
```

---

## 📊 Example Results

### Without DeepSpeed
- Training time: 2h 30m
- GPU memory: 22GB
- GPUs needed: 1x RTX 3090 (24GB)

### With DeepSpeed ZeRO-2
- Training time: 1h 45m (4 GPUs)
- GPU memory: 12GB per GPU
- GPUs needed: 4x RTX 3060 (12GB)

### With DeepSpeed ZeRO-3
- Training time: 3h 15m
- GPU memory: 8GB
- GPUs needed: 1x RTX 3070 (8GB)

---

## 🎯 Best Practices

1. **Start simple**: Run without DeepSpeed first
2. **Test locally**: Validate on small dataset
3. **Monitor GPU**: Use `nvidia-smi` or `nvtop`
4. **Use TensorBoard**: Track metrics in real-time
5. **Save configs**: Version control your YAML files

---

## 📚 More Info

- **Detailed Guide**: See `DEEPSPEED_USAGE.md`
- **Migration**: See `MIGRATION_GUIDE.md`
- **Summary**: See `UPDATE_SUMMARY.md`

---

## 🆘 Need Help?

### Check Logs
```bash
tail -f runs/dl_solution/fold_0/logs/events.out.tfevents.*
```

### Test Setup
```bash
python test_trainer_setup.py
```

### Common Issues

**"deepspeed not found"**
```bash
pip install deepspeed
```

**"CUDA out of memory"**
- Use smaller batch size
- Use ZeRO-3
- Reduce max_len

**"Training stuck"**
- Check GPU utilization with `nvidia-smi`
- Verify data loading with `num_workers`
- Check TensorBoard logs

---

That's it! You're ready to train with Trainer API + DeepSpeed! 🎉

