import torch
import sys


def check_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return True, gpu_name, gpu_memory
    return False, None, None


def main():
    print("="*80)
    print("CODE COMMENT CLASSIFICATION - APPROACH SELECTOR")
    print("="*80)
    
    has_gpu, gpu_name, gpu_memory = check_gpu()
    
    print("\n📊 System Information:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    
    if has_gpu:
        print(f"  GPU: ✅ {gpu_name}")
        print(f"  GPU Memory: {gpu_memory:.1f} GB")
    else:
        print(f"  GPU: ❌ Not available")
    
    print("\n" + "="*80)
    print("APPROACH COMPARISON")
    print("="*80)
    
    print("\n🔥 Deep Learning (CodeBERT)")
    print("  Expected F1: 75-85%")
    print("  Training Time: 2-3 hours (GPU) / 8-12 hours (CPU)")
    print("  Memory: 2 GB GPU / 16 GB RAM")
    print("  Hardware: GPU recommended")
    print("  Inference: 500 samples/sec")
    print("  Best For: Maximum accuracy")
    
    print("\n✅ Traditional ML (Ensemble)")
    print("  Expected F1: 60-70%")
    print("  Training Time: ~2 hours (CPU)")
    print("  Memory: 500 MB RAM")
    print("  Hardware: CPU only")
    print("  Inference: 1000 samples/sec")
    print("  Best For: Resource-constrained, fast inference")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if has_gpu and gpu_memory >= 6:
        print("\n🏆 RECOMMENDED: Deep Learning")
        print("  Reason: You have a capable GPU")
        print("  Command: python dl_solution.py")
        print("  Expected: 75-85% F1 in 2-3 hours")
        print("\n  Alternative: python ml_ultra_optimized.py (60-70% F1)")
    elif has_gpu and gpu_memory >= 4:
        print("\n✅ RECOMMENDED: Deep Learning (with reduced batch size)")
        print("  Reason: You have a GPU (limited memory)")
        print("  Command: python dl_solution.py")
        print("  Note: Edit configs/dl_optimized.yaml, set batch_size: 16")
        print("  Expected: 75-85% F1 in 3-4 hours")
        print("\n  Alternative: python ml_ultra_optimized.py (60-70% F1)")
    else:
        print("\n✅ RECOMMENDED: Traditional ML")
        print("  Reason: No GPU available")
        print("  Command: python ml_ultra_optimized.py")
        print("  Expected: 60-70% F1 in 2 hours")
        print("\n  Alternative: python dl_solution.py (75-85% F1, but 8-12 hours)")
    
    print("\n" + "="*80)
    print("QUICK DECISION GUIDE")
    print("="*80)
    
    print("\nChoose Deep Learning if:")
    print("  ✅ You have a GPU (≥4 GB)")
    print("  ✅ You want maximum accuracy (75-85% F1)")
    print("  ✅ You can wait 2-3 hours")
    print("  ✅ You need state-of-the-art results")
    
    print("\nChoose Traditional ML if:")
    print("  ✅ You don't have a GPU")
    print("  ✅ You need fast inference (1000/sec)")
    print("  ✅ You have limited memory (<2 GB)")
    print("  ✅ 60-70% F1 is sufficient")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    if has_gpu:
        print("\n1. Run Deep Learning:")
        print("   python dl_solution.py")
        print("\n2. If you want to try ML too:")
        print("   python ml_ultra_optimized.py")
        print("\n3. Compare results:")
        print("   python compare_ml_dl.py")
    else:
        print("\n1. Run Traditional ML:")
        print("   python ml_ultra_optimized.py")
        print("\n2. If you want to try DL (slow on CPU):")
        print("   python dl_solution.py")
        print("   (Warning: Will take 8-12 hours)")
        print("\n3. Compare results:")
        print("   python compare_ml_dl.py")
    
    print("\n" + "="*80)
    print("DOCUMENTATION")
    print("="*80)
    
    print("\nDeep Learning:")
    print("  • DEEP_LEARNING_APPROACH.md - Comprehensive guide")
    print("  • QUICK_START_DL.md - Quick reference")
    print("  • MODEL_RECOMMENDATIONS.md - Model selection")
    
    print("\nTraditional ML:")
    print("  • documentation/ADVANCED_ML_STRATEGY.md - ML strategy")
    print("  • documentation/FINAL_RESULTS_REPORT.md - ML results")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

