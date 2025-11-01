import sys
import os
from pathlib import Path


def check_file(path, description):
    if Path(path).exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} (MISSING)")
        return False


def check_directory(path, description):
    if Path(path).is_dir():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} (MISSING)")
        return False


def main():
    print("=" * 60)
    print("Code Comment Classification - Setup Verification")
    print("=" * 60)
    print()
    
    all_ok = True
    
    print("Configuration Files:")
    all_ok &= check_file("configs/default.yaml", "Default config")
    all_ok &= check_file("configs/lora_modernbert.yaml", "LoRA config")
    all_ok &= check_file("configs/setfit.yaml", "SetFit config")
    all_ok &= check_file("configs/tfidf.yaml", "TF-IDF config")
    print()
    
    print("Source Code:")
    all_ok &= check_file("src/__init__.py", "Package init")
    all_ok &= check_file("src/utils.py", "Utilities")
    all_ok &= check_file("src/labels.py", "Label management")
    all_ok &= check_file("src/losses.py", "Loss functions")
    all_ok &= check_file("src/data.py", "Data loading")
    all_ok &= check_file("src/split.py", "Data splitting")
    all_ok &= check_file("src/models.py", "Model definitions")
    all_ok &= check_file("src/chains.py", "Classifier chains")
    all_ok &= check_file("src/train.py", "Training script")
    all_ok &= check_file("src/thresholding.py", "Threshold tuning")
    all_ok &= check_file("src/metrics.py", "Evaluation metrics")
    all_ok &= check_file("src/inference.py", "Inference script")
    all_ok &= check_file("src/plotting.py", "Plotting utilities")
    all_ok &= check_file("src/setfit_baseline.py", "SetFit baseline")
    all_ok &= check_file("src/tfidf_baseline.py", "TF-IDF baseline")
    print()
    
    print("Experiment Scripts:")
    all_ok &= check_file("experiments/run_lora.sh", "LoRA training script")
    all_ok &= check_file("experiments/run_setfit.sh", "SetFit script")
    all_ok &= check_file("experiments/run_tfidf.sh", "TF-IDF script")
    all_ok &= check_file("experiments/tune_thresholds.sh", "Threshold tuning script")
    print()
    
    print("Tests:")
    all_ok &= check_file("tests/test_asl.py", "ASL loss tests")
    all_ok &= check_file("tests/test_splits.py", "Splitting tests")
    all_ok &= check_file("tests/test_metrics.py", "Metrics tests")
    print()
    
    print("Documentation:")
    all_ok &= check_file("README.md", "Main README")
    all_ok &= check_file("QUICKSTART.md", "Quick start guide")
    all_ok &= check_file("REPRODUCE.md", "Reproduction guide")
    all_ok &= check_file("PROJECT_SUMMARY.md", "Project summary")
    print()
    
    print("Build Files:")
    all_ok &= check_file("requirements.txt", "Python dependencies")
    all_ok &= check_file("Makefile", "Build automation")
    all_ok &= check_file(".gitignore", "Git ignore rules")
    all_ok &= check_file("run_full_pipeline.sh", "Full pipeline script")
    print()
    
    print("Directories:")
    all_ok &= check_directory("data/raw", "Raw data directory")
    all_ok &= check_directory("data/processed", "Processed data directory")
    all_ok &= check_directory("configs", "Config directory")
    all_ok &= check_directory("src", "Source directory")
    all_ok &= check_directory("experiments", "Experiments directory")
    all_ok &= check_directory("tests", "Tests directory")
    print()
    
    print("=" * 60)
    if all_ok:
        print("✓ All files and directories are present!")
        print()
        print("Next steps:")
        print("  1. Place your dataset at: data/raw/sentences.csv")
        print("  2. Run: ./run_full_pipeline.sh")
        print("  3. Or follow steps in REPRODUCE.md")
        return 0
    else:
        print("✗ Some files or directories are missing!")
        print("Please check the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
