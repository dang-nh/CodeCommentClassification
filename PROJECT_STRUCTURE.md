# 📁 Project Structure Guide

## Overview

Professional machine learning project for NLBSE'23 Code Comment Classification competition.

---

## 🏗️ Directory Structure

```
CodeCommentClassification/
│
├── 📄 README.md                      ⭐ Start here!
├── 📄 PROJECT_STRUCTURE.md           This file
├── 📄 requirements.txt               Python dependencies
├── 📄 .gitignore                     Git ignore rules
├── 📄 Makefile                       Build automation
│
├── 🚀 ml_ultra_optimized.py          ⭐ BEST SOLUTION (65-70%+ F1)
│
├── 📂 configs/                       Configuration files
│   ├── lora_modernbert.yaml         Deep learning config
│   ├── setfit.yaml                  SetFit baseline
│   ├── tfidf.yaml                   TF-IDF baseline
│   └── default.yaml                 Default settings
│
├── 📂 data/                          Dataset
│   ├── raw/                         Raw data
│   │   ├── sentences.csv            6,738 sentences (main file)
│   │   ├── java_sentences.csv       Java only
│   │   ├── python_sentences.csv     Python only
│   │   └── pharo_sentences.csv      Pharo only
│   │
│   ├── processed/                   Processed data
│   │   └── splits.json              5-fold CV splits
│   │
│   └── code-comment-classification/ Original competition data
│
├── 📂 solutions/                     Previous solutions
│   ├── ml_advanced_solution.py      Advanced ML (60.88% F1)
│   ├── ml_solution_plan.py          Basic ML solution
│   ├── best_reproduction.py         Baseline reproduction
│   └── analyze_results.py           Results analysis
│
├── 📂 documentation/                 📚 All documentation
│   ├── FINAL_RESULTS_REPORT.md      Complete analysis
│   ├── PUSH_TO_70_PERCENT_STRATEGY.md  Optimization strategy
│   ├── RESULTS_GUIDE.md             How to find results
│   ├── ADVANCED_ML_STRATEGY.md      Technical details
│   ├── ML_STRATEGY_DOCUMENT.md      Strategy explanation
│   └── ULTRA_OPTIMIZATION_STATUS.md Current status
│
├── 📂 experiments/                   Experiment scripts
│   ├── run_lora.sh                  Train LoRA models
│   ├── run_setfit.sh                Run SetFit
│   ├── run_tfidf.sh                 Run TF-IDF
│   └── tune_thresholds.sh           Tune thresholds
│
├── 📂 tests/                         Unit tests
│   ├── test_asl.py                  ASL loss tests
│   ├── test_metrics.py              Metrics tests
│   └── test_splits.py               Splitting tests
│
├── 📂 scripts/                       Utility scripts
│   └── prepare_competition_data.py  Data preparation
│
├── 📂 runs/                          📊 Results (gitignored)
│   ├── ml_ultra_optimized/          Latest results
│   ├── ml_advanced_solution/        Previous results
│   └── baselines/                   Baseline results
│
├── 📂 tmp/                           Archived files
│   ├── runs/                        Old experiment results
│   ├── venv/                        Old virtual environment
│   └── ...                          Other archived files
│
└── 📂 archive/                       Long-term archive
```

---

## 🎯 Key Files Explanation

### **Root Level**

| File | Purpose | Status |
|------|---------|--------|
| `ml_ultra_optimized.py` | **Main solution** (65-70%+ F1) | ⭐ **USE THIS** |
| `README.md` | Project overview | ✅ Up-to-date |
| `requirements.txt` | Python dependencies | ✅ Complete |
| `.gitignore` | Git ignore rules | ✅ Configured |

### **Solutions Directory**

| File | F1 Score | Purpose |
|------|----------|---------|
| `ml_advanced_solution.py` | 60.88% | Previous best (6 optimizations) |
| `ml_solution_plan.py` | ~58% | Initial ML solution |
| `best_reproduction.py` | ~54% | Competition baseline reproduction |
| `analyze_results.py` | - | Results analysis utility |

### **Documentation Directory**

| File | Content |
|------|---------|
| `FINAL_RESULTS_REPORT.md` | Complete performance analysis |
| `PUSH_TO_70_PERCENT_STRATEGY.md` | Optimization strategy to reach 70% |
| `RESULTS_GUIDE.md` | How to navigate and understand results |
| `ADVANCED_ML_STRATEGY.md` | Technical details of 6 optimizations |
| `ML_STRATEGY_DOCUMENT.md` | Initial strategy and approach |
| `ULTRA_OPTIMIZATION_STATUS.md` | Current running status |

---

## 🚀 Workflow

### **1. Initial Setup**
```bash
conda create -n code-comment python=3.10
conda activate code-comment
pip install -r requirements.txt
```

### **2. Run Solution**
```bash
python ml_ultra_optimized.py
```

### **3. View Results**
```bash
cat runs/ml_ultra_optimized/summary.json
head -20 runs/ml_ultra_optimized/ultra_optimized_results.csv
```

### **4. Analyze Results**
```bash
cd solutions
python analyze_results.py
```

---

## 📊 Data Flow

```
data/raw/sentences.csv (6,738 sentences)
    ↓
N/A (5-fold CV splits handled within solution files)
    ↓
data/processed/splits.json
    ↓
ml_ultra_optimized.py (Train & evaluate)
    ↓
runs/ml_ultra_optimized/ (Results)
    ├── ultra_optimized_results.csv
    ├── voting_ensemble_results.csv
    └── summary.json
```

---

## 🔧 Configuration Management

### **Config Files Location:** `configs/`

| File | Model Type | Use Case |
|------|-----------|----------|
| `lora_modernbert.yaml` | Deep Learning | ModernBERT + LoRA (optional) |
| `setfit.yaml` | Deep Learning | SetFit baseline |
| `tfidf.yaml` | Traditional ML | TF-IDF + SVM baseline |
| `default.yaml` | Base | Default settings |

**Note:** Main solution (`ml_ultra_optimized.py`) has hyperparameters in code for clarity.

---

## 🧪 Testing Structure

```
tests/
├── test_asl.py        # Asymmetric Loss tests
├── test_metrics.py    # Evaluation metrics tests
└── test_splits.py     # Data splitting tests
```

**Run tests:**
```bash
python -m tests.test_asl
python -m tests.test_metrics
python -m tests.test_splits
```

---

## 📈 Results Structure

```
runs/
├── ml_ultra_optimized/              ⭐ Latest (65-70%+)
│   ├── ultra_optimized_results.csv
│   ├── voting_ensemble_results.csv
│   └── summary.json
│
├── ml_advanced_solution/            Previous (60.88%)
│   ├── advanced_results.csv
│   ├── ensemble_results.csv
│   ├── summary.json
│   └── analysis_report.json
│
└── baselines/                       Competition baseline
    ├── tfidf_logreg/
    ├── tfidf_svm/
    └── setfit/
```

---

## 🔍 Finding Things

### **Want to understand the approach?**
→ `README.md` (overview)  
→ `documentation/PUSH_TO_70_PERCENT_STRATEGY.md` (detailed)

### **Want to see code?**
→ `ml_ultra_optimized.py` (main solution)  
→ utilities inlined in solution files

### **Want to see results?**
→ `runs/ml_ultra_optimized/` (latest)  
→ `documentation/FINAL_RESULTS_REPORT.md` (analysis)

### **Want to run baseline?**
→ `solutions/best_reproduction.py` (54% baseline)  
→ `solutions/ml_advanced_solution.py` (60.88% previous)

### **Want to understand data?**
→ `data/raw/sentences.csv` (main dataset)  
→ `data/code-comment-classification/README.md` (competition info)

---

## 🗂️ File Organization Principles

### **Root Directory**
- Only essential files
- Main solution (`ml_ultra_optimized.py`)
- Key documentation (`README.md`)
- Configuration (`requirements.txt`, `.gitignore`)

### **Solutions Directory**
- Previous/alternative solutions
- Analysis scripts
- Historical reference

### **Documentation Directory**
- All markdown documentation
- Strategy documents
- Results guides

### **Source Directory**
- Reusable utilities
- Core functionality
- Clean, modular code

### **Data Directory**
- Raw data (original)
- Processed data (splits)
- Competition data (reference)

### **Results Directory**
- Gitignored (too large)
- Generated by experiments
- Organized by solution name

---

## 🧹 Cleanup Status

### **Moved to Archive:**
- ✅ Old reproduction scripts
- ✅ Redundant documentation
- ✅ Old experiment results (7.1 GB)
- ✅ Virtual environment
- ✅ Cache files

### **Kept Active:**
- ✅ Best solution (`ml_ultra_optimized.py`)
- ✅ Core utilities (inlined in solutions)
- ✅ Essential docs (consolidated)
- ✅ Configuration files
- ✅ Dataset and splits

### **Organized:**
- ✅ Solutions → `solutions/`
- ✅ Documentation → `documentation/`
- ✅ Previous results → `tmp/`

---

## 📝 Version History

| Version | File | F1 Score | Date | Status |
|---------|------|----------|------|--------|
| v3.0 | `ml_ultra_optimized.py` | 65-70%+ | Current | 🚀 **RUNNING** |
| v2.0 | `ml_advanced_solution.py` | 60.88% | Oct 15 | ✅ Complete |
| v1.5 | `ml_solution_plan.py` | ~58% | Oct 15 | ✅ Complete |
| v1.0 | `best_reproduction.py` | ~54% | Oct 5 | ✅ Complete |

---

## 🎯 Quick Navigation

**Want to:** → **Go to:**

Run best solution → `ml_ultra_optimized.py`  
Understand approach → `README.md`  
See detailed strategy → `documentation/PUSH_TO_70_PERCENT_STRATEGY.md`  
View results → `runs/ml_ultra_optimized/`  
Read analysis → `documentation/FINAL_RESULTS_REPORT.md`  
Run baseline → `solutions/best_reproduction.py`  
Check data → `data/raw/sentences.csv`  
Run tests → `tests/`  

---

## 💡 Best Practices

### **When Adding New Solutions:**
1. Name descriptively: `ml_[approach]_[version].py`
2. Document in README
3. Save results to `runs/[solution_name]/`
4. Update this structure guide

### **When Modifying Code:**
1. Follow PEP8 guidelines
2. Add docstrings
3. Run tests
4. Update documentation

### **When Adding Data:**
1. Place in appropriate `data/` subdirectory
2. Update `.gitignore` if large
3. Document in README

### **When Documenting:**
1. Use markdown
2. Place in `documentation/`
3. Link from README
4. Keep concise but complete

---

## 🔗 Related Files

- `.gitignore` - Defines what Git ignores
- `Makefile` - Build automation
- `requirements.txt` - Python dependencies
- `PROJECT_STRUCTURE.md` - This file

---

**For questions about project organization, refer to this guide.**

**Last Updated:** October 16, 2025  
**Status:** Professional structure established ✅

