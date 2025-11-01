# Scripts Directory

This directory contains utility scripts and experiment runners for the Code Comment Classification project.

## 📁 Structure

```
scripts/
├── README.md                    # This file
├── monitor_training.sh          # Training monitoring script
├── prepare_competition_data.py  # Data preparation utility
└── utils/                       # Utility scripts
    ├── analyze_comment_lengths.py  # Token length analysis
    ├── choose_approach.py          # System check and approach selection
    ├── compare_ml_dl.py            # ML vs DL comparison
    └── verify_setup.py             # Setup verification
```

## 🔧 Utility Scripts

### `utils/choose_approach.py`
Check system capabilities and recommend the best approach (ML or DL).

```bash
python scripts/utils/choose_approach.py
```

### `utils/compare_ml_dl.py`
Compare results from ML and DL solutions.

```bash
python scripts/utils/compare_ml_dl.py
```

### `utils/analyze_comment_lengths.py`
Analyze comment lengths and tokenization statistics.

```bash
python scripts/utils/analyze_comment_lengths.py
```

### `utils/verify_setup.py`
Verify that all required files and directories are present.

```bash
python scripts/utils/verify_setup.py
```

### `prepare_competition_data.py`
Prepare competition data from raw sources.

```bash
python scripts/prepare_competition_data.py
```

### `monitor_training.sh`
Monitor training progress (bash script).

```bash
bash scripts/monitor_training.sh
```

## 📝 Notes

- All utility scripts are located in `scripts/utils/` for easy discovery
- Main solution scripts (`dl_solution.py`, `ml_ultra_optimized.py`) remain in the project root as entry points
- Experiment scripts are in the `experiments/` directory

