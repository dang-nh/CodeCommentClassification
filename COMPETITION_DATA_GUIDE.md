# NLBSE'23 Competition Data Guide

## Overview

This project uses the official NLBSE'23 Tool Competition dataset for code comment classification. The dataset includes comments from three programming languages: Java, Python, and Pharo.

## Dataset Statistics

### Overall
- **Total sentences:** 6,738
- **Training set:** 5,392 (80%)
- **Test set:** 1,346 (20%)
- **Total categories:** 16 unique labels

### By Language

| Language | Sentences | Train | Test | Categories |
|----------|-----------|-------|------|------------|
| Java | 2,418 | 1,933 | 485 | 7 |
| Python | 2,555 | 2,042 | 513 | 5 |
| Pharo | 1,765 | 1,417 | 348 | 7 |

### Categories by Language

**Java (7 categories):**
- `summary` - Summary of class functionality
- `ownership` - Ownership and authorship information
- `expand` - Expanded explanations
- `usage` - Usage examples and instructions
- `pointer` - Pointers to related code/documentation
- `deprecation` - Deprecation notices
- `rational` - Design rationale

**Python (5 categories):**
- `summary` - Summary of class functionality
- `usage` - Usage examples and instructions
- `expand` - Expanded explanations
- `parameters` - Parameter descriptions
- `developmentnotes` - Development notes

**Pharo (7 categories):**
- `intent` - Intent and purpose
- `responsibilities` - Class responsibilities
- `collaborators` - Collaborating classes
- `example` - Usage examples
- `keyimplementationpoints` - Key implementation details
- `keymessages` - Key messages/methods
- `classreferences` - References to other classes

## Data Format

### Original Competition Format

The competition provides CSV files with the following columns:
- `comment_sentence_id` - Unique sentence identifier
- `class` - Source code class name
- `comment_sentence` - The actual comment text
- `partition` - 0 for train, 1 for test
- `instance_type` - 0 for negative, 1 for positive instance
- `category` - The label category

**Note:** The original format has one row per (sentence, category) pair, so each sentence appears multiple times if it has multiple labels.

### Our Unified Format

We convert the data to a multi-label format with one row per sentence:

```csv
id,class_id,sentence,partition,labels,lang
0,Abfss.java,this impl delegates to the old filesystem,0,expand,JAVA
1,FileSystemApplicationHistoryStore.java,@link #applicationstarted...,1,expand;pointer,JAVA
```

Columns:
- `id` - Unique sentence identifier
- `class_id` - Source class (used for group-aware splitting)
- `sentence` - Comment text
- `partition` - 0 for train, 1 for test (from competition)
- `labels` - Semicolon-separated list of categories
- `lang` - Programming language (JAVA, PY, PHARO)

## Data Preparation

### Step 1: Prepare Competition Data

```bash
python scripts/prepare_competition_data.py --combine
```

This script:
1. Reads the original competition CSV files
2. Converts from per-category format to multi-label format
3. Normalizes category names to lowercase
4. Adds language identifiers
5. Creates separate files for each language + combined file

**Output files:**
- `data/raw/java_sentences.csv` - Java data only
- `data/raw/python_sentences.csv` - Python data only
- `data/raw/pharo_sentences.csv` - Pharo data only
- `data/raw/sentences.csv` - All languages combined

### Step 2: Create Custom Splits

While the competition provides a train/test split, we create our own 5-fold CV splits for robust evaluation:

```bash
python -m src.split \
    --input data/raw/sentences.csv \
    --out data/processed/splits.json \
    --test_size 0.2 \
    --folds 5
```

This creates:
- 80/20 holdout test set (group-aware, no class_id leakage)
- 5-fold CV on the 80% training data
- Stratified by label distribution
- Group-aware (no class_id in both train and val)

**Note:** You can also use the competition's original train/test split by filtering on the `partition` column.

## Baseline Results (Competition)

The competition baseline uses Random Forest with TF-IDF + NLP features:

### Java Results (F1-scores)

| Category | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| summary | 0.79 | 0.85 | 0.82 |
| ownership | 0.71 | 0.68 | 0.69 |
| expand | 0.73 | 0.76 | 0.74 |
| usage | 0.75 | 0.72 | 0.73 |
| pointer | 0.68 | 0.65 | 0.66 |
| deprecation | 0.82 | 0.79 | 0.80 |
| rational | 0.70 | 0.67 | 0.68 |

### Python Results (F1-scores)

| Category | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| summary | 0.81 | 0.83 | 0.82 |
| usage | 0.76 | 0.74 | 0.75 |
| expand | 0.72 | 0.70 | 0.71 |
| parameters | 0.78 | 0.75 | 0.76 |
| developmentnotes | 0.69 | 0.66 | 0.67 |

### Pharo Results (F1-scores)

| Category | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| intent | 0.74 | 0.77 | 0.75 |
| responsibilities | 0.71 | 0.73 | 0.72 |
| collaborators | 0.68 | 0.65 | 0.66 |
| example | 0.79 | 0.81 | 0.80 |
| keyimplementationpoints | 0.66 | 0.63 | 0.64 |
| keymessages | 0.70 | 0.68 | 0.69 |
| classreferences | 0.72 | 0.70 | 0.71 |

**Overall baseline (weighted average):** F1 ≈ 0.72-0.74

## Our Goal

Beat the competition baseline by achieving:
- **Micro-F1 > 0.80** (vs baseline ~0.72-0.74)
- **Macro-F1 > 0.75** (vs baseline ~0.70-0.72)
- **Per-label F1 improvements** across all categories

## Training Strategy

### Option 1: Language-Specific Models

Train separate models for each language:

```bash
# Java only
python -m src.split --input data/raw/java_sentences.csv --out data/processed/java_splits.json
python -m src.train --config configs/lora_modernbert.yaml --fold 0

# Python only
python -m src.split --input data/raw/python_sentences.csv --out data/processed/python_splits.json
python -m src.train --config configs/lora_modernbert.yaml --fold 0

# Pharo only
python -m src.split --input data/raw/pharo_sentences.csv --out data/processed/pharo_splits.json
python -m src.train --config configs/lora_modernbert.yaml --fold 0
```

**Advantages:**
- Specialized models per language
- Can tune hyperparameters per language
- Easier to debug and analyze

**Disadvantages:**
- Requires training 3 separate models
- Can't leverage cross-language patterns

### Option 2: Unified Multi-Language Model (Recommended)

Train a single model on all languages:

```bash
python -m src.split --input data/raw/sentences.csv --out data/processed/splits.json
python -m src.train --config configs/lora_modernbert.yaml --fold 0
```

**Advantages:**
- Single model handles all languages
- Leverages language tokens ([JAVA], [PY], [PHARO])
- Can learn cross-language patterns
- More data for training

**Disadvantages:**
- Slightly more complex
- May need more training time

## Evaluation

### Using Competition Test Set

To evaluate on the exact competition test set:

```python
import pandas as pd

df = pd.read_csv('data/raw/sentences.csv')
test_df = df[df['partition'] == 1]  # Competition test set
train_df = df[df['partition'] == 0]  # Competition train set
```

### Using Our 5-Fold CV

For more robust evaluation:

```bash
./run_full_pipeline.sh
```

This trains 5 models and ensembles predictions.

## Data Characteristics

### Label Distribution

The dataset is **imbalanced**:
- Common labels: `summary`, `expand`, `usage` (appear in all languages)
- Rare labels: `deprecation`, `rational`, `developmentnotes`
- Multi-label: ~30% of sentences have multiple labels

### Sentence Length

- **Average:** 15-20 words
- **Max:** ~100 words
- **Min:** 2-3 words

### Special Characteristics

1. **Code references:** Many sentences contain `@link`, `@code`, `@param` tags
2. **Incomplete sentences:** Some are phrases, not complete sentences
3. **Technical jargon:** Heavy use of programming terminology
4. **Cross-references:** References to classes, methods, parameters

## Tips for Better Performance

1. **Use language tokens:** Prepend [JAVA]/[PY]/[PHARO] to each sentence
2. **Handle code tags:** Keep `@link`, `@code`, etc. in the text
3. **Longer sequences:** Consider max_len=192 for longer comments
4. **Class imbalance:** Use Asymmetric Loss (ASL) instead of BCE
5. **Per-label thresholds:** Tune thresholds separately for each label
6. **Ensemble:** Train multiple models and average predictions

## Competition Rules

From the original competition:
1. Must use the provided train/test split
2. No external data allowed
3. Must report per-category P/R/F1
4. Must beat the Random Forest baseline

## Citation

If you use this dataset, cite the original paper:

```bibtex
@inproceedings{nlbse2023,
  title={NLBSE'23 Tool Competition: Code Comment Classification},
  booktitle={International Workshop on Natural Language-based Software Engineering},
  year={2023}
}
```

## Additional Resources

- Competition details: https://nlbse2023.github.io/tools/
- Original repository: https://github.com/nlbse/code-comment-classification
- Baseline results: `data/code-comment-classification/baseline_results_summary.xlsx`
