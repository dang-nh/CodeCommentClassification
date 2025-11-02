# Code Comment Classification

This repository contains the official implementation for the paper **"A Deep Learning Approach for Multi-Label Code Comment Classification"**. We systematically investigate both traditional machine learning and advanced deep learning techniques for classifying code comments into 16 functional categories.

Our best-performing model, based on **CodeBERT**, achieves a **71% F1 score**, significantly outperforming traditional ML baselines. This work was developed for the NLBSE'23 Tool Competition.

## Performance Highlights

| Model Approach      | Average F1-Score |
| ------------------- | ---------------- |
| **Deep Learning (CodeBERT)** | **71.2%**        |
| Traditional ML      | ~65%             |
| Competition Baseline| 54.0%            |

## Methodology

We evaluate and compare two primary approaches for this multi-label classification task.

### 1. Deep Learning Approach (Recommended)

Our state-of-the-art solution leverages a pre-trained Transformer model, fine-tuned for this specific task. The architecture and training are designed to maximize performance on code-related text.

-   **Model Architecture**: We use `microsoft/codebert-base`, a powerful model pre-trained on a massive corpus of source code. The base model is augmented with a custom classification head consisting of a multi-layer perceptron with `GELU` activations, `LayerNorm`, and `Dropout` for regularization. This head takes the final hidden state of the `[CLS]` token and projects it into the 16-dimensional label space.

-   **Loss Function**: To address the significant class imbalance inherent in the dataset, we employ advanced loss functions. The primary configuration uses **Asymmetric Loss (ASL)**, which dynamically down-weights the contribution of easy negative samples, allowing the model to focus on more challenging examples. The framework also supports Focal Loss as an alternative.

-   **Training and Fine-Tuning**: The model is trained using the AdamW optimizer with a cosine learning rate scheduler and warmup. We utilize full-model fine-tuning for our primary results, though the framework also supports more efficient fine-tuning via Low-Rank Adaptation (LoRA). The training process includes early stopping to prevent overfitting.

-   **Threshold Optimization**: A critical post-processing step involves optimizing the decision threshold for each of the 16 labels independently. After training, we evaluate a range of thresholds (from 0.1 to 0.9) on the validation set probabilities, selecting the threshold for each class that maximizes its F1 score. This step alone contributes significantly to the final performance.

-   **Evaluation**: The model is rigorously evaluated using either a single, stratified 80/20 train/test split (`MultilabelStratifiedShuffleSplit`) or a 5-fold stratified cross-validation strategy to ensure robust and reliable performance metrics.

### 2. Traditional Machine Learning Approach

As a comprehensive baseline, we implemented a sophisticated traditional ML pipeline featuring extensive feature engineering and an optimized ensemble model.

-   **Feature Engineering**: Our pipeline generates a rich, high-dimensional feature set from the raw text:
    -   **Text Representations**: We create a combined feature space from three different vectorization techniques:
        1.  **Word TF-IDF**: Captures the importance of word n-grams (1 to 3).
        2.  **Character TF-IDF**: Models sub-word patterns using character n-grams (3 to 5).
        3.  **Word Counts**: Binary features indicating the presence of word n-grams (1 to 2).
    -   **Hand-Crafted NLP Features**: Over 60 domain-specific features are extracted to capture code-specific and linguistic patterns. These include the presence of Javadoc tags (`@param`, `@return`), code-like structures (`()`, `{}`), text statistics (word count, uppercase ratio), and linguistic cues (modal verbs, question words).

-   **Feature Selection**: From the combined set of over 25,000 features, we use a `SelectKBest` strategy with a Chi-squared (`chi2`) test to retain the top 5,000 most discriminative features.

-   **Class Imbalance**: We apply **SMOTE** (Synthetic Minority Over-sampling Technique) during training for labels where the minority class is heavily underrepresented.

-   **Ensemble Model**: The final classifier is a **Stacking Ensemble**.
    -   **Base Models**: A diverse set of classifiers serves as the first level of the stack: Logistic Regression, Linear SVC, and Random Forest.
    -   **Meta-Classifier**: A `GradientBoostingClassifier` is used as the final estimator, which learns to combine the predictions from the base models to produce the final output.

-   **Evaluation**: Each model within the pipeline is evaluated using a 5-fold stratified cross-validation.

## Getting Started

### Prerequisites

-   Python 3.10+
-   Conda or another environment manager

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/CodeCommentClassification.git
    cd CodeCommentClassification
    ```

2.  **Create and activate the conda environment:**
    ```bash
    conda create -n code-comment python=3.10
    conda activate code-comment
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Solutions

-   **To run the Deep Learning solution (recommended):**
    ```bash
    python dl_solution.py configs/dl_best_config.yaml
    ```
    -   **Expected F1 Score**: ~71%
    -   **Runtime**: Approximately 2-3 hours on a GPU.
    -   **Output**: Results will be saved in the `runs/dl_solution/` directory.

-   **To run the Traditional ML solution:**
    ```bash
    python ml_ultra_optimized.py
    ```
    -   **Expected F1 Score**: ~65%
    -   **Runtime**: Approximately 2 hours on a CPU.
    -   **Output**: Results will be saved in `runs/ml_ultra_optimized/`.

## Project Structure

```
CodeCommentClassification/
├── dl_solution.py                 # 🏆 Deep Learning (71% F1) - BEST
├── ml_ultra_optimized.py          # Traditional ML (~65% F1)
├── configs/                       # Model configurations
│   ├── dl_best_config.yaml        # CodeBERT (recommended)
│   ├── dl_graphcodebert.yaml      # GraphCodeBERT
│   ├── dl_codet5_config.yaml      # CodeT5
│   └── ...
├── data/
│   ├── raw/
│   │   └── sentences.csv
│   └── processed/
│       └── splits.json
├── docs/                          # Documentation
├── scripts/                       # Utility scripts
├── tests/                         # Unit tests
└── runs/                          # Results (gitignored)
```

## Citation

If you use this work, please cite our paper:

```bibtex
@inproceedings{your-name-2023-codecomment,
    title={A Deep Learning Approach for Multi-Label Code Comment Classification},
    author={Your Name},
    booktitle={Proceedings of the International Conference on...},
    year={2023}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
