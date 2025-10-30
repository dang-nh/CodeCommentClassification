import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    make_scorer,
    f1_score
)
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from scipy.sparse import hstack
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from src.utils import set_seed, save_json


def extract_advanced_nlp_features(text):
    """
    ADVANCED NLP feature extraction with 40+ features
    for maximum discriminative power.
    """
    features = {}
    text_lower = text.lower()
    
    features['has_link'] = int('@link' in text_lower)
    features['has_param'] = int('@param' in text_lower)
    features['has_return'] = int('@return' in text_lower)
    features['has_see'] = int('@see' in text_lower)
    features['has_code'] = int('@code' in text_lower)
    features['has_deprecated'] = int('deprecat' in text_lower)
    features['has_example'] = int('example' in text_lower)
    features['has_note'] = int('note' in text_lower)
    features['has_todo'] = int('todo' in text_lower)
    features['has_author'] = int('author' in text_lower)
    features['has_version'] = int('version' in text_lower)
    features['has_since'] = int('since' in text_lower)
    features['has_throws'] = int('throw' in text_lower)
    features['has_exception'] = int('exception' in text_lower)
    
    features['has_question'] = int('?' in text)
    features['has_exclamation'] = int('!' in text)
    features['has_colon'] = int(':' in text)
    features['has_semicolon'] = int(';' in text)
    features['has_comma'] = int(',' in text)
    features['has_period'] = int('.' in text)
    
    words = text.split()
    features['word_count'] = len(words)
    features['char_count'] = len(text)
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    features['max_word_length'] = max([len(w) for w in words]) if words else 0
    features['sentence_length_bin'] = min(len(words) // 5, 10)
    
    features['has_uppercase'] = int(any(c.isupper() for c in text))
    features['has_number'] = int(any(c.isdigit() for c in text))
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
    
    features['has_class_ref'] = int(bool(re.search(r'[A-Z][a-z]+[A-Z]', text)))
    features['has_method_call'] = int('()' in text)
    features['has_curly_braces'] = int('{' in text or '}' in text)
    features['has_brackets'] = int('[' in text or ']' in text)
    features['has_parentheses'] = int('(' in text or ')' in text)
    
    features['sentence_starts_with_capital'] = int(text and text[0].isupper())
    features['ends_with_period'] = int(text.endswith('.'))
    features['starts_with_at'] = int(text.startswith('@'))
    
    features['has_this'] = int('this' in text_lower)
    features['has_method'] = int('method' in text_lower)
    features['has_class'] = int('class' in text_lower)
    features['has_function'] = int('function' in text_lower)
    features['has_variable'] = int('variable' in text_lower)
    features['has_parameter'] = int('parameter' in text_lower)
    
    features['has_should'] = int('should' in text_lower)
    features['has_will'] = int('will' in text_lower)
    features['has_may'] = int('may' in text_lower)
    features['has_can'] = int('can' in text_lower)
    features['has_must'] = int('must' in text_lower)
    
    features['has_if'] = int(' if ' in text_lower)
    features['has_when'] = int('when' in text_lower)
    features['has_where'] = int('where' in text_lower)
    features['has_why'] = int('why' in text_lower)
    features['has_how'] = int('how' in text_lower)
    features['has_what'] = int('what' in text_lower)
    
    return features


def extract_features_for_dataset(sentences):
    return pd.DataFrame([extract_advanced_nlp_features(s) for s in sentences]).values


def create_advanced_text_features(X_train_text, X_test_text):
    """
    Create multiple text representations and combine them.
    """
    word_tfidf = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9,
        lowercase=True,
        stop_words='english',
        sublinear_tf=True,
        analyzer='word'
    )
    
    char_tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(3, 5),
        analyzer='char',
        lowercase=True
    )
    
    word_count = CountVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        binary=True
    )
    
    X_train_word_tfidf = word_tfidf.fit_transform(X_train_text)
    X_test_word_tfidf = word_tfidf.transform(X_test_text)
    
    X_train_char_tfidf = char_tfidf.fit_transform(X_train_text)
    X_test_char_tfidf = char_tfidf.transform(X_test_text)
    
    X_train_word_count = word_count.fit_transform(X_train_text)
    X_test_word_count = word_count.transform(X_test_text)
    
    return (X_train_word_tfidf, X_test_word_tfidf,
            X_train_char_tfidf, X_test_char_tfidf,
            X_train_word_count, X_test_word_count)


def get_optimized_models():
    """
    Optimized models with aggressive hyperparameters for 60-70%+ performance.
    """
    models = {
        'Logistic Regression (Optimized)': LogisticRegression(
            C=5.0,
            max_iter=3000,
            solver='saga',
            penalty='elasticnet',
            l1_ratio=0.5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'Linear SVC (Optimized)': LinearSVC(
            C=1.5,
            max_iter=3000,
            random_state=42,
            dual=False,
            class_weight='balanced',
            loss='squared_hinge'
        ),
        'Random Forest (Optimized)': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample'
        ),
        'Gradient Boosting (Optimized)': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.15,
            max_depth=7,
            min_samples_split=4,
            subsample=0.9,
            random_state=42
        )
    }
    return models


def create_ensemble_model():
    """
    Create stacked ensemble for maximum performance.
    """
    base_estimators = [
        ('lr', LogisticRegression(C=5.0, solver='saga', max_iter=2000, class_weight='balanced')),
        ('svc', LinearSVC(C=1.5, max_iter=2000, class_weight='balanced', dual=False)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', n_jobs=-1))
    ]
    
    final_estimator = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=3,
        n_jobs=-1
    )
    
    return stacking


def train_with_advanced_kfold(df, category, clf_name, clf, n_folds=5):
    """
    Advanced k-fold training with feature engineering, selection, and SMOTE.
    """
    print(f"\n  🚀 Training {clf_name} on '{category}' with advanced techniques...")
    
    category_data = df[df['category'] == category].copy()
    X_text = category_data['comment_sentence'].values
    y = category_data['instance_type'].values
    
    print(f"     Class distribution: {dict(Counter(y))}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_text, y)):
        X_train_text = X_text[train_idx]
        X_test_text = X_text[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        (X_train_word_tfidf, X_test_word_tfidf,
         X_train_char_tfidf, X_test_char_tfidf,
         X_train_word_count, X_test_word_count) = create_advanced_text_features(
            X_train_text, X_test_text
        )
        
        X_train_nlp = extract_features_for_dataset(X_train_text)
        X_test_nlp = extract_features_for_dataset(X_test_text)
        
        X_train_combined = hstack([
            X_train_word_tfidf,
            X_train_char_tfidf,
            X_train_word_count,
            X_train_nlp
        ])
        X_test_combined = hstack([
            X_test_word_tfidf,
            X_test_char_tfidf,
            X_test_word_count,
            X_test_nlp
        ])
        
        if X_train_combined.shape[1] > 5000:
            selector = SelectKBest(chi2, k=min(5000, X_train_combined.shape[1]))
            X_train_combined = selector.fit_transform(X_train_combined, y_train)
            X_test_combined = selector.transform(X_test_combined)
        
        class_counts = Counter(y_train)
        if class_counts[1] < class_counts[0] * 0.3 and class_counts[1] >= 6:
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, class_counts[1]-1))
                X_train_combined, y_train = smote.fit_resample(X_train_combined, y_train)
            except:
                pass
        
        fold_clf = clf.__class__(**clf.get_params())
        fold_clf.fit(X_train_combined, y_train)
        y_pred = fold_clf.predict(X_test_combined)
        
        try:
            if hasattr(fold_clf, 'predict_proba'):
                y_proba = fold_clf.predict_proba(X_test_combined)[:, 1]
            elif hasattr(fold_clf, 'decision_function'):
                y_proba = fold_clf.decision_function(X_test_combined)
            else:
                y_proba = y_pred.astype(float)
        except:
            y_proba = y_pred.astype(float)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', pos_label=1, zero_division=0
        )
        
        fold_results.append({
            'fold': fold_idx + 1,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        })
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    
    avg_precision = np.mean([f['precision'] for f in fold_results])
    avg_recall = np.mean([f['recall'] for f in fold_results])
    avg_f1 = np.mean([f['f1'] for f in fold_results])
    std_f1 = np.std([f['f1'] for f in fold_results])
    
    try:
        roc_auc = roc_auc_score(all_y_true, all_y_proba)
    except:
        roc_auc = 0.0
    
    print(f"     ✅ F1: {avg_f1:.4f} ± {std_f1:.4f}, P: {avg_precision:.4f}, R: {avg_recall:.4f}, AUC: {roc_auc:.4f}")
    
    return {
        'category': category,
        'classifier': clf_name,
        'avg_precision': float(avg_precision),
        'avg_recall': float(avg_recall),
        'avg_f1': float(avg_f1),
        'std_f1': float(std_f1),
        'roc_auc': float(roc_auc)
    }


def train_ensemble_model(df, category, n_folds=5):
    """
    Train stacked ensemble for ultimate performance.
    """
    print(f"\n  🏆 Training ENSEMBLE STACK on '{category}'...")
    
    category_data = df[df['category'] == category].copy()
    X_text = category_data['comment_sentence'].values
    y = category_data['instance_type'].values
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_text, y)):
        X_train_text = X_text[train_idx]
        X_test_text = X_text[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        (X_train_word_tfidf, X_test_word_tfidf,
         X_train_char_tfidf, X_test_char_tfidf,
         X_train_word_count, X_test_word_count) = create_advanced_text_features(
            X_train_text, X_test_text
        )
        
        X_train_nlp = extract_features_for_dataset(X_train_text)
        X_test_nlp = extract_features_for_dataset(X_test_text)
        
        X_train = hstack([X_train_word_tfidf, X_train_char_tfidf, X_train_nlp])
        X_test = hstack([X_test_word_tfidf, X_test_char_tfidf, X_test_nlp])
        
        ensemble = create_ensemble_model()
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', pos_label=1, zero_division=0
        )
        
        fold_results.append({'precision': precision, 'recall': recall, 'f1': f1})
    
    avg_f1 = np.mean([f['f1'] for f in fold_results])
    print(f"     🎯 ENSEMBLE F1: {avg_f1:.4f}")
    
    return {
        'category': category,
        'classifier': 'Ensemble Stack',
        'avg_f1': float(avg_f1)
    }


def main():
    set_seed(42)
    
    print("="*100)
    print("ADVANCED ML SOLUTION - TARGET: 60-70%+ F1 SCORES")
    print("Techniques: Advanced Features + Ensemble + SMOTE + Feature Selection")
    print("="*100)
    
    languages = {
        'Java': 'data/code-comment-classification/java/input/java.csv',
        'Python': 'data/code-comment-classification/python/input/python.csv',
        'Pharo': 'data/code-comment-classification/pharo/input/pharo.csv'
    }
    
    models = get_optimized_models()
    all_results = []
    ensemble_results = []
    
    n_folds = 5
    
    for lang_name, csv_path in languages.items():
        print(f"\n{'='*100}")
        print(f"📊 Processing {lang_name}...")
        print(f"{'='*100}")
        
        df = pd.read_csv(csv_path)
        categories = df['category'].unique()
        
        for category in categories:
            print(f"\n📁 Category: {category}")
            
            for clf_name, clf in models.items():
                result = train_with_advanced_kfold(df, category, clf_name, clf, n_folds)
                result['language'] = lang_name
                all_results.append(result)
            
            ensemble_result = train_ensemble_model(df, category, n_folds)
            ensemble_result['language'] = lang_name
            ensemble_results.append(ensemble_result)
    
    output_dir = Path('runs/ml_advanced_solution')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(all_results)
    ensemble_df = pd.DataFrame(ensemble_results)
    
    print("\n" + "="*100)
    print("FINAL RESULTS SUMMARY")
    print("="*100)
    
    print("\n1️⃣  ADVANCED MODELS PERFORMANCE:")
    print("-"*100)
    classifier_summary = results_df.groupby('classifier').agg({
        'avg_precision': 'mean',
        'avg_recall': 'mean',
        'avg_f1': 'mean'
    }).round(4).sort_values('avg_f1', ascending=False)
    print(classifier_summary)
    
    print("\n2️⃣  ENSEMBLE PERFORMANCE:")
    print("-"*100)
    ensemble_avg = ensemble_df['avg_f1'].mean()
    print(f"Average Ensemble F1: {ensemble_avg:.4f}")
    
    print("\n3️⃣  BEST RESULTS PER CATEGORY:")
    print("-"*100)
    all_combined = pd.concat([results_df, ensemble_df])
    best_per_cat = all_combined.loc[all_combined.groupby('category')['avg_f1'].idxmax()]
    
    for _, row in best_per_cat.iterrows():
        status = "🔥" if row['avg_f1'] >= 0.70 else "✅" if row['avg_f1'] >= 0.60 else "📊"
        print(f"{status} {row['category']:<20} → {row['classifier']:<30} F1: {row['avg_f1']:.4f}")
    
    overall_best_f1 = all_combined['avg_f1'].max()
    overall_avg_f1 = all_combined.groupby('classifier')['avg_f1'].mean().max()
    
    print(f"\n{'='*100}")
    print(f"🎯 BEST SINGLE RESULT: F1 = {overall_best_f1:.4f}")
    print(f"📈 BEST AVERAGE PERFORMANCE: F1 = {overall_avg_f1:.4f}")
    
    if overall_avg_f1 >= 0.70:
        print(f"🔥🔥🔥 OUTSTANDING! Target 70%+ ACHIEVED!")
    elif overall_avg_f1 >= 0.60:
        print(f"🎉🎉 EXCELLENT! Target 60%+ ACHIEVED!")
    else:
        print(f"📊 Progress: {overall_avg_f1:.1%} (target: 60-70%)")
    
    results_df.to_csv(output_dir / 'advanced_results.csv', index=False)
    ensemble_df.to_csv(output_dir / 'ensemble_results.csv', index=False)
    
    save_json({
        'best_f1': float(overall_best_f1),
        'avg_best_f1': float(overall_avg_f1),
        'ensemble_avg': float(ensemble_avg),
        'configuration': {
            'n_folds': n_folds,
            'features': 'Word TF-IDF + Char TF-IDF + Count + 50+ NLP',
            'techniques': 'SMOTE + Feature Selection + Ensemble'
        }
    }, output_dir / 'summary.json')
    
    print(f"\n📁 Results saved to: {output_dir}/")
    print("="*100)


if __name__ == '__main__':
    main()


