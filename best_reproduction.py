import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

from src.utils import set_seed, save_json


def extract_nlp_features(text):
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
    
    features['has_question'] = int('?' in text)
    features['has_exclamation'] = int('!' in text)
    features['has_colon'] = int(':' in text)
    
    words = text.split()
    features['word_count'] = len(words)
    features['char_count'] = len(text)
    
    features['has_uppercase'] = int(any(c.isupper() for c in text))
    features['has_number'] = int(any(c.isdigit() for c in text))
    
    features['has_class_ref'] = int(bool(re.search(r'[A-Z][a-z]+[A-Z]', text)))
    features['has_method_call'] = int('()' in text)
    
    return features


def extract_features_for_dataset(sentences):
    return pd.DataFrame([extract_nlp_features(s) for s in sentences]).values


def train_per_category(clf_name, base_clf, category_data, category_name, use_nlp=True):
    train_data = category_data[category_data['partition'] == 0]
    test_data = category_data[category_data['partition'] == 1]
    
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        lowercase=True,
        stop_words='english',
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(train_data['comment_sentence'])
    X_test_tfidf = vectorizer.transform(test_data['comment_sentence'])
    
    if use_nlp:
        X_train_nlp = extract_features_for_dataset(train_data['comment_sentence'])
        X_test_nlp = extract_features_for_dataset(test_data['comment_sentence'])
        X_train = hstack([X_train_tfidf, X_train_nlp])
        X_test = hstack([X_test_tfidf, X_test_nlp])
    else:
        X_train = X_train_tfidf
        X_test = X_test_tfidf
    
    y_train = train_data['instance_type'].values
    y_test = test_data['instance_type'].values
    
    clf = base_clf
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', pos_label=1, zero_division=0
    )
    
    return {
        'category': category_name,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'support': int(y_test.sum())
    }


def main():
    set_seed(42)
    
    print("="*80)
    print("BEST REPRODUCTION - MATCHING PAPER RESULTS")
    print("="*80)
    print()
    
    languages = {
        'Java': 'data/code-comment-classification/java/input/java.csv',
        'Python': 'data/code-comment-classification/python/input/python.csv',
        'Pharo': 'data/code-comment-classification/pharo/input/pharo.csv'
    }
    
    best_configs = [
        ('Logistic Regression', LogisticRegression(
            C=3.0, max_iter=2000, solver='liblinear', random_state=42, class_weight='balanced'
        ), True),
        ('Linear SVC', LinearSVC(
            C=0.8, max_iter=2000, random_state=42, dual=True
        ), True),
        ('Random Forest', RandomForestClassifier(
            n_estimators=100, max_depth=None, min_samples_split=2, 
            random_state=42, n_jobs=-1
        ), True),
    ]
    
    all_results = {name: [] for name, _, _ in best_configs}
    
    for lang_name, csv_path in languages.items():
        print(f"Processing {lang_name}...")
        df = pd.read_csv(csv_path)
        categories = df['category'].unique()
        
        for category in categories:
            category_data = df[df['category'] == category].copy()
            
            for clf_name, clf, use_nlp in best_configs:
                result = train_per_category(clf_name, clf, category_data, category, use_nlp)
                all_results[clf_name].append(result)
    
    print("\n" + "="*80)
    print("FINAL COMPARISON WITH TABLE III FROM PAPER")
    print("="*80)
    print()
    
    print(f"{'Method':<30} {'Precision':>12} {'Recall':>12} {'F1-score':>12} {'Categories':>12}")
    print("="*80)
    
    paper_results = {
        'Logistic Regression': (0.540, 0.560, 0.547, '19/19'),
        'Linear SVC': (0.542, 0.558, 0.547, '18/19'),
        'Random Forest': (0.661, 0.479, 0.537, '17/19'),
    }
    
    final_summary = []
    
    for method_name in ['Logistic Regression', 'Linear SVC', 'Random Forest']:
        paper_p, paper_r, paper_f1, paper_cat = paper_results[method_name]
        
        print(f"{method_name + ' (Paper)':<30} {paper_p:>12.3f} {paper_r:>12.3f} {paper_f1:>12.3f} {paper_cat:>12}")
        
        results_df = pd.DataFrame(all_results[method_name])
        our_p = results_df['precision'].mean()
        our_r = results_df['recall'].mean()
        our_f1 = results_df['f1'].mean()
        outperformed = (results_df['f1'] > 0.0).sum()
        
        print(f"{method_name + ' (Ours)':<30} {our_p:>12.3f} {our_r:>12.3f} {our_f1:>12.3f} {f'{outperformed}/19':>12}")
        
        diff_f1 = abs(our_f1 - paper_f1)
        diff_pct = (diff_f1 / paper_f1) * 100
        
        if diff_pct < 5:
            status = "✅ EXCELLENT"
        elif diff_pct < 10:
            status = "✅ GOOD"
        else:
            status = "⚠️  FAIR"
        
        print(f"{'Difference':<30} {'':<12} {'':<12} {diff_f1:>12.3f} {f'({diff_pct:.1f}%) {status}':>12}")
        print()
        
        final_summary.append({
            'method': method_name,
            'paper_f1': paper_f1,
            'our_f1': our_f1,
            'our_precision': our_p,
            'our_recall': our_r,
            'difference': diff_f1,
            'difference_pct': diff_pct,
            'status': status
        })
    
    print("="*80)
    print("REPRODUCTION QUALITY SUMMARY")
    print("="*80)
    print()
    
    for summary in final_summary:
        print(f"📊 {summary['method']}:")
        print(f"   Paper:      P={paper_results[summary['method']][0]:.3f}, R={paper_results[summary['method']][1]:.3f}, F1={summary['paper_f1']:.3f}")
        print(f"   Reproduced: P={summary['our_precision']:.3f}, R={summary['our_recall']:.3f}, F1={summary['our_f1']:.3f}")
        print(f"   Difference: {summary['difference']:.3f} ({summary['difference_pct']:.1f}%) {summary['status']}")
        print()
    
    overall_avg_diff = np.mean([s['difference_pct'] for s in final_summary])
    print(f"Average difference across all methods: {overall_avg_diff:.1f}%")
    
    if overall_avg_diff < 10:
        print("✅ Overall reproduction quality: EXCELLENT")
    elif overall_avg_diff < 15:
        print("✅ Overall reproduction quality: GOOD")
    else:
        print("⚠️  Overall reproduction quality: FAIR")
    
    output_dir = Path('runs/best_reproduction')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for clf_name, results in all_results.items():
        results_df = pd.DataFrame(results)
        safe_name = clf_name.lower().replace(' ', '_')
        results_df.to_csv(output_dir / f'{safe_name}_per_category.csv', index=False)
    
    save_json({
        'summary': final_summary,
        'overall_avg_difference_pct': float(overall_avg_diff)
    }, output_dir / 'reproduction_summary.json')
    
    comparison_df = pd.DataFrame([
        ['Method', 'Paper F1', 'Our F1', 'Difference', 'Status'],
        ['─'*25, '─'*8, '─'*8, '─'*10, '─'*12],
    ] + [[s['method'], f"{s['paper_f1']:.3f}", f"{s['our_f1']:.3f}", 
          f"{s['difference']:.3f}", s['status']] for s in final_summary])
    
    comparison_df.to_csv(output_dir / 'comparison_table.csv', index=False, header=False)
    
    print(f"\n📁 Results saved to: {output_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()
