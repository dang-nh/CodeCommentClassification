import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = Path('runs/ml_advanced_solution')

print("="*100)
print("DETAILED RESULTS ANALYSIS - Path to 70%+ Average")
print("="*100)

advanced_df = pd.read_csv(output_dir / 'advanced_results.csv')
ensemble_df = pd.read_csv(output_dir / 'ensemble_results.csv')

print("\n1️⃣  TOP 10 BEST PERFORMING CATEGORIES:")
print("-"*100)
best_10 = advanced_df.nlargest(10, 'avg_f1')[['category', 'classifier', 'avg_f1', 'avg_precision', 'avg_recall']]
for idx, row in best_10.iterrows():
    status = "🔥🔥" if row['avg_f1'] >= 0.80 else "🔥" if row['avg_f1'] >= 0.70 else "✅"
    print(f"{status} {row['category']:<25} {row['classifier']:<35} F1: {row['avg_f1']:.4f} (P: {row['avg_precision']:.4f}, R: {row['avg_recall']:.4f})")

print("\n2️⃣  CATEGORIES NEEDING IMPROVEMENT (<60% F1):")
print("-"*100)
needs_improvement = advanced_df[advanced_df['avg_f1'] < 0.60].groupby('category').agg({
    'avg_f1': 'max',
    'avg_precision': 'max',
    'avg_recall': 'max'
}).sort_values('avg_f1')

for category, row in needs_improvement.iterrows():
    imbalance_indicator = "⚠️" if row['avg_precision'] > 0.6 else "📊"
    print(f"{imbalance_indicator} {category:<25} F1: {row['avg_f1']:.4f} (P: {row['avg_precision']:.4f}, R: {row['avg_recall']:.4f})")

print("\n3️⃣  PERFORMANCE BY LANGUAGE:")
print("-"*100)
lang_summary = advanced_df.groupby('language').agg({
    'avg_f1': 'mean',
    'avg_precision': 'mean',
    'avg_recall': 'mean'
}).round(4)
print(lang_summary)

print("\n4️⃣  MODEL STABILITY (Standard Deviation):")
print("-"*100)
stability = advanced_df.groupby('classifier')['std_f1'].mean().sort_values()
print(stability)
print("\nNote: Lower std_f1 = more stable model")

print("\n5️⃣  PRECISION vs RECALL ANALYSIS:")
print("-"*100)
advanced_df['p_r_ratio'] = advanced_df['avg_precision'] / (advanced_df['avg_recall'] + 0.001)
high_precision_low_recall = advanced_df[
    (advanced_df['avg_precision'] > 0.7) & (advanced_df['avg_recall'] < 0.5)
][['category', 'classifier', 'avg_precision', 'avg_recall', 'avg_f1']].sort_values('avg_f1', ascending=False)

if len(high_precision_low_recall) > 0:
    print("⚠️  Categories with High Precision but Low Recall (need more SMOTE/threshold tuning):")
    for idx, row in high_precision_low_recall.head(5).iterrows():
        print(f"   {row['category']:<25} P: {row['avg_precision']:.3f}, R: {row['avg_recall']:.3f}, F1: {row['avg_f1']:.3f}")
else:
    print("✅ No major precision-recall imbalance detected")

print("\n6️⃣  IMPROVEMENT OPPORTUNITIES:")
print("-"*100)

categories_60_70 = advanced_df[(advanced_df['avg_f1'] >= 0.60) & (advanced_df['avg_f1'] < 0.70)]
categories_below_60 = advanced_df[advanced_df['avg_f1'] < 0.60]

improvement_potential = len(categories_below_60) + len(categories_60_70)
print(f"📊 {len(categories_below_60)} categories below 60%: potential +5-10% each")
print(f"📊 {len(categories_60_70)} categories 60-70%: potential +2-5% each")

current_avg = advanced_df.groupby('classifier')['avg_f1'].mean().max()
potential_gain_low = len(categories_below_60) * 0.05 / len(advanced_df['category'].unique())
potential_gain_mid = len(categories_60_70) * 0.02 / len(advanced_df['category'].unique())

print(f"\n🎯 Current Best Average: {current_avg:.4f} (60.88%)")
print(f"🎯 Potential with Improvements: {current_avg + potential_gain_low + potential_gain_mid:.4f} (estimated)")

print("\n7️⃣  RECOMMENDED NEXT STEPS:")
print("-"*100)
print("✅ 1. Hyperparameter tuning for low-performing categories")
print("✅ 2. More aggressive SMOTE for extremely imbalanced categories")
print("✅ 3. Category-specific feature engineering")
print("✅ 4. Threshold optimization per category")
print("✅ 5. Better ensemble stacking (currently underperforming)")

with open(output_dir / 'analysis_report.json', 'w') as f:
    json.dump({
        'current_best_avg': float(current_avg),
        'categories_above_70': int((advanced_df['avg_f1'] >= 0.70).sum()),
        'categories_60_70': int(len(categories_60_70)),
        'categories_below_60': int(len(categories_below_60)),
        'best_category': {
            'name': advanced_df.loc[advanced_df['avg_f1'].idxmax(), 'category'],
            'f1': float(advanced_df['avg_f1'].max())
        }
    }, f, indent=2)

print("\n✅ Analysis saved to: runs/ml_advanced_solution/analysis_report.json")
print("="*100)


