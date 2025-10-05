import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict


JAVA_CATEGORIES = [
    'summary', 'ownership', 'expand', 'usage', 
    'pointer', 'deprecation', 'rational'
]

PYTHON_CATEGORIES = [
    'summary', 'usage', 'expand', 
    'parameters', 'developmentnotes'
]

PHARO_CATEGORIES = [
    'intent', 'responsibilities', 'collaborators', 
    'example', 'keyimplementation', 'keymessages', 
    'classreferences'
]


def load_language_data(csv_path, categories):
    df = pd.read_csv(csv_path)
    
    sentence_labels = defaultdict(lambda: {
        'class': None,
        'sentence': None,
        'partition': None,
        'labels': set()
    })
    
    for _, row in df.iterrows():
        sent_id = row['comment_sentence_id']
        
        if sentence_labels[sent_id]['sentence'] is None:
            sentence_labels[sent_id]['class'] = row['class']
            sentence_labels[sent_id]['sentence'] = row['comment_sentence']
            sentence_labels[sent_id]['partition'] = row['partition']
        
        if row['instance_type'] == 1:
            sentence_labels[sent_id]['labels'].add(row['category'].lower())
    
    records = []
    for sent_id, data in sentence_labels.items():
        if data['sentence'] is not None and len(data['labels']) > 0:
            records.append({
                'id': sent_id,
                'class_id': data['class'],
                'sentence': data['sentence'],
                'partition': data['partition'],
                'labels': ';'.join(sorted(data['labels']))
            })
    
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare NLBSE competition data for training'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/code-comment-classification',
        help='Path to competition data directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/raw',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--combine',
        action='store_true',
        help='Combine all languages into one file'
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("NLBSE Competition Data Preparation")
    print("=" * 60)
    print()
    
    all_dfs = []
    
    print("Processing Java data...")
    java_df = load_language_data(
        data_dir / 'java' / 'input' / 'java.csv',
        JAVA_CATEGORIES
    )
    java_df['lang'] = 'JAVA'
    print(f"  Loaded {len(java_df)} Java sentences")
    print(f"  Categories: {JAVA_CATEGORIES}")
    print(f"  Train: {(java_df['partition'] == 0).sum()}, Test: {(java_df['partition'] == 1).sum()}")
    java_df.to_csv(output_dir / 'java_sentences.csv', index=False)
    print(f"  Saved to {output_dir / 'java_sentences.csv'}")
    all_dfs.append(java_df)
    print()
    
    print("Processing Python data...")
    python_df = load_language_data(
        data_dir / 'python' / 'input' / 'python.csv',
        PYTHON_CATEGORIES
    )
    python_df['lang'] = 'PY'
    print(f"  Loaded {len(python_df)} Python sentences")
    print(f"  Categories: {PYTHON_CATEGORIES}")
    print(f"  Train: {(python_df['partition'] == 0).sum()}, Test: {(python_df['partition'] == 1).sum()}")
    python_df.to_csv(output_dir / 'python_sentences.csv', index=False)
    print(f"  Saved to {output_dir / 'python_sentences.csv'}")
    all_dfs.append(python_df)
    print()
    
    print("Processing Pharo data...")
    pharo_df = load_language_data(
        data_dir / 'pharo' / 'input' / 'pharo.csv',
        PHARO_CATEGORIES
    )
    pharo_df['lang'] = 'PHARO'
    print(f"  Loaded {len(pharo_df)} Pharo sentences")
    print(f"  Categories: {PHARO_CATEGORIES}")
    print(f"  Train: {(pharo_df['partition'] == 0).sum()}, Test: {(pharo_df['partition'] == 1).sum()}")
    pharo_df.to_csv(output_dir / 'pharo_sentences.csv', index=False)
    print(f"  Saved to {output_dir / 'pharo_sentences.csv'}")
    all_dfs.append(pharo_df)
    print()
    
    if args.combine:
        print("Combining all languages...")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df['id'] = range(len(combined_df))
        combined_df.to_csv(output_dir / 'sentences.csv', index=False)
        print(f"  Total sentences: {len(combined_df)}")
        print(f"  Train: {(combined_df['partition'] == 0).sum()}, Test: {(combined_df['partition'] == 1).sum()}")
        print(f"  Saved to {output_dir / 'sentences.csv'}")
        print()
        
        all_categories = set()
        for labels_str in combined_df['labels']:
            all_categories.update(labels_str.split(';'))
        print(f"  Total unique categories: {len(all_categories)}")
        print(f"  Categories: {sorted(all_categories)}")
    
    print()
    print("=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Review the generated CSV files in", output_dir)
    print("  2. Run: python -m src.split --input data/raw/sentences.csv --out data/processed/splits.json")
    print("  3. Train models with: python -m src.train --config configs/lora_modernbert.yaml")


if __name__ == '__main__':
    main()
