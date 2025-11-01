import pandas as pd
import numpy as np
from transformers import AutoTokenizer

df = pd.read_csv('data/raw/sentences.csv')

char_lengths = df['sentence'].str.len()
print("="*80)
print("CODE COMMENT LENGTH ANALYSIS")
print("="*80)

print(f"\n📊 Character-level Statistics:")
print(f"  Mean:   {char_lengths.mean():.1f} characters")
print(f"  Median: {char_lengths.median():.1f} characters")
print(f"  Max:    {char_lengths.max()} characters")
print(f"  95th percentile: {np.percentile(char_lengths, 95):.1f} characters")
print(f"  99th percentile: {np.percentile(char_lengths, 99):.1f} characters")

tokenizers = [
    ("CodeBERT", "microsoft/codebert-base"),
    ("GraphCodeBERT", "microsoft/graphcodebert-base"),
    ("DeBERTa-v3", "microsoft/deberta-v3-base"),
]

for name, model_name in tokenizers:
    print(f"\n📊 {name} Token Length Analysis:")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        token_lengths = []
        for text in df['sentence'][:1000]:
            tokens = tokenizer.encode(str(text), add_special_tokens=True)
            token_lengths.append(len(tokens))
        
        token_lengths = np.array(token_lengths)
        print(f"  Mean:   {token_lengths.mean():.1f} tokens")
        print(f"  Median: {np.median(token_lengths):.1f} tokens")
        print(f"  Max:    {token_lengths.max()} tokens (from 1000 samples)")
        print(f"  95th percentile: {np.percentile(token_lengths, 95):.1f} tokens")
        print(f"  99th percentile: {np.percentile(token_lengths, 99):.1f} tokens")
        print(f"  % samples > 128 tokens: {(token_lengths > 128).sum() / len(token_lengths) * 100:.2f}%")
        print(f"  % samples > 256 tokens: {(token_lengths > 256).sum() / len(token_lengths) * 100:.2f}%")
        print(f"  % samples > 512 tokens: {(token_lengths > 512).sum() / len(token_lengths) * 100:.2f}%")
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
if np.percentile(char_lengths, 99) < 300:
    print("✅ Max length 128-256 tokens is likely SUFFICIENT for 99% of samples")
    print("✅ Max length 512 tokens is DEFINITELY MORE THAN ENOUGH")
print("="*80)

