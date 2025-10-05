import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
from transformers import AutoTokenizer


LANG_TOKENS = {
    'JAVA': '[JAVA]',
    'PY': '[PY]',
    'PHARO': '[PHARO]'
}


class CodeCommentDataset(Dataset):
    def __init__(
        self,
        sentences: List[str],
        labels: torch.Tensor,
        langs: List[str],
        tokenizer: AutoTokenizer,
        max_len: int = 128
    ):
        self.sentences = sentences
        self.labels = labels
        self.langs = langs
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        lang = self.langs[idx]
        label = self.labels[idx]
        
        lang_token = LANG_TOKENS.get(lang.upper(), '')
        text = f"{lang_token} {sentence}".strip()
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label
        }


def load_raw_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = ['id', 'class_id', 'sentence', 'lang', 'labels']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    df = df.dropna(subset=['sentence', 'labels'])
    df['sentence'] = df['sentence'].astype(str).str.strip()
    df['labels'] = df['labels'].astype(str).str.strip()
    df = df[df['sentence'] != '']
    df = df[df['labels'] != '']
    
    return df


def prepare_tokenizer(model_name: str, add_lang_tokens: bool = True) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if add_lang_tokens:
        special_tokens = list(LANG_TOKENS.values())
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    return tokenizer
