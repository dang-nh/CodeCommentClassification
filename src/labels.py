import random
from typing import List, Set
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


class LabelManager:
    def __init__(self, label_names: List[str]):
        self.label_names = sorted(label_names)
        self.mlb = MultiLabelBinarizer(classes=self.label_names)
        self.mlb.fit([self.label_names])
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_names)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
    def encode(self, label_lists: List[List[str]]) -> np.ndarray:
        return self.mlb.transform(label_lists).astype(np.float32)
    
    def decode(self, binary_matrix: np.ndarray, threshold: float = 0.5) -> List[List[str]]:
        predictions = (binary_matrix >= threshold).astype(int)
        return self.mlb.inverse_transform(predictions)
    
    def get_num_labels(self) -> int:
        return len(self.label_names)
    
    def generate_label_orders(self, num_orders: int = 3, seed: int = 42) -> List[List[int]]:
        rng = random.Random(seed)
        orders = []
        for _ in range(num_orders):
            order = list(range(self.get_num_labels()))
            rng.shuffle(order)
            orders.append(order)
        return orders
    
    def parse_label_string(self, label_str: str) -> List[str]:
        if not label_str or label_str.strip() == '':
            return []
        separators = [';', ',']
        for sep in separators:
            if sep in label_str:
                labels = [l.strip() for l in label_str.split(sep) if l.strip()]
                return labels
        return [label_str.strip()]
