import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np


class ClassifierChains(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        num_labels: int,
        label_orders: List[List[int]],
        hidden_size: int = 768
    ):
        super(ClassifierChains, self).__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.label_orders = label_orders
        self.num_chains = len(label_orders)
        
        self.chain_adapters = nn.ModuleList([
            nn.Linear(hidden_size + i, 1)
            for i in range(num_labels)
        ])
        
    def forward_chain(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_order: List[int],
        ground_truth: Optional[torch.Tensor] = None,
        use_teacher_forcing: bool = True
    ):
        base_logits = self.base_model(input_ids, attention_mask)
        batch_size = base_logits.size(0)
        device = base_logits.device
        
        chain_probs = torch.zeros(batch_size, self.num_labels, device=device)
        
        outputs = self.base_model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'last_hidden_state'):
            pooled = outputs.last_hidden_state[:, 0, :]
        else:
            pooled = outputs[0][:, 0, :]
        
        for step, label_idx in enumerate(label_order):
            if step == 0:
                chain_input = pooled
            else:
                prev_probs = chain_probs[:, label_order[:step]]
                chain_input = torch.cat([pooled, prev_probs], dim=1)
            
            logit = self.chain_adapters[step](chain_input).squeeze(-1)
            prob = torch.sigmoid(logit)
            
            if use_teacher_forcing and ground_truth is not None:
                chain_probs[:, label_idx] = ground_truth[:, label_idx]
            else:
                chain_probs[:, label_idx] = prob
        
        return chain_probs
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        training: bool = True
    ):
        if training and labels is not None:
            chain_idx = np.random.randint(0, self.num_chains)
            probs = self.forward_chain(
                input_ids, attention_mask,
                self.label_orders[chain_idx],
                ground_truth=labels,
                use_teacher_forcing=True
            )
            return torch.logit(probs.clamp(1e-7, 1 - 1e-7))
        else:
            all_probs = []
            for label_order in self.label_orders:
                probs = self.forward_chain(
                    input_ids, attention_mask,
                    label_order,
                    ground_truth=None,
                    use_teacher_forcing=False
                )
                all_probs.append(probs)
            
            avg_probs = torch.stack(all_probs).mean(dim=0)
            return torch.logit(avg_probs.clamp(1e-7, 1 - 1e-7))
