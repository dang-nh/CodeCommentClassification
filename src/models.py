import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional, Dict, Any


class MultiLabelClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        use_peft: bool = False,
        peft_config: Optional[Dict[str, Any]] = None,
        gradient_checkpointing: bool = True
    ):
        super(MultiLabelClassifier, self).__init__()
        
        self.num_labels = num_labels
        self.use_peft = use_peft
        
        try:
            config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name, config=config)
        except Exception as e:
            print(f"Failed to load {model_name}, falling back to microsoft/deberta-v3-base")
            model_name = "microsoft/deberta-v3-base"
            config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name, config=config)
        
        if hasattr(self.encoder, 'embeddings') and hasattr(self.encoder.embeddings, 'word_embeddings'):
            self.encoder.resize_token_embeddings(self.encoder.embeddings.word_embeddings.num_embeddings + 3)
        
        if gradient_checkpointing and hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()
        
        if use_peft and peft_config:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=peft_config.get('r', 8),
                lora_alpha=peft_config.get('alpha', 16),
                lora_dropout=peft_config.get('dropout', 0.05),
                target_modules=["query", "key", "value", "dense"]
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.encoder.print_trainable_parameters()
        
        hidden_size = config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        if hasattr(outputs, 'last_hidden_state'):
            pooled = outputs.last_hidden_state[:, 0, :]
        else:
            pooled = outputs[0][:, 0, :]
        
        logits = self.classifier(pooled)
        return logits
    
    def save_pretrained(self, path: str):
        if self.use_peft:
            self.encoder.save_pretrained(path)
        else:
            self.encoder.save_pretrained(path)
        torch.save(self.classifier.state_dict(), f"{path}/classifier.pt")
    
    def load_pretrained(self, path: str):
        if self.use_peft:
            self.encoder = self.encoder.from_pretrained(path)
        else:
            self.encoder = AutoModel.from_pretrained(path)
        self.classifier.load_state_dict(torch.load(f"{path}/classifier.pt"))
