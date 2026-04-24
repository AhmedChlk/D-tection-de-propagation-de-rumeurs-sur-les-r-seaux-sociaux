"""
models.py — Architectures 4-Classes (Rumor Detection)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel

class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 100, num_classes: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1),
            nn.Conv1d(embed_dim, 128, kernel_size=4, padding=2),
            nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        ])
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 3, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1) 
        convs = [F.relu(conv(x)) for conv in self.convs]
        pools = [torch.max(c, dim=2)[0] for c in convs] 
        x = torch.cat(pools, dim=1)
        return self.fc(self.dropout(x))

class DistilBERTFineTuned(nn.Module):
    def __init__(self, dropout: float = 0.3, num_classes: int = 4):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        # Gel partiel : on dégèle seulement les 2 dernières couches pour adapter au vocabulaire Twitter
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.transformer.layer[-2:].parameters():
            param.requires_grad = True
            
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)