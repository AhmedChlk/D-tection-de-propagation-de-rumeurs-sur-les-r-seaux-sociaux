"""
test_fakenewsnet.py — Script indépendant pour extraire les vraies métriques de FakeNewsNet (Phase 1 du rapport)
"""
import os
import re
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from transformers import DistilBertTokenizer
from sklearn.metrics import f1_score, classification_report
import random

# Importation de tes modèles et de ton entraîneur
from src.models import TextCNN, DistilBERTFineTuned
from src.trainer import train_model

DATA_DIR = "data/kaggle/FakeNewsNet"
TARGET_NAMES = ['Real', 'Fake']

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_fakenewsnet(data_dir):
    csv_files = sorted(glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True))
    dfs = []
    print("  [INFO] Chargement de FakeNewsNet...")
    for path in csv_files:
        fname = os.path.basename(path).lower()
        try: df = pd.read_csv(path)
        except: continue
        
        text_col = next((c for c in ['text', 'content'] if c in df.columns), None)
        title_col = 'title' if 'title' in df.columns else None
        if not text_col: continue
            
        label = 1 if 'fake' in fname else 0
        raw_combined = df[title_col].fillna('') + " " + df[text_col].fillna('') if title_col else df[text_col].fillna('')
        
        tmp = pd.DataFrame({'text': raw_combined.apply(clean_text), 'label': label})
        dfs.append(tmp)
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"  [INFO] FakeNewsNet prêt : {len(df)} articles (Fake: {df['label'].sum()}, Real: {len(df)-df['label'].sum()})")
    return df

# --- Utilitaires CNN ---
def tokenize(text): return str(text).split()
class SimpleVocab:
    def __init__(self, texts, specials=["<pad>", "<unk>"]):
        self.stoi = {tok: i for i, tok in enumerate(specials)}
        counter = Counter()
        for text in texts: counter.update(tokenize(text))
        for token, count in counter.most_common():
            if count >= 2 and token not in self.stoi: self.stoi[token] = len(self.stoi)
        self.unk_idx, self.pad_idx = self.stoi.get("<unk>", 1), self.stoi.get("<pad>", 0)
    def __getitem__(self, token): return self.stoi.get(token, self.unk_idx)
    def __len__(self): return len(self.stoi)

class TextDataset(Dataset):
    def __init__(self, texts, labels): self.texts, self.labels = texts, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.texts[idx], self.labels[idx]

def get_cnn_dataloaders(df, batch_size=16, max_len=300):
    n = len(df)
    indices = np.random.RandomState(42).permutation(n)
    t1, t2 = int(0.70 * n), int(0.85 * n)
    train_idx, val_idx, test_idx = indices[:t1], indices[t1:t2], indices[t2:]
    
    train_texts, train_labels = df['text'].values[train_idx], df['label'].values[train_idx]
    vocab = SimpleVocab(train_texts)
    
    def collate_fn(batch):
        texts, labels = zip(*batch)
        encoded = [torch.tensor([vocab[t] for t in tokenize(text)][:max_len], dtype=torch.long) for text in texts]
        return pad_sequence(encoded, batch_first=True, padding_value=vocab.pad_idx), torch.tensor(labels, dtype=torch.long)

    return (
        DataLoader(TextDataset(train_texts, train_labels), batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        DataLoader(TextDataset(df['text'].values[val_idx], df['label'].values[val_idx]), batch_size=batch_size, collate_fn=collate_fn),
        DataLoader(TextDataset(df['text'].values[test_idx], df['label'].values[test_idx]), batch_size=batch_size, collate_fn=collate_fn),
        len(vocab)
    )

# --- Utilitaires BERT ---
class BertDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings, self.labels = encodings, torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

def get_bert_dataloaders(df, batch_size=8, max_len=256):
    n = len(df)
    indices = np.random.RandomState(42).permutation(n)
    t1, t2 = int(0.70 * n), int(0.85 * n)
    train_idx, val_idx, test_idx = indices[:t1], indices[t1:t2], indices[t2:]
    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    def create_ds(idx_subset):
        encodings = tokenizer(df['text'].values[idx_subset].tolist(), truncation=True, padding=True, max_length=max_len)
        return BertDataset(encodings, df['label'].values[idx_subset])

    return DataLoader(create_ds(train_idx), batch_size=batch_size, shuffle=True), DataLoader(create_ds(val_idx), batch_size=batch_size), DataLoader(create_ds(test_idx), batch_size=batch_size)

def evaluate(model, test_loader, device, save_path, is_bert=False):
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            if is_bert:
                inputs = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)}
                labels = batch['labels'].to(device)
                out = model(**inputs)
            else:
                X, labels = batch[0].to(device), batch[1].to(device)
                out = model(X)
            all_preds.extend(out.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print("\n" + classification_report(all_labels, all_preds, target_names=TARGET_NAMES))
    return acc, f1

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = load_fakenewsnet(DATA_DIR)
    
    print("\n" + "═" * 60 + "\n  TEST FAKENEWSNET : TextCNN (2 Classes)\n" + "═" * 60)
    train_loader, val_loader, test_loader, vocab_size = get_cnn_dataloaders(df)
    model_cnn = TextCNN(vocab_size=vocab_size, num_classes=2).to(device)
    optimizer = optim.Adam(model_cnn.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_model(model_cnn, train_loader, val_loader, optimizer, criterion, device, epochs=15, patience=3, is_bert=False, save_path="fakenewsnet_cnn.pt")
    acc_cnn, f1_cnn = evaluate(model_cnn, test_loader, device, "fakenewsnet_cnn.pt", is_bert=False)
    
    print("\n" + "═" * 60 + "\n  TEST FAKENEWSNET : DistilBERT (2 Classes)\n" + "═" * 60)
    train_loader, val_loader, test_loader = get_bert_dataloaders(df)
    model_bert = DistilBERTFineTuned(num_classes=2).to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_bert.parameters()), lr=3e-5)
    criterion = nn.CrossEntropyLoss()
    train_model(model_bert, train_loader, val_loader, optimizer, criterion, device, epochs=10, patience=3, is_bert=True, save_path="fakenewsnet_bert.pt")
    acc_bert, f1_bert = evaluate(model_bert, test_loader, device, "fakenewsnet_bert.pt", is_bert=True)
    
    print("\n\n" + "═" * 60)
    print("  MÉTRIQUES À COPIER DANS LE RAPPORT LATEX (Section 2.1)")
    print("═" * 60)
    print(f"  FakeNewsNet CNN  -> Acc: {acc_cnn*100:.2f}% | F1: {f1_cnn:.4f}")
    print(f"  FakeNewsNet BERT -> Acc: {acc_bert*100:.2f}% | F1: {f1_bert:.4f}")

if __name__ == "__main__":
    main()