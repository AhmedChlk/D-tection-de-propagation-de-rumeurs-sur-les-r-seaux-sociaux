"""
data_loader.py — Twitter15 & Twitter16 Rumor Detection Dataset
"""
import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text) # Retire les @mentions (très fréquent sur Twitter)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_twitter_data(data_dir: str) -> pd.DataFrame:
    """Charge et fusionne Twitter15 et Twitter16"""
    dfs = []
    
    print("  [INFO] Chargement de Twitter15 et Twitter16...")
    for dataset_name in ['twitter15', 'twitter16']:
        d_path = os.path.join(data_dir, dataset_name)
        label_file = os.path.join(d_path, 'label.txt')
        text_file = os.path.join(d_path, 'source_tweets.txt')
        
        if not os.path.exists(label_file) or not os.path.exists(text_file):
            print(f"  [WARN] Fichiers introuvables dans {d_path}")
            continue
            
        # 1. Lire les labels (Format: label:tweet_id)
        labels_dict = {}
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) >= 2:
                    labels_dict[parts[1].strip()] = parts[0].strip()
                    
        # 2. Lire les tweets (Format: tweet_id \t text)
        texts_dict = {}
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    texts_dict[parts[0].strip()] = " ".join(parts[1:]).strip()
                    
        # 3. Combiner
        for tid, text in texts_dict.items():
            if tid in labels_dict:
                dfs.append({'tweet_id': tid, 'text': text, 'label_str': labels_dict[tid]})
                
    df = pd.DataFrame(dfs)
    
    if len(df) == 0:
        raise ValueError(f"Aucune donnée chargée. Vérifiez le chemin {data_dir}.")

    # Mapping académique standard en 4 classes
    label_map = {'non-rumor': 0, 'false': 1, 'true': 2, 'unverified': 3}
    df['label_str'] = df['label_str'].str.lower()
    df = df[df['label_str'].isin(label_map.keys())].copy()
    df['label'] = df['label_str'].map(label_map)
    
    print("  [INFO] Nettoyage des tweets en cours...")
    df['text'] = df['text'].apply(clean_text)
    
    print(f"  [INFO] Dataset prêt : {len(df)} tweets au total.")
    print(f"  [INFO] Répartition : {df['label_str'].value_counts().to_dict()}")
    return df

# ═══════════════════════════════════════════════════════════════
# PREPARATION POUR CNN
# ═══════════════════════════════════════════════════════════════
def tokenize(text): return str(text).split()

class SimpleVocab:
    def __init__(self, texts, specials=["<pad>", "<unk>"]):
        self.stoi = {tok: i for i, tok in enumerate(specials)}
        counter = Counter()
        for text in texts: 
            counter.update(tokenize(text))
        for token, count in counter.most_common():
            if count >= 2: 
                if token not in self.stoi: 
                    self.stoi[token] = len(self.stoi)
        self.unk_idx, self.pad_idx = self.stoi.get("<unk>", 1), self.stoi.get("<pad>", 0)
    
    def __getitem__(self, token): return self.stoi.get(token, self.unk_idx)
    def __len__(self): return len(self.stoi)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = texts, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.texts[idx], self.labels[idx]

def get_cnn_dataloaders(data_dir: str, batch_size: int = 32, max_len: int = 100):
    df = load_twitter_data(data_dir)
    n = len(df)
    indices = np.random.RandomState(42).permutation(n)
    t1, t2 = int(0.70 * n), int(0.85 * n)
    train_idx, val_idx, test_idx = indices[:t1], indices[t1:t2], indices[t2:]
    
    train_texts, train_labels = df['text'].values[train_idx], df['label'].values[train_idx]
    vocab = SimpleVocab(train_texts)
    
    def collate_fn(batch):
        texts, labels = zip(*batch)
        encoded = [torch.tensor([vocab[t] for t in tokenize(text)][:max_len], dtype=torch.long) for text in texts]
        padded = pad_sequence(encoded, batch_first=True, padding_value=vocab.pad_idx)
        return padded, torch.tensor(labels, dtype=torch.long)

    return (
        DataLoader(TextDataset(train_texts, train_labels), batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        DataLoader(TextDataset(df['text'].values[val_idx], df['label'].values[val_idx]), batch_size=batch_size, collate_fn=collate_fn),
        DataLoader(TextDataset(df['text'].values[test_idx], df['label'].values[test_idx]), batch_size=batch_size, collate_fn=collate_fn),
        len(vocab)
    )

# ═══════════════════════════════════════════════════════════════
# PREPARATION POUR BERT
# ═══════════════════════════════════════════════════════════════
class BertDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

def get_bert_dataloaders(data_dir: str, batch_size: int = 16, max_len: int = 100):
    df = load_twitter_data(data_dir)
    n = len(df)
    indices = np.random.RandomState(42).permutation(n)
    t1, t2 = int(0.70 * n), int(0.85 * n)
    train_idx, val_idx, test_idx = indices[:t1], indices[t1:t2], indices[t2:]
    
    print("  [INFO] Tokenisation avec DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    def create_ds(idx_subset):
        encodings = tokenizer(df['text'].values[idx_subset].tolist(), truncation=True, padding=True, max_length=max_len)
        return BertDataset(encodings, df['label'].values[idx_subset])

    return (
        DataLoader(create_ds(train_idx), batch_size=batch_size, shuffle=True),
        DataLoader(create_ds(val_idx), batch_size=batch_size),
        DataLoader(create_ds(test_idx), batch_size=batch_size)
    )