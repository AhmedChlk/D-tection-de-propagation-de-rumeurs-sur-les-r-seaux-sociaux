import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import re
import string
from collections import Counter
import os

def preprocess_text(text):
    """
    Nettoyage de base : minuscules, retrait de ponctuation, et tokenisation simple.
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    # Retrait de la ponctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    # Tokenisation par espace
    tokens = text.split()
    return tokens

def build_vocab(tokenized_texts, min_freq=2):
    """
    Construit un dictionnaire mot -> index.
    0 est réservé au padding, 1 aux mots inconnus (OOV).
    """
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    
    # Filtrage par fréquence
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def texts_to_sequences(tokenized_texts, vocab, max_len=100):
    """
    Transforme les tokens en séquences d'entiers avec padding/tronquage.
    """
    sequences = []
    for tokens in tokenized_texts:
        seq = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
        # Tronquage
        seq = seq[:max_len]
        # Padding
        seq += [vocab["<PAD>"]] * (max_len - len(seq))
        sequences.append(seq)
    return torch.tensor(sequences)

def load_fakenewsnet_data(data_dir="data/raw"):
    """
    Charge et concatène les fichiers BuzzFeed et PolitiFact.
    Label 0 pour 'real', 1 pour 'fake'.
    """
    files = {
        "BuzzFeed_fake_news_content.csv": 1,
        "BuzzFeed_real_news_content.csv": 0,
        "PolitiFact_fake_news_content.csv": 1,
        "PolitiFact_real_news_content.csv": 0
    }
    
    dfs = []
    for filename, label in files.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['label'] = label
            # On garde le titre et le texte pour la fusion ou le choix
            dfs.append(df[['title', 'text', 'label']])
        else:
            print(f"Attention : Fichier {path} non trouvé.")
            
    if not dfs:
        raise FileNotFoundError("Aucun fichier CSV trouvé dans data/raw/")
        
    full_df = pd.concat(dfs, ignore_index=True)
    # On privilégie 'text', si vide on prend 'title'
    full_df['content'] = full_df['text'].fillna('') + " " + full_df['title'].fillna('')
    return full_df

def get_dataloaders(batch_size=32, max_len=100, train_split=0.7, val_split=0.15):
    """
    Pipeline complet pour obtenir les DataLoaders de PyTorch.
    """
    # 1. Chargement
    df = load_fakenewsnet_data()
    
    # 2. Prétraitement
    print("Prétraitement des textes...")
    tokenized_texts = [preprocess_text(t) for t in df['content']]
    
    # 3. Vocabulaire et séquençage
    vocab = build_vocab(tokenized_texts)
    vocab_size = len(vocab)
    print(f"Taille du vocabulaire : {vocab_size}")
    
    X = texts_to_sequences(tokenized_texts, vocab, max_len=max_len)
    y = torch.tensor(df['label'].values)
    
    # 4. Création des datasets
    dataset = TensorDataset(X, y)
    
    total = len(dataset)
    train_size = int(train_split * total)
    val_size = int(val_split * total)
    test_size = total - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 5. DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, vocab_size

if __name__ == "__main__":
    # Test rapide
    train_l, val_l, test_l, v_size = get_dataloaders()
    print(f"Train batches: {len(train_l)}")
    print(f"Val batches: {len(val_l)}")
    print(f"Test batches: {len(test_l)}")
    print(f"Vocab size: {v_size}")
