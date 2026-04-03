import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import re
import string
from collections import Counter
import os
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    """
    KDD Prétraitement : Nettoyage du texte.
    - Minuscules
    - Suppression des URLs
    - Suppression de la ponctuation
    - Tokenisation simple
    """
    if not isinstance(text, str):
        return []
    
    # Passage en minuscules
    text = text.lower()
    
    # Suppression des URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Suppression de la ponctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    
    # Tokenisation par espace
    tokens = text.split()
    return tokens

def build_vocab(tokenized_texts, min_freq=2):
    """
    KDD Transformation : Construction du vocabulaire.
    """
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def texts_to_sequences(tokenized_texts, vocab, max_len=100):
    """
    KDD Transformation : Conversion en séquences d'entiers avec padding.
    """
    sequences = []
    for tokens in tokenized_texts:
        seq = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
        seq = seq[:max_len] # Tronquage
        seq += [vocab["<PAD>"]] * (max_len - len(seq)) # Padding
        sequences.append(seq)
    return torch.tensor(sequences)

def load_fakenewsnet_data(data_dir="data/raw"):
    """
    KDD Sélection : Chargement et fusion des fichiers FakeNewsNet.
    Label 1 pour 'fake', 0 pour 'real'.
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
            dfs.append(df[['title', 'text', 'label']])
        else:
            print(f"Attention : Fichier {path} non trouvé.")
            
    if not dfs:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {data_dir}")
        
    full_df = pd.concat(dfs, ignore_index=True)
    # On fusionne le titre et le texte pour avoir plus de contexte
    full_df['content'] = full_df['text'].fillna('') + " " + full_df['title'].fillna('')
    return full_df

def get_dataloaders(data_dir='data/raw', batch_size=32, max_len=100):
    """
    Pipeline KDD complet pour obtenir les DataLoaders (Séquences pour CNN).
    """
    # 1. Sélection
    df = load_fakenewsnet_data(data_dir)
    
    # 2. Prétraitement
    print("KDD Prétraitement : Nettoyage et tokenisation...")
    tokenized_texts = [preprocess_text(t) for t in df['content']]
    
    # 3. Transformation
    print("KDD Transformation : Vectorisation et Padding...")
    vocab = build_vocab(tokenized_texts)
    vocab_size = len(vocab)
    
    X = texts_to_sequences(tokenized_texts, vocab, max_len=max_len)
    y = torch.tensor(df['label'].values)
    
    # 4. Création des datasets
    dataset = TensorDataset(X, y)
    
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
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

def get_tfidf_dataloaders(data_dir='data/raw', batch_size=32, max_features=5000):
    """
    Pipeline KDD avec vectorisation TF-IDF (pour MLP).
    """
    # 1. Sélection
    df = load_fakenewsnet_data(data_dir)
    
    # 2 & 3. Prétraitement & Transformation (TF-IDF)
    print(f"KDD Transformation : TF-IDF Vectorization (max_features={max_features})...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_tfidf = vectorizer.fit_transform(df['content'])
    
    X = torch.tensor(X_tfidf.toarray(), dtype=torch.float32)
    y = torch.tensor(df['label'].values)
    
    # 4. Création des datasets
    dataset = TensorDataset(X, y)
    
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 5. DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, max_features

if __name__ == "__main__":
    # EXÉCUTION : Résumé des données pour validation
    try:
        print("Test Séquences (CNN):")
        train_l, val_l, test_l, v_size = get_dataloaders()
        print(f"Vocab size: {v_size}, Train batches: {len(train_l)}")
        
        print("\nTest TF-IDF (MLP):")
        train_l_tfidf, val_l_tfidf, test_l_tfidf, feat_size = get_tfidf_dataloaders()
        print(f"Features size: {feat_size}, Train batches: {len(train_l_tfidf)}")
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
