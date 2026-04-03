import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import re
import string
from collections import Counter
import os
import glob
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

def build_vocab(tokenized_texts, min_freq=5):
    """
    KDD Transformation : Construction du vocabulaire.
    - min_freq=5 pour filtrer les mots rares sur un gros dataset.
    """
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def texts_to_sequences(tokenized_texts, vocab, max_len=200):
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
    KDD Sélection : Chargement dynamique de tous les fichiers CSV.
    Détection automatique des colonnes de texte, de titre et de labels.
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {data_dir}")
        
    print(f"Chargement de {len(csv_files)} fichiers CSV...")
    
    dfs = []
    for path in csv_files:
        filename = os.path.basename(path).lower()
        df = pd.read_csv(path)
        
        # 1. Détection dynamique du texte et du titre
        text_cols = ['text', 'body_text', 'content']
        title_cols = ['title', 'headline']
        
        found_text_col = next((c for c in text_cols if c in df.columns), None)
        found_title_col = next((c for c in title_cols if c in df.columns), None)
        
        if not found_text_col and not found_title_col:
            print(f"Saut du fichier {filename} : aucune colonne de texte/titre trouvée.")
            continue
            
        # Fusion du contenu
        text_data = df[found_text_col].fillna('') if found_text_col else ""
        title_data = df[found_title_col].fillna('') if found_title_col else ""
        df['final_content'] = text_data.astype(str) + " " + title_data.astype(str)
        
        # 2. Détection dynamique du Label
        label_cols = ['label', 'class', 'type']
        found_label_col = next((c for c in label_cols if c in df.columns), None)
        
        if found_label_col:
            # On s'assure que c'est numérique (0/1)
            # Certains datasets utilisent 'fake'/'real' ou '0'/'1'
            labels = df[found_label_col].astype(str).str.lower()
            df['final_label'] = labels.apply(lambda x: 1 if 'fake' in x or '1' in x or 'unreliable' in x else 0)
        else:
            # Fallback : déduction via le nom du fichier
            label_val = 1 if 'fake' in filename else 0
            df['final_label'] = label_val
            
        dfs.append(df[['final_content', 'final_label']])
            
    if not dfs:
        raise ValueError("Aucune donnée valide n'a pu être extraite des fichiers CSV.")
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.columns = ['content', 'label']
    print(f"Total des échantillons chargés : {len(full_df)}")
    return full_df

def get_dataloaders(data_dir='data/raw', batch_size=32, max_len=200):
    """
    Pipeline KDD pour Séquences (CNN). Adapté aux gros datasets.
    """
    df = load_fakenewsnet_data(data_dir)
    
    print("KDD Prétraitement : Nettoyage et tokenisation...")
    tokenized_texts = [preprocess_text(t) for t in df['content']]
    
    print("KDD Transformation : Vocabulaire (min_freq=5)...")
    vocab = build_vocab(tokenized_texts, min_freq=5)
    vocab_size = len(vocab)
    
    X = texts_to_sequences(tokenized_texts, vocab, max_len=max_len)
    y = torch.tensor(df['label'].values)
    
    dataset = TensorDataset(X, y)
    
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size),
            DataLoader(test_loader := DataLoader(test_ds, batch_size=batch_size)), 
            vocab_size)

def get_tfidf_dataloaders(data_dir='data/raw', batch_size=32, max_features=1000):
    """
    Pipeline KDD TF-IDF (MLP). Adapté aux gros datasets.
    """
    df = load_fakenewsnet_data(data_dir)
    
    print(f"KDD Transformation : TF-IDF (max_features={max_features})...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_tfidf = vectorizer.fit_transform(df['content'])
    
    X = torch.tensor(X_tfidf.toarray(), dtype=torch.float32)
    y = torch.tensor(df['label'].values)
    
    dataset = TensorDataset(X, y)
    
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size),
            DataLoader(test_ds, batch_size=batch_size),
            max_features)

if __name__ == "__main__":
    try:
        train_l, val_l, test_l, v_size = get_dataloaders()
        print(f"\nSuccès : {v_size} mots uniques (freq >= 5).")
        print(f"Nombre total de batchs (train): {len(train_l)}")
    except Exception as e:
        print(f"Erreur : {e}")
