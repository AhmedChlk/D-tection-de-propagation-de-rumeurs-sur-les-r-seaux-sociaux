"""
main.py — Projet M1 : Rumor Detection on Social Networks (Twitter15/16)
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
from sklearn.metrics import f1_score, classification_report

from src.data_loader import get_cnn_dataloaders, get_bert_dataloaders
from src.models import TextCNN, DistilBERTFineTuned
from src.trainer import train_model

DATA_DIR = "data/kaggle/Rumor-Detection-Dataset" 
TARGET_NAMES = ['Non-Rumor', 'False', 'True', 'Unverified']

def set_seed(seed=42):
    """Fixe la graine aléatoire pour garantir la reproductibilité des résultats (Critère académique)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def run_cnn(device):
    print("\n" + "═" * 60 + "\n  MODÈLE : TextCNN (Twitter15/16)\n" + "═" * 60)
    train_loader, val_loader, test_loader, vocab_size = get_cnn_dataloaders(DATA_DIR, batch_size=32)
    
    model = TextCNN(vocab_size=vocab_size, num_classes=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    save_path = "best_cnn.pt"
    train_model(model, train_loader, val_loader, optimizer, criterion, device, 
                epochs=20, patience=5, is_bert=False, save_path=save_path)
                
    acc, f1 = evaluate(model, test_loader, device, save_path, is_bert=False)
    print(f"\n{'═'*60}\n  RÉSULTATS CNN | Acc: {acc*100:.2f}% | F1: {f1:.4f}\n{'═'*60}")
    return acc

def run_bert(device):
    print("\n" + "═" * 60 + "\n  MODÈLE : DistilBERT Fine-Tuned (Twitter15/16)\n" + "═" * 60)
    train_loader, val_loader, test_loader = get_bert_dataloaders(DATA_DIR, batch_size=16)
    
    model = DistilBERTFineTuned(num_classes=4).to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5)
    criterion = nn.CrossEntropyLoss()
    
    save_path = "best_bert.pt"
    train_model(model, train_loader, val_loader, optimizer, criterion, device, 
                epochs=10, patience=3, is_bert=True, save_path=save_path)
                
    acc, f1 = evaluate(model, test_loader, device, save_path, is_bert=True)
    print(f"\n{'═'*60}\n  RÉSULTATS BERT | Acc: {acc*100:.2f}% | F1: {f1:.4f}\n{'═'*60}")
    return acc

def main():
    # 1. Reproductibilité
    set_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all', choices=['cnn', 'bert', 'all'])
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(DATA_DIR):
        print(f"\n[ERREUR] Dossier {DATA_DIR} introuvable.")
        return

    if args.model == 'all':
        run_cnn(device)
        run_bert(device)
    elif args.model == 'cnn':
        run_cnn(device)
    elif args.model == 'bert':
        run_bert(device)

if __name__ == '__main__':
    main()