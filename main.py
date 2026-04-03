import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Ajout du dossier courant au path pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.prepare_data import get_dataloaders, load_fakenewsnet_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.models.baseline_cnn import BaselineCNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    parser = argparse.ArgumentParser(description="Pipeline Deep Learning : Détection de Rumeurs (CNN sur Dataset Massif)")
    parser.add_argument("--epochs", type=int, default=20, help="Nombre d'époques d'entraînement.")
    parser.add_argument("--batch_size", type=int, default=64, help="Taille des lots (augmentée pour gros dataset).")
    parser.add_argument("--lr", type=float, default=0.001, help="Taux d'apprentissage (Adam).")
    parser.add_argument("--patience", type=int, default=5, help="Patience pour l'Early Stopping.")
    parser.add_argument("--dry-run", action="store_true", help="Mode test rapide (1 époque, 1 batch).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Appareil utilisé : {device}")
    model_path = "best_model.pt"

    # --- ÉTAPE 1 & 2 : SÉLECTION & PRÉTRAITEMENT ---
    print("\n--- ÉTAPE KDD 1 & 2 : Préparation des données (Séquences) ---")
    max_len = 200 
    train_loader, val_loader, test_loader, vocab_size = get_dataloaders(
        batch_size=args.batch_size, 
        max_len=max_len
    )

    # --- ÉTAPE 3 : TRANSFORMATION (Initialisation Modèle CNN) ---
    print(f"\n--- ÉTAPE KDD 3 : Initialisation du modèle CNN (Vocab size: {vocab_size}) ---")
    model = BaselineCNN(
        vocab_size=vocab_size, 
        embedding_dim=100, 
        n_filters=50,
        dropout=0.5
    ).to(device)

    # Régularisation modérée (1e-4) adaptée à un volume de données important
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # --- ÉTAPE 4 : DATA MINING (Entraînement) ---
    print("\n--- ÉTAPE KDD 4 : Data Mining (Entraînement) ---")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        clip_value=2.0,
        dry_run=args.dry_run
    )

    # --- ÉTAPE 5 : INTERPRÉTATION & ÉVALUATION ---
    print("\n--- ÉTAPE KDD 5 : Interprétation & Évaluation finale ---")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] Meilleur modèle chargé depuis {model_path}")
    
    y_pred, y_true, errors = evaluate_model(model, test_loader, device)
    
    # Calcul des métriques
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*40)
    print("BILAN FINAL DU PROJET (DATASET MASSIF)")
    print("="*40)
    print(f"Accuracy   : {acc:.4f}")
    print(f"Précision  : {prec:.4f}")
    print(f"Rappel     : {rec:.4f}")
    print(f"F1-Score   : {f1:.4f}")
    print("\nMatrice de Confusion :")
    print(cm)
    print("="*40)

    # Analyse qualitative
    print("\n--- Analyse qualitative des erreurs ---")
    df = load_fakenewsnet_data()
    test_indices = test_loader.dataset.indices
    
    fp_found, fn_found = False, False
    for err in errors:
        content = df.iloc[test_indices[err['test_idx']]]['content']
        if not fp_found and err['pred'] == 1 and err['label'] == 0:
            print(f"\n[FAUX POSITIF] :\nTexte : {content[:400]}...")
            fp_found = True
        if not fn_found and err['pred'] == 0 and err['label'] == 1:
            print(f"\n[FAUX NÉGATIF] :\nTexte : {content[:400]}...")
            fn_found = True
        if fp_found and fn_found: break

    print("\n[SUCCÈS] Pipeline CNN terminé.")

if __name__ == "__main__":
    main()
