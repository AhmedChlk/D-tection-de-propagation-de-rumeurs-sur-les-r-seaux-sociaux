import torch
import torch.nn as nn
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Imports locaux
from src.data.prepare_data import get_dataloaders, load_fakenewsnet_data
from src.models.baseline_cnn import BaselineCNN

def evaluate_model(model, test_loader, device):
    """
    Exécute l'évaluation sur le test_loader.
    """
    model.eval()
    all_preds = []
    all_labels = []
    misclassified_indices = []
    current_idx = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            batch_size = labels.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(inputs)
            # Application de Sigmoïde sur les logits pour le seuillage
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
            # Stockage des erreurs pour l'analyse KDD
            preds_np = preds.cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()
            for i in range(batch_size):
                if preds_np[i] != labels_np[i]:
                    misclassified_indices.append({
                        'test_idx': current_idx + i,
                        'pred': preds_np[i],
                        'label': labels_np[i]
                    })
            current_idx += batch_size
            
    return np.array(all_preds), np.array(all_labels), misclassified_indices

def main():
    parser = argparse.ArgumentParser(description="Évaluation KDD du modèle de détection de rumeurs.")
    parser.add_argument("--path", type=str, default="best_model.pt", help="Chemin vers le modèle (.pt).")
    parser.add_argument("--batch_size", type=int, default=32, help="Taille des lots.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil : {device}")

    # 1. Chargement des données (max_len=200 pour correspondre à l'entraînement massif)
    max_len = 200
    _, _, test_loader, vocab_size = get_dataloaders(batch_size=args.batch_size, max_len=max_len)
    df = load_fakenewsnet_data()

    # 2. Instanciation du modèle CNN
    model = BaselineCNN(vocab_size=vocab_size).to(device)
    
    if not os.path.exists(args.path):
        print(f"Erreur : Modèle {args.path} introuvable.")
        return

    model.load_state_dict(torch.load(args.path, map_location=device))
    
    # 3. Évaluation
    print("Évaluation du modèle CNN...")
    y_pred, y_true, errors = evaluate_model(model, test_loader, device)
    
    # 4. Métriques Scikit-Learn
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*40)
    print("RÉSULTATS DE L'ÉVALUATION FINALE (CNN)")
    print("="*40)
    print(f"Accuracy   : {acc:.4f}")
    print(f"Précision  : {prec:.4f}")
    print(f"Rappel     : {rec:.4f}")
    print(f"F1-Score   : {f1:.4f}")
    print("\nMatrice de Confusion :")
    print(cm)
    print("="*40)
    
    # 5. KDD Interprétation : Analyse visuelle
    print("\n--- Analyse des Erreurs (KDD Interprétation) ---")
    test_indices = test_loader.dataset.indices
    
    fp_found, fn_found = False, False
    for err in errors:
        content = df.iloc[test_indices[err['test_idx']]]['content']
        
        if not fp_found and err['pred'] == 1 and err['label'] == 0:
            print(f"\n[FAUX POSITIF] - Prédit RUMEUR, alors que c'est RÉEL")
            print(f"Texte : {content[:500]}...")
            fp_found = True
            
        if not fn_found and err['pred'] == 0 and err['label'] == 1:
            print(f"\n[FAUX NÉGATIF] - Prédit RÉEL, alors que c'est RUMEUR")
            print(f"Texte : {content[:500]}...")
            fn_found = True
            
        if fp_found and fn_found: break

if __name__ == "__main__":
    main()
