import torch
import torch.nn as nn
import argparse
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Imports locaux
from src.data.prepare_data import get_dataloaders, load_fakenewsnet_data
from src.models.baseline_mlp import BaselineMLP
from src.models.baseline_cnn import BaselineCNN

def evaluate_model(model, test_loader, device, model_type):
    """
    Évalue le modèle sur le test_loader et retourne les prédictions et les étiquettes réelles.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    # Pour l'analyse d'erreurs (KDD Interprétation)
    misclassified_indices = []
    current_idx = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            batch_size = labels.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            # Conversion en float pour le MLP si nécessaire
            if model_type == "mlp":
                inputs = inputs.float()
                
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
            # Identifier les erreurs dans ce batch
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
    parser = argparse.ArgumentParser(description="Évaluation du modèle de détection de rumeurs.")
    parser.add_argument("--model", type=str, choices=["mlp", "cnn"], default="mlp", help="Type de modèle à évaluer.")
    parser.add_argument("--path", type=str, default="best_model.pt", help="Chemin vers le fichier du modèle sauvegardé.")
    parser.add_argument("--batch_size", type=int, default=32, help="Taille des lots.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil utilisé pour l'évaluation : {device}")

    # 1. Chargement des données et du vocabulaire
    max_len = 100
    _, _, test_loader, vocab_size = get_dataloaders(batch_size=args.batch_size, max_len=max_len)
    df = load_fakenewsnet_data()

    # 2. Instanciation et chargement du modèle
    if args.model == "mlp":
        model = BaselineMLP(input_size=max_len).to(device)
    else:
        model = BaselineCNN(vocab_size=vocab_size).to(device)
    
    if not os.path.exists(args.path):
        print(f"Erreur : Le fichier {args.path} n'existe pas. Veuillez d'abord entraîner le modèle.")
        return

    print(f"Chargement du modèle depuis {args.path}...")
    model.load_state_dict(torch.load(args.path, map_location=device))
    
    # 3. Évaluation
    print("Évaluation en cours sur le jeu de test...")
    y_pred, y_true, errors = evaluate_model(model, test_loader, device, args.model)
    
    # 4. Calcul des métriques via Scikit-Learn
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\n" + "="*30)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("="*30)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Précision : {prec:.4f}")
    print(f"Rappel    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("="*30)
    
    # 5. KDD Interprétation : Analyse visuelle des erreurs
    print("\n--- Analyse des erreurs (KDD Interprétation) ---")
    
    # Récupérer les indices originaux du dataset de test
    # test_loader.dataset est un Subset, il a un attribut 'indices'
    test_indices = test_loader.dataset.indices
    
    fp_found = False
    fn_found = False
    
    for err in errors:
        original_idx = test_indices[err['test_idx']]
        content = df.iloc[original_idx]['content']
        
        # Faux Positif (Prédit Fake (1), alors que c'est Real (0))
        if not fp_found and err['pred'] == 1 and err['label'] == 0:
            print(f"\n[FAUX POSITIF] - Prédit RUMEUR, alors que c'est RÉEL")
            print(f"Texte : {content[:500]}...")
            fp_found = True
            
        # Faux Négatif (Prédit Real (0), alors que c'est Fake (1))
        if not fn_found and err['pred'] == 0 and err['label'] == 1:
            print(f"\n[FAUX NÉGATIF] - Prédit RÉEL, alors que c'est RUMEUR")
            print(f"Texte : {content[:500]}...")
            fn_found = True
            
        if fp_found and fn_found:
            break
            
    if not fp_found:
        print("\nAucun Faux Positif trouvé dans le jeu de test.")
    if not fn_found:
        print("\nAucun Faux Négatif trouvé dans le jeu de test.")

if __name__ == "__main__":
    main()
