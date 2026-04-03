import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.models.baseline_mlp import BaselineMLP
from src.models.baseline_cnn import BaselineCNN
from src.data.prepare_data import get_dataloaders

class EarlyStopping:
    """
    Arrête l'entraînement si la perte de validation ne s'améliore pas après une certaine patience.
    Respecte la directive de régularisation du projet.
    """
    def __init__(self, patience=5, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, path)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, path)
            self.counter = 0

    def save_checkpoint(self, model, path):
        """Enregistre le meilleur modèle."""
        if self.verbose:
            print(f"Validation loss decreased. Saving model to {path}...")
        torch.save(model.state_dict(), path)

def calculate_accuracy(y_pred, y_true):
    """Calcul de la précision pour la classification binaire avec logits."""
    preds = torch.sigmoid(y_pred) > 0.5
    correct = (preds == y_true).float().sum()
    return correct / y_true.shape[0]

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience, clip_value, dry_run=False):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    model_path = "best_model.pt"
    
    for epoch in range(epochs):
        # PHASE D'ENTRAÎNEMENT
        model.train()
        train_loss = 0
        train_acc = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # ÉCRÊTAGE DE GRADIENT (Gradient Clipping)
            # Directive : seuil fixé à 2.0 pour éviter l'explosion des gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += calculate_accuracy(outputs, labels).item()
            
            if dry_run:
                print("Dry-run: un seul batch d'entraînement complété.")
                break
        
        avg_train_loss = train_loss / (1 if dry_run else len(train_loader))
        avg_train_acc = train_acc / (1 if dry_run else len(train_loader))
        
        # PHASE DE VALIDATION
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += calculate_accuracy(outputs, labels).item()
                if dry_run:
                    print("Dry-run: un seul batch de validation complété.")
                    break
        
        avg_val_loss = val_loss / (1 if dry_run else len(val_loader))
        avg_val_acc = val_acc / (1 if dry_run else len(val_loader))
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}")
        
        if dry_run:
            print("PHASE DE TEST (Dry-run) terminée avec succès.")
            return

        # VÉRIFICATION ARRÊT PRÉMATURÉ (Early Stopping)
        early_stopping(avg_val_loss, model, model_path)
        if early_stopping.early_stop:
            print("Arrêt prématuré activé.")
            break

def main():
    parser = argparse.ArgumentParser(description="Entraînement des modèles de détection de rumeurs.")
    parser.add_argument("--model", type=str, choices=["mlp", "cnn"], default="mlp", help="Modèle à entraîner.")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre maximal d'époques.")
    parser.add_argument("--batch_size", type=int, default=32, help="Taille des lots.")
    parser.add_argument("--lr", type=float, default=0.001, help="Taux d'apprentissage initial (Adam).")
    parser.add_argument("--patience", type=int, default=5, help="Patience pour l'Early Stopping.")
    parser.add_argument("--clip", type=float, default=2.0, help="Seuil d'écrêtage du gradient.")
    parser.add_argument("--dry-run", action="store_true", help="Mode test : une seule époque, un seul batch.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil utilisé : {device}")

    # CHARGEMENT DES DONNÉES VIA PIPELINE KDD
    max_len = 100
    train_loader, val_loader, test_loader, vocab_size = get_dataloaders(batch_size=args.batch_size, max_len=max_len)

    if args.model == "mlp":
        # Pour le MLP, on utilise max_len comme taille d'entrée (vecteur de séquences d'indices)
        # Note: Un MLP sur des indices n'est pas idéal, mais respecte la structure actuelle
        model = BaselineMLP(input_size=max_len).to(device)
    else:
        model = BaselineCNN(vocab_size=vocab_size).to(device)

    # CONFIGURATION OPTIMISEUR ET PERTE
    # Directive : Adam et BCEWithLogitsLoss pour une meilleure stabilité numérique
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Démarrage de l'entraînement : {args.model.upper()}")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        clip_value=args.clip,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
