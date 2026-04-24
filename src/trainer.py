"""
trainer.py — Entraîneur
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, f1_score

def train_model(model, train_loader, val_loader, optimizer, criterion, device, 
                epochs: int = 15, patience: int = 4, is_bert: bool = False, 
                save_path: str = "best_model.pt"): # <--- NOUVEAU PARAMÈTRE
    
    model = model.to(device)
    best_val_f1, patience_counter = 0.0, 0

    print(f"\n  {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Acc':>7} | {'Val F1':>6}")
    print("  " + "-" * 52)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, all_preds, all_labels = 0.0, [], []
        
        for batch in train_loader:
            if is_bert:
                inputs = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)}
                labels = batch['labels'].to(device)
                out = model(**inputs)
            else:
                X, labels = batch[0].to(device), batch[1].to(device)
                out = model(X)
                
            optimizer.zero_grad()
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(out.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        train_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                if is_bert:
                    inputs = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)}
                    labels = batch['labels'].to(device)
                    out = model(**inputs)
                else:
                    X, labels = batch[0].to(device), batch[1].to(device)
                    out = model(X)
                    
                val_preds.extend(out.argmax(dim=1).cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

        print(f"  {epoch:>5} | {total_loss/len(train_loader):>10.4f} | {train_acc:>9.4f} | {val_acc:>7.4f} | {val_f1:>6.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path) # <--- SAUVEGARDE DYNAMIQUE
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [INFO] Early stopping activé à l'époque {epoch}.")
                break

    return best_val_f1