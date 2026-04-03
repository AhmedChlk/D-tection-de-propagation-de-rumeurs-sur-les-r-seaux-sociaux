import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    """
    Modèle de référence : Perceptron Multicouches (PMC) pour la classification binaire.
    
    Architecture :
    - Couches cachées en "entonnoir" (nombre de neurones décroissant).
    - Activation : ReLU (cachées), Logits (sortie).
    - Stabilisation : Batch Normalization.
    - Régularisation : Dropout (p=0.5).
    - Initialisation : He (Kaiming).
    """
    def __init__(self, input_size, hidden_layers=[512, 256, 128]):
        super(BaselineMLP, self).__init__()
        
        layers = []
        in_features = input_size
        
        # Construction des couches cachées en entonnoir
        for h_size in hidden_layers:
            # Couche Linéaire
            layers.append(nn.Linear(in_features, h_size))
            # Normalisation par lots (Batch Normalization)
            layers.append(nn.BatchNorm1d(h_size))
            # Activation ReLU
            layers.append(nn.ReLU())
            # Régularisation Dropout
            layers.append(nn.Dropout(p=0.5))
            
            in_features = h_size
            
        # Couche de sortie pour classification binaire (logits)
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_features, 1)
        
        # Initialisation explicite de He (Kaiming)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Applique l'initialisation de He (Kaiming) sur les poids des couches linéaires.
        Les biais sont initialisés à zéro.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Passage vers l'avant (forward pass).
        Retourne les logits bruts.
        """
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

if __name__ == "__main__":
    # Test rapide de l'architecture
    input_dim = 1000
    model = BaselineMLP(input_size=input_dim)
    print(model)
    
    # Test de passage (batch_size=8)
    dummy_input = torch.randn(8, input_dim)
    output = model(dummy_input)
    print(f"\nSortie du modèle (shape): {output.shape}")
    print(f"Exemple de probabilités :\n{output.detach()}")
