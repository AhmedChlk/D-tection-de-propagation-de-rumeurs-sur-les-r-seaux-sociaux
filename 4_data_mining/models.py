import torch
import torch.nn as nn

class FakeNewsClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FakeNewsClassifier, self).__init__()
        
        # Couche dense 1
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Normalisation par lots (Batch Normalization) pour stabiliser les activations
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        # Prévention du surajustement via Dropout
        self.drop1 = nn.Dropout(p=0.5)
        
        # Couche dense 2
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.5)
        
        # Couche de sortie
        self.out = nn.Linear(hidden_size // 2, num_classes)
        
        # Initialisation de He (Kaiming) pour les couches utilisant ReLU
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        
        # La fonction de perte d'entropie croisée (CrossEntropyLoss) en PyTorch 
        # inclut le Softmax, on renvoie donc les logits.
        x = self.out(x)
        return x
