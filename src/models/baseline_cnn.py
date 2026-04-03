import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    """
    Modèle CNN (1D) haute performance pour la détection de rumeurs.
    Optimisé pour les datasets massifs selon les directives GEMINI.md.
    
    Architecture :
    - Embedding trainable pour apprendre des représentations denses.
    - Convolutions Multi-échelles (3, 4, 5) pour capturer les n-grammes.
    - Batch Normalization après chaque convolution pour stabiliser les activations.
    - Global Max Pooling pour extraire les caractéristiques les plus saillantes.
    - Dropout pour la régularisation anti-overfitting.
    - Initialisation de He (Kaiming) pour ReLU.
    """
    def __init__(self, vocab_size, embedding_dim=100, n_filters=50, filter_sizes=[3, 4, 5], dropout=0.5):
        super(BaselineCNN, self).__init__()
        
        # Couche d'Embedding (KDD Transformation)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutions 1D parallèles avec Batch Normalization (Directive DL)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=fs),
                nn.BatchNorm1d(n_filters), # Stabilisation des activations
                nn.ReLU()
            )
            for fs in filter_sizes
        ])
        
        # Couche de sortie (Logits bruts)
        self.fc = nn.Linear(len(filter_sizes) * n_filters, 1)
        self.dropout = nn.Dropout(dropout) # Régularisation
        
        # Initialisation explicite de He (Directive DL)
        self._initialize_weights()

    def _initialize_weights(self):
        """Applique l'initialisation de He (Kaiming) sur les poids."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, text):
        # text: [batch size, sent len]
        
        embedded = self.embedding(text)
        # embedded: [batch size, sent len, emb dim]
        
        embedded = embedded.permute(0, 2, 1)
        # embedded: [batch size, emb dim, sent len]
        
        # Application des convolutions et activation ReLU (via Sequential)
        conved = [conv(embedded) for conv in self.convs]
        
        # Global Max Pooling 1D
        pooled = [nn.functional.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved]
        
        # Concaténation et Dropout
        cat = self.dropout(torch.cat(pooled, dim=1))
        
        # Retourne les Logits pour BCEWithLogitsLoss
        return self.fc(cat)

if __name__ == "__main__":
    V = 5000
    model = BaselineCNN(vocab_size=V)
    print(model)
    dummy_input = torch.randint(0, V, (8, 200))
    output = model(dummy_input)
    print(f"\nSortie Logits (shape): {output.shape}")
