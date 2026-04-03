import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    """
    Modèle CNN (1D) pour la classification de fake news.
    Architecture adaptée pour le traitement de séquences (ex: plongements de mots).
    
    Règles respectées :
    - Initialisation de He (Kaiming) pour ReLU.
    - Batch Normalization pour la stabilisation.
    - Dropout pour la régularisation.
    - Classification binaire avec Logits.
    """
    def __init__(self, vocab_size, embedding_dim=50, n_filters=20, filter_sizes=[3, 4, 5], dropout=0.6):
        super(BaselineCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutions 1D avec différentes tailles de filtres (capturant différents n-grammes)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=fs),
                nn.BatchNorm1d(n_filters),
                nn.ReLU()
            )
            for fs in filter_sizes
        ])
        
        # Couche dense finale (retourne les logits)
        self.fc = nn.Linear(len(filter_sizes) * n_filters, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Initialisation de He (Kaiming)
        self._initialize_weights()

    def _initialize_weights(self):
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
        
        conved = [conv(embedded) for conv in self.convs]
        # conved[n]: [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled[n]: [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat: [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)

if __name__ == "__main__":
    # Test rapide
    V = 5000 # Taille du vocabulaire
    model = BaselineCNN(vocab_size=V)
    print(model)
    
    dummy_input = torch.randint(0, V, (8, 50)) # [batch_size=8, seq_len=50]
    output = model(dummy_input)
    print(f"\nSortie (shape): {output.shape}")
