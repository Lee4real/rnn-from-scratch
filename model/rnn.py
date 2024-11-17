import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5):
        super(RNN, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)  # Initialize embeddings

        # RNN weights
        self.Wxh = nn.Parameter(torch.empty(embedding_dim, hidden_dim))
        self.Whh = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bh = nn.Parameter(torch.zeros(hidden_dim))

        # Output weights
        self.Why = nn.Parameter(torch.empty(hidden_dim, output_dim))
        self.by = nn.Parameter(torch.zeros(output_dim))

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.xavier_uniform_(self.Wxh)
        nn.init.orthogonal_(self.Whh)
        nn.init.xavier_uniform_(self.Why)

    def forward(self, x):
        x_embedded = self.embedding(x)
        h = torch.zeros(x.size(0), self.Whh.size(0)).to(x.device)

        for t in range(x_embedded.size(1)):
            xt = x_embedded[:, t, :]
            h = torch.tanh(torch.matmul(xt, self.Wxh) + torch.matmul(h, self.Whh) + self.bh)

        h = self.dropout(h)  # Apply dropout to the hidden state
        out = torch.matmul(h, self.Why) + self.by
        return out