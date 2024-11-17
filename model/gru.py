import torch
from torch import nn

class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5):
        super(GRU, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)  # Initialize embeddings

        # GRU weights
        self.Wxr = nn.Parameter(torch.empty(embedding_dim, hidden_dim))  # Reset gate
        self.Whr = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.br = nn.Parameter(torch.zeros(hidden_dim))

        self.Wxz = nn.Parameter(torch.empty(embedding_dim, hidden_dim))  # Update gate
        self.Whz = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bz = nn.Parameter(torch.zeros(hidden_dim))

        self.Wxh = nn.Parameter(torch.empty(embedding_dim, hidden_dim))  # Candidate hidden state
        self.Whh = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bh = nn.Parameter(torch.zeros(hidden_dim))

        # Output weights
        self.Why = nn.Parameter(torch.empty(hidden_dim, output_dim))
        self.by = nn.Parameter(torch.zeros(output_dim))

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        for param in [self.Wxr, self.Whr, self.Wxz, self.Whz, self.Wxh, self.Whh, self.Why]:
            nn.init.xavier_uniform_(param)

    def forward(self, x):
        x_embedded = self.embedding(x)  # Shape: [batch_size, seq_len, embedding_dim]
        h = torch.zeros(x.size(0), self.Why.size(0)).to(x.device)  # Hidden state

        for t in range(x_embedded.size(1)):
            xt = x_embedded[:, t, :]
            r = torch.sigmoid(torch.matmul(xt, self.Wxr) + torch.matmul(h, self.Whr) + self.br)
            z = torch.sigmoid(torch.matmul(xt, self.Wxz) + torch.matmul(h, self.Whz) + self.bz)
            h_tilde = torch.tanh(torch.matmul(xt, self.Wxh) + torch.matmul(r * h, self.Whh) + self.bh)
            h = (1 - z) * h + z * h_tilde  # Update hidden state

        h = self.dropout(h)  # Apply dropout to the hidden state
        out = torch.matmul(h, self.Why) + self.by
        return out