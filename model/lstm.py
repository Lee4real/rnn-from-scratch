import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5):
        super(LSTM, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)  # Initialize embeddings

        # LSTM weights
        self.Wxi = nn.Parameter(torch.empty(embedding_dim, hidden_dim))
        self.Whi = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bi = nn.Parameter(torch.zeros(hidden_dim))

        self.Wxf = nn.Parameter(torch.empty(embedding_dim, hidden_dim))
        self.Whf = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bf = nn.Parameter(torch.zeros(hidden_dim))

        self.Wxc = nn.Parameter(torch.empty(embedding_dim, hidden_dim))
        self.Whc = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bc = nn.Parameter(torch.zeros(hidden_dim))

        self.Wxo = nn.Parameter(torch.empty(embedding_dim, hidden_dim))
        self.Who = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bo = nn.Parameter(torch.zeros(hidden_dim))

        # Output weights
        self.Why = nn.Parameter(torch.empty(hidden_dim, output_dim))
        self.by = nn.Parameter(torch.zeros(output_dim))

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        for param in [self.Wxi, self.Whi, self.Wxf, self.Whf, self.Wxc, self.Whc, self.Wxo, self.Who, self.Why]:
            nn.init.xavier_uniform_(param)

    def forward(self, x):
        x_embedded = self.embedding(x)  # Shape: [batch_size, seq_len, embedding_dim]
        h = torch.zeros(x.size(0), self.Why.size(0)).to(x.device)  # Hidden state
        c = torch.zeros(x.size(0), self.Why.size(0)).to(x.device)  # Cell state

        for t in range(x_embedded.size(1)):
            xt = x_embedded[:, t, :]
            i = torch.sigmoid(torch.matmul(xt, self.Wxi) + torch.matmul(h, self.Whi) + self.bi)
            f = torch.sigmoid(torch.matmul(xt, self.Wxf) + torch.matmul(h, self.Whf) + self.bf)
            g = torch.tanh(torch.matmul(xt, self.Wxc) + torch.matmul(h, self.Whc) + self.bc)
            o = torch.sigmoid(torch.matmul(xt, self.Wxo) + torch.matmul(h, self.Who) + self.bo)

            c = f * c + i * g  # Update cell state
            h = o * torch.tanh(c)  # Update hidden state

        h = self.dropout(h)  # Apply dropout to the hidden state
        out = torch.matmul(h, self.Why) + self.by
        return out