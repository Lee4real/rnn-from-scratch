import torch
import torch.optim as optim
from model import RNN, LSTM, GRU
from model.train import train_model
from data.preprocess_data import load_preprocessed_data
from plots.plot_loss_curve import plot_loss_curve
from plots.plot_accuracy import plot_accuracy
from plots.plot_training_time import plot_training_time

# load preprocessed data
train_loader, test_loader, word2idx = load_preprocessed_data()

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
EMBEDDING_DIM = 100  # Dimensionality of word embeddings
HIDDEN_DIM = 128    # Number of hidden units in the RNN
OUTPUT_DIM = 2      # Binary classification (positive/negative)
DROPOUT = 0.5       # Dropout rate

# Initialize models
rnn_model = RNN(vocab_size=len(word2idx), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT).to(device)
lstm_model = LSTM(vocab_size=len(word2idx), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT).to(device)
gru_model = GRU(vocab_size=len(word2idx), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT).to(device)

# Optimizers
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.001)

if __name__ == "__main__":
    # Train models
    rnn_results = train_model(rnn_model, train_loader, test_loader, rnn_optimizer, loss_fn, n_epochs=5, model_name="RNN")
    lstm_results = train_model(lstm_model, train_loader, test_loader, lstm_optimizer, loss_fn, n_epochs=5, model_name="LSTM")
    gru_results = train_model(gru_model, train_loader, test_loader, gru_optimizer, loss_fn, n_epochs=5, model_name="GRU")

    # Unpack results
    rnn_train_losses, rnn_test_losses, rnn_train_acc, rnn_test_acc, rnn_times = rnn_results
    lstm_train_losses, lstm_test_losses, lstm_train_acc, lstm_test_acc, lstm_times = lstm_results
    gru_train_losses, gru_test_losses, gru_train_acc, gru_test_acc, gru_times = gru_results

    # Plot loss curves
    plot_loss_curve(rnn_train_losses, rnn_test_losses, lstm_train_losses, lstm_test_losses, gru_train_losses, gru_test_losses)
    plot_accuracy(rnn_train_acc, rnn_test_acc, lstm_train_acc, lstm_test_acc, gru_train_acc, gru_test_acc)
    plot_training_time(rnn_times, lstm_times, gru_times)


