import matplotlib.pyplot as plt

def plot_loss_curve(rnn_train_losses, rnn_test_losses, lstm_train_losses, lstm_test_losses, gru_train_losses, gru_test_losses):
    # Plot Loss Curves
    plt.figure(figsize=(12, 6))

    # Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(rnn_train_losses, label="RNN Train Loss", color='blue')
    plt.plot(lstm_train_losses, label="LSTM Train Loss", color='orange')
    plt.plot(gru_train_losses, label="GRU Train Loss", color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()

    # Test Loss
    plt.subplot(1, 2, 2)
    plt.plot(rnn_test_losses, label="RNN Test Loss", color='blue')
    plt.plot(lstm_test_losses, label="LSTM Test Loss", color='orange')
    plt.plot(gru_test_losses, label="GRU Test Loss", color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Test Loss Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()
