import matplotlib.pyplot as plt

def plot_accuracy(rnn_train_acc, rnn_test_acc, lstm_train_acc, lstm_test_acc, gru_train_acc, gru_test_acc):
    # Plot Accuracy
    plt.figure(figsize=(12, 6))

    # Training Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(rnn_train_acc, label="RNN Train Accuracy", color='blue')
    plt.plot(lstm_train_acc, label="LSTM Train Accuracy", color='orange')
    plt.plot(gru_train_acc, label="GRU Train Accuracy", color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Comparison')
    plt.legend()

    # Test Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(rnn_test_acc, label="RNN Test Accuracy", color='blue')
    plt.plot(lstm_test_acc, label="LSTM Test Accuracy", color='orange')
    plt.plot(gru_test_acc, label="GRU Test Accuracy", color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()