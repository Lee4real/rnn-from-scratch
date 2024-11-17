import matplotlib.pyplot as plt
import numpy as np

def plot_training_time(rnn_times, lstm_times, gru_times):
    # Training Time Comparison
    models = ['RNN', 'LSTM', 'GRU']
    avg_times = [np.mean(rnn_times), np.mean(lstm_times), np.mean(gru_times)]

    plt.figure(figsize=(8, 6))
    plt.bar(models, avg_times, color=['blue', 'orange', 'green'])
    plt.xlabel('Models')
    plt.ylabel('Average Training Time per Epoch (s)')
    plt.title('Training Time Comparison')
    plt.show()