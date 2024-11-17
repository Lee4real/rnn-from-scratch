# RNN from Scratch

This project demonstrates the implementation and training of various Recurrent Neural Network (RNN) architectures from scratch using PyTorch. The models included are:

- Simple RNN
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

## Project Structure

- `main.py`: The main script to load data, initialize models, train them, and plot results.
- `model/`: Directory containing the model definitions.
  - `RNN.py`: Implementation of a simple RNN.
  - `LSTM.py`: Implementation of an LSTM.
  - `GRU.py`: Implementation of a GRU.
- `model/train.py`: Contains the `train_model` function to train the models and collect metrics.
- `data/preprocess_data.py`: Script to preprocess the IMDB dataset and create DataLoaders.
- `plots/`: Directory containing scripts to plot training results.
  - `plot_loss_curve.py`: Script to plot training and test loss curves.
  - `plot_accuracy.py`: Script to plot training and test accuracy.
  - `plot_training_time.py`: Script to plot average training time per epoch for each model.
- `requirements.txt`: List of required Python packages.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Additional Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/lee4real/rnn-from-scratch.git
   cd rnn-from-scratch
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

1. Ensure you have the IMDB dataset in CSV format. Update the path in `data/preprocess_data.py` if necessary.
2. Run the main script:
   ```bash
   python main.py
   ```

## Results

The training process will output the following plots:

- Training and test loss curves for each model.
- Training and test accuracy for each model.
- Average training time per epoch for each model.

## Acknowledgements

- The IMDB dataset used in this project is available at [Kaggle](https://www.kaggle.com/).
- The project uses PyTorch for building and training the models.

## License

This project is licensed under the MIT License.
