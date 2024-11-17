import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import string
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_preprocessed_data():
    nltk.download('punkt')
    nltk.download('stopwords')


    data = pd.read_csv('/kaggle/input/imdb-dataset-sentiment-analysis-in-csv-format/Train.csv')

    # Initialize stemmer and stopwords
    stemmer = PorterStemmer()

    stop_words = set(stopwords.words('english'))

    # Preprocessing function
    def preprocess(text):
        # Lowercase the text
        text = text.lower()
        
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stop words
        tokens = [token for token in tokens if token not in stop_words]
        
        # Optional: Apply stemming
        tokens = [stemmer.stem(token) for token in tokens]
        
        return tokens

    # Apply preprocessing to the reviews column
    data['tokens'] = data['text'].apply(preprocess)

    # Create a vocabulary from the entire dataset (all tokens)
    all_tokens = [token for tokens in data['tokens'] for token in tokens]
    vocab = Counter(all_tokens)

    # Limit the vocabulary size (optional)
    MAX_VOCAB_SIZE = 25000
    vocab = dict(vocab.most_common(MAX_VOCAB_SIZE - 1))  # Include the <unk> token
    vocab['<unk>'] = 0  # Add an unknown token for out-of-vocabulary words

    # Create a word-to-index mapping
    word2idx = {word: idx for idx, (word, _) in enumerate(vocab.items())}

    # Convert tokens to numerical indices
    def tokens_to_indices(tokens):
        return [word2idx.get(token, word2idx['<unk>']) for token in tokens]

    # Apply to all reviews
    data['indexed'] = data['tokens'].apply(tokens_to_indices)

    # Define padding function to pad sequences to the same length
    def pad_sequences(sequences, max_len):
        padded_sequences = [seq[:max_len] if len(seq) > max_len else seq + [0] * (max_len - len(seq)) for seq in sequences]
        return torch.tensor(padded_sequences)

    # Define maximum sequence length (you can tune this value)
    MAX_SEQUENCE_LENGTH = 500

    # Pad all sequences to the same length
    padded_data = pad_sequences(data['indexed'], MAX_SEQUENCE_LENGTH)

    # Convert to tensor
    labels = torch.tensor(data['label'].values)

    # Create TensorDataset
    trainig_dataset = TensorDataset(padded_data[0:28000], labels[0:28000])
    testing_dataset = TensorDataset(padded_data[28000:-1], labels[28000:-1])

    # Create DataLoader (batch size of 64)
    batch_size = 64
    train_loader = DataLoader(trainig_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, word2idx