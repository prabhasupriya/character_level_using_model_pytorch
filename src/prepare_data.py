import torch
import numpy as np
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing dataset at {file_path}. Please add a text file.")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if len(text) < 1000:
        print("Warning: Your input text is very short. Training might fail.")
        
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    
    encoded = np.array([char_to_int[ch] for ch in text])
    return encoded, vocab_size, char_to_int, int_to_char

def get_batches(data, batch_size, seq_length):
    # Calculate how many characters we need for one full batch
    batch_total = batch_size * seq_length
    n_batches = len(data) // batch_total
    
    if n_batches == 0:
        # Fallback for small datasets: adjust batch size to fit the data
        n_batches = 1
        batch_size = 1 
        
    data = data[:n_batches * batch_size * seq_length]
    data = data.reshape((batch_size, -1))
    
    for n in range(0, data.shape[1], seq_length):
        x = data[:, n:n+seq_length]
        # Target is input shifted by 1
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], data[:, n+seq_length]
        except IndexError:
            # Wrap around to the beginning for the very last character
            y[:, :-1], y[:, -1] = x[:, 1:], data[:, 0]
        
        yield torch.tensor(x).long(), torch.tensor(y).long()