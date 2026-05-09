import torch
import torch.nn.functional as F
import json
import argparse
import os
from model_lstm import LSTMModel
from model_transformer import TransformerModel

def sample(logits, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

def generate_text(model, start_str, length, char_to_int, int_to_char, model_type, device):
    model.eval()
    # Ensure all characters in seed are in vocabulary
    filtered_seed = "".join([c for c in start_str if c in char_to_int])
    if not filtered_seed:
        filtered_seed = list(char_to_int.keys())[0] # Fallback to first char in vocab
        
    input_ids = torch.tensor([char_to_int[s] for s in filtered_seed]).unsqueeze(0).to(device)
    generated_text = filtered_seed
    
    with torch.no_grad():
        for _ in range(length):
            if model_type == 'lstm':
                hidden = model.init_hidden(input_ids.size(0))
                output, _ = model(input_ids, hidden)
            else:
                output = model(input_ids)
            
            # Get logits for the LAST character predicted
            last_logits = output[-1, :]
            idx = sample(last_logits, temperature=1.0) 
            char = int_to_char[idx]
            generated_text += char
            
            # Prepare next input
            new_id = torch.tensor([[idx]]).to(device)
            input_ids = torch.cat((input_ids, new_id), dim=1)
            if input_ids.size(1) > 100:
                input_ids = input_ids[:, 1:]
                
    return generated_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['lstm', 'transformer'], required=True)
    args = parser.parse_args()
    
    device = torch.device('cpu')
    model_path = f'models/{args.model}_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    checkpoint = torch.load(model_path, weights_only=True)
    c2i, i2c = checkpoint['char_to_int'], checkpoint['int_to_char']
    vocab_size = checkpoint['vocab_size']
    
    if args.model == 'lstm':
        model = LSTMModel(vocab_size).to(device)
    else:
        model = TransformerModel(vocab_size).to(device)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # FIXED SEED: Using only characters guaranteed to be in your text snippet
    seed = "SAMPSON" 
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    temps = [0.5, 1.0, 1.5]
    model_results = {}

    print(f"Generating samples for {args.model}...")
    for t in temps:
        # Temperature sampling logic applied here
        samples = []
        for _ in range(2):
            samples.append(generate_text(model, seed, 100, c2i, i2c, args.model, device))
        model_results[f"temperature_{t}"] = samples
    
    json_path = os.path.join(results_dir, 'generated_samples.json')
    
    # Handle the JSON error by starting fresh if file is corrupt
    final_data = {}
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        try:
            with open(json_path, 'r') as f:
                final_data = json.load(f)
        except json.JSONDecodeError:
            final_data = {}

    final_data[args.model] = model_results
    with open(json_path, 'w') as f:
        json.dump(final_data, f, indent=4)
    print(f"Success! Saved {args.model} samples to {json_path}")

if __name__ == "__main__":
    main()