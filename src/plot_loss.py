import matplotlib.pyplot as plt
import torch
import os

def plot_loss():
    plt.figure(figsize=(10, 6))
    
    # Check for both model files
    for model_name in ['lstm', 'transformer']:
        path = f'models/{model_name}_model.pth'
        if os.path.exists(path):
            checkpoint = torch.load(path)
            # Pull the loss history we saved during training
            losses = checkpoint['loss_history']
            plt.plot(range(1, len(losses) + 1), losses, label=f'{model_name.upper()}', marker='o')

    plt.title('Character-Level Model Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True)
    
    # Save to the results folder as required
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/loss_curves.png')
    print("Success! Loss curves saved to results/loss_curves.png")

if __name__ == "__main__":
    plot_loss()