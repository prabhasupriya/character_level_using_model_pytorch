# Character-Level Text Generation (LSTM vs. Transformer)

This project implements and compares two neural network architectures for character-level text generation using PyTorch.

## Setup and Installation
1. **Using Docker (Recommended):**
   - Build the container: `docker-compose build`
   - Train LSTM: `docker-compose run --rm app python src/train.py --model lstm`
   - Train Transformer: `docker-compose run --rm app python src/train.py --model transformer`
   - Generate Text: `docker-compose run --rm app python src/generate.py --model transformer`

2. **Manual Setup:**
   - Install dependencies: `pip install torch numpy matplotlib`
   - Run scripts directly from the `src/` directory.

## Project Structure
- `src/`: Model definitions and execution scripts.
- `input/`: Training data (Shakespeare text).
- `models/`: Saved model weights (.pth files).
- `results/`: Loss curves, generated samples, and comparison report.

## Findings
The Transformer model outperformed the LSTM in capturing the structural nuances of the text with a significantly lower perplexity score. Detailed analysis is available in `results/comparison_report.md`.

# youtude link -[watch youtude video here](https://youtu.be/o3SM2-WvL4A) 