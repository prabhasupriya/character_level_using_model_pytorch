# Character-Level Text Generation: LSTM vs Transformer

### Perplexity Comparison
| Model       | Final Training Loss | Perplexity ($e^{Loss}$) |
|-------------|---------------------|--------------------------|
| LSTM        | 3.2292              | 25.26                    |
| Transformer | 1.9330              | 6.91                     |

*Perplexity represents the model's uncertainty. A lower score means the model is more confident in its character predictions.*

### Qualitative Analysis
1. **Architecture Comparison:** The Transformer achieved a significantly lower loss (~1.93) compared to the LSTM (~3.23). This is because the self-attention mechanism is better at capturing character patterns than the linear recurrence of the LSTM.
2. **Text Coherence:** While both models are in early stages, the Transformer began to learn structural elements like line breaks and capitalization (e.g., "SAMPSON" followed by uppercase dialogue starters) much faster than the LSTM.
3. **Effect of Temperature:** 
   - **0.5:** The models were more "safe" but repetitive. 
   - **1.0:** This provided the most "Shakespeare-like" character distributions.
   - **1.5:** The text became highly chaotic as the models took more risks with low-probability characters.