# Neural-Machine-Translation
Sure, here's a comprehensive README for your project:

---

# Transformer-Based Language Model

## Overview

This project implements a Transformer-based language model for natural language processing tasks. It includes components for tokenization, embedding, multi-head attention, and visualizing attention matrices. The code leverages the PyTorch library for building and training the model.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Tokenization](#tokenization)
  - [Dataset Mapping](#dataset-mapping)
  - [Multi-Head Attention](#multi-head-attention)
  - [Embedding Layer](#embedding-layer)
  - [Attention Matrix Visualization](#attention-matrix-visualization)

```

- `transformers/`: Contains modules for tokenization, attention, and embeddings.
- `visualization/`: Contains the module for plotting attention matrices.
- `data/`: Contains modules for dataset loading and processing.
- `README.md`: Project documentation.
- `requirements.txt`: List of Python dependencies.

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/transformer-language-model.git
    cd transformer-language-model
    ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Tokenization

Tokenization is performed using pre-trained tokenizers for source and target languages.

```python
from transformers import AutoTokenizer

source_tokenizer = AutoTokenizer.from_pretrained("resources/tokenizer_fr")
target_tokenizer = AutoTokenizer.from_pretrained("resources/tokenizer_en")

# Example usage
example_sentence = "we have an example"
tokenizer_output = target_tokenizer(example_sentence)
decoded_sequence = [target_tokenizer.decode(token) for token in tokenizer_output["input_ids"]]
reconstructed_sentence = "".join(decoded_sequence).replace("▁", " ")
```

### Dataset Mapping

Mapping examples in the dataset to tokenized input IDs.

```python
from typing import Dict, List

def map_example(example: Dict[str, str]) -> Dict[str, List[int]]:
    source_text = example["text_en"]
    target_text = example["text_fr"]
    source_tokens = source_tokenizer.encode(source_text, add_special_tokens=True)
    target_tokens = target_tokenizer.encode(target_text, add_special_tokens=True)
    return {"encoder_input_ids": source_tokens, "decoder_input_ids": target_tokens}

# Apply mapping to dataset
tokenized_datasets = raw_text_datasets.map(map_example, batched=False)
tokenized_datasets = tokenized_datasets.remove_columns(raw_text_datasets.column_names["train"])
```

### Multi-Head Attention

Implementation of the multi-head attention mechanism.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, is_causal_attention: bool = False, is_cross_attention: bool = False):
        super().__init__()
        self.head_dim = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        # Define projection layers for query, key, value, and output
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.FloatTensor, key_padding_mask: torch.BoolTensor, encoder_outputs: Optional[torch.FloatTensor] = None):
        query = self.query_projection(hidden_states)
        key = self.key_projection(hidden_states)
        value = self.value_projection(hidden_states)
        query = query.view(hidden_states.size(0), hidden_states.size(1), self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = key.view(hidden_states.size(0), hidden_states.size(1), self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = value.view(hidden_states.size(0), hidden_states.size(1), self.num_attention_heads, self.head_dim).transpose(1, 2)
        if self.is_causal_attention:
            causal_mask = self.causal_attention_mask(hidden_states.size(1), device=hidden_states.device)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = (torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())).masked_fill_(key_padding_mask, float('-inf'))
            attention_scores += causal_mask
        attention_probs = torch.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(hidden_states.size(0), hidden_states.size(1), self.hidden_size)
        output = self.output_projection(context)
        return output, attention_probs
```

### Embedding Layer

Defines the embedding layer with token and positional embeddings.

```python
import torch
import torch.nn as nn

class TransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_sequence_length: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_sequence_length, hidden_size)
        self.register_buffer('position_ids', torch.arange(max_sequence_length).expand((1, -1)))

    def compute_logits(self, decoder_output: torch.FloatTensor) -> torch.FloatTensor:
        logits = torch.matmul(decoder_output, self.token_embeddings.weight.T)
        return logits

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        token_embeddings = self.token_embeddings(input_ids)
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        return embeddings
```

### Attention Matrix Visualization

Plotting the attention matrix as a heatmap.

```python
import matplotlib.pyplot as plt

def plot_attention_matrix(attention_matrix, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(attention_matrix, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Key Token Position')
    plt.ylabel('Query Token Position')
    plt.title(title)
    plt.show()

# Example usage
# plot_attention_matrix(cross_attention_weights[0,0].detach().numpy(), "cross-attention, head 1")
# plot_attention_matrix(cross_attention_weights[0,1].detach().numpy(), "cross-attention, head 2")
# plot_attention_matrix(causal_attention_weights[0,0].detach().numpy(), "causal self-attention, head 1")
# plot_attention_matrix(causal_attention_weights[0,1].detach().numpy(), "causal self-attention, head 2")
```
