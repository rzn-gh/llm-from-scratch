# LLM from Scratch

A clean, modular implementation of a GPT-style decoder-only Transformer built from scratch in PyTorch. This repository demonstrates how language models work under the hood, covering tokenization, sliding-window batching, multi-head self-attention, pre-LN architecture, autoregressive generation, and model training.

---

## Project Structure

```text
llm-from-scratch/
├── checkpoints/          # Saved model weights (.pt files)
├── src/
│   ├── dataset.py        # Sliding-window dataset and PyTorch DataLoader creation
│   ├── model.py          # Transformer decoder architecture (Attention, FFN, Blocks, GPT)
│   ├── tokenizer.py      # BPE tokenization wrapper around tiktoken (gpt2)
│   └── utils.py          # Autoregressive generation and checkpointing utilities
├── train.py              # Main training loop script
├── requirements.txt      # Project dependencies
├── LICENSE               # MIT License
└── README.md             # Project documentation

```

---

## Core Components

* **Tokenizer (`src/tokenizer.py`):** Wraps OpenAI's `tiktoken` (`gpt2` encoding) with a vocabulary size of 50,257 tokens.
* **Dataset (`src/dataset.py`):** Formats raw text into input-target pairs ($x$, $y$) shifted by one position for Next-Token Prediction using a sliding-window approach.
* **Model Architecture (`src/model.py`):**
* **Causal Self-Attention:** Multi-head attention with a triangular causal mask preventing future token visibility.
* **Feed-Forward Network (FFN):** Two-layer linear network with GELU activations expanded by 4x hidden dimension.
* **Transformer Block:** Pre-LayerNorm configuration with residual skip connections.
* **GPT:** Full network combining token and positional embeddings, stacked Transformer blocks, and tied embedding/lm_head weights.


* **Utilities (`src/utils.py`):** Handles checkpointing (`save_checkpoint`, `load_checkpoint`) and autoregressive text generation using greedy decoding or temperature-based sampling.

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE&utm_source=gemini) file for details.