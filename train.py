import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW

# Ensure project root is in Python path
sys.path.append(str(Path(__file__).resolve().parent))

from src.tokenizer import Tokenizer
from src.dataset import create_dataloader
from src.model import GPT
from src.utils import generate_text, save_checkpoint


def train():
    # --- 1. Configurations & Hyperparameters ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    vocab_size = 50257
    d_model = 128
    n_heads = 4
    n_layers = 2
    seq_len = 64
    batch_size = 4
    epochs = 100
    learning_rate = 1e-3

    # --- 2. Sample Training Corpus ---
    training_data = (
        "Building a Large Language Model from scratch requires understanding "
        "tokenization, datasets, neural network architectures, and training loops. "
        "Step by step, we construct every component in PyTorch using clean modular code. "
        "The Transformer model uses causal self-attention mechanisms to learn dependencies "
        "between words and predict the next token in a text sequence efficiently."
    )

    # --- 3. Initialize Components ---
    tokenizer = Tokenizer("gpt2")
    dataloader = create_dataloader(
        training_data, tokenizer, batch_size=batch_size, seq_len=seq_len, shuffle=True
    )
    
    model = GPT(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers, seq_len=seq_len
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # --- 4. Training Loop ---
    model.train()
    print("\nStarting Training...")
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero gradients from previous step
            optimizer.zero_grad()

            # Forward pass: shape (B, T, vocab_size)
            logits = model(inputs)

            # Reshape for CrossEntropyLoss: (B * T, vocab_size) vs (B * T)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

            # Backward pass & update weights
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{epochs:3d}] | Loss: {avg_loss:.4f}")

    # --- 5. Save Checkpoint & Generate Text ---
    print("\nTraining Complete!")
    save_checkpoint(model, optimizer, epochs, avg_loss, "checkpoints/llm_model.pt")

    print("\n--- Generating Text After Training ---")
    prompt = "Building a Large Language Model"
    sample_output = generate_text(model, tokenizer, prompt=prompt, max_new_tokens=50, temperature=0.1, device=device)
    print(f"Prompt : '{prompt}'")
    print(f"Output : '{sample_output}'")


if __name__ == "__main__":
    train()