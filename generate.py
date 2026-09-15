import sys
from pathlib import Path
import torch

# Ensure project root is in Python path
sys.path.append(str(Path(__file__).resolve().parent))

from src.tokenizer import Tokenizer
from src.model import GPT
from src.utils import generate_text, load_checkpoint


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model hyperparameters (must match training settings)
    vocab_size = 50257
    d_model = 128
    n_heads = 4
    n_layers = 2
    seq_len = 64
    
    # 1. Initialize Tokenizer & Model Architecture
    tokenizer = Tokenizer("gpt2")
    model = GPT(
        vocab_size=vocab_size, 
        d_model=d_model, 
        n_heads=n_heads, 
        n_layers=n_layers, 
        seq_len=seq_len
    ).to(device)

    # 2. Load Saved Weights from Checkpoint
    checkpoint_path = "checkpoints/llm_model.pt"
    try:
        load_checkpoint(checkpoint_path, model, device=device)
    except FileNotFoundError:
        print(f"Error: No checkpoint found at '{checkpoint_path}'. Run train.py first!")
        return

    # 3. Run Inference Prompt
    prompt = "Building a Large Language Model "
    print(f"\n--- Generating Output ---")
    print(f"Prompt: '{prompt}'")
    
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=30,
        device=device
    )
    print(f"Output: '{output}'\n")


if __name__ == "__main__":
    main()