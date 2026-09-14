import sys 
from pathlib import Path
import torch 
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.tokenizer import Tokenizer


def generate_text(model: nn.Module, tokenizer: Tokenizer, prompt: str,
                  max_new_tokens: int = 30, temperature: float = 1.0,
                  device: str = "cpu") -> str:
    """
    Generates new tokens autoregressively given a starting text prompt.
    """
    model.eval()

    # Encode prompt string into integer tensor
    input_ids = tokenizer.encode(prompt).unsqueeze(0).to(device) # Shape: (1, seq_len)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop input context if it exceeds the model's max sequence length
            idx_cond = input_ids if input_ids.size(1) <= model.seq_len else input_ids[:, -model.seq_len:]

        # Forward pass To get output logits
        logits = model(idx_cond)

        # Focus onl^on the logits of the last token position 
        logits = logits[:, -1, :] / temperature

        # Use Greedy Decoding (always pick highest probability token): 
        next_token = torch.argmax(logits, dim=1, keepdim=True)

        # Append sampled token to sequence
        input_ids = torch.cat((input_ids, next_token), dim=1)

    # Decode full token sequence back to text
    return tokenizer.decode(input_ids.squeeze(0))


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, loss: float, filepath: str = "checkpoints/model.pt"):
    """
    Saves model weights and optimizer state to disk.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to '{filepath}'")


def load_checkpoint(filepath: str, model: nn.Module, optimizer: torch.optim.Optimizer = None, device: str = "cpu"):   
    """
    Loads saved model weights and optimizer state from disk.
    """ 
    if not Path(filepath).exists():
        raise FileNotFoundError(f"No checkpoint found at '{filepath}'")

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state-dict"])

    print(f"Loaded checkpoint from '{filepath}' (Epoch {checkpoint.get('epoch', 'N/A')})")

    
# --- Self-Test Execution ---
if __name__ == "__main__":
    from src.model import GPT

    tokenizer = Tokenizer("gpt2")
    model = GPT(vocab_size=50257, d_model=128, n_heads=4, n_layers=2, seq_len=32)
    
    # Test generation with an untrained model (outputs random text)
    sample_prompt = "Building an LLM"
    output = generate_text(model, tokenizer, prompt=sample_prompt, max_new_tokens=10)
    
    print("--- Autoregressive Generation Test ---")
    print(f"Prompt : '{sample_prompt}'")
    print(f"Output : '{output}'")   