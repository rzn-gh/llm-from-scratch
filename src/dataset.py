import sys
from pathlib import Path
import torch 
from torch.utils.data import Dataset, DataLoader

# Ensure project root is in Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.tokenizer import Tokenizer 

class TextDataset(Dataset):
    """
    PyTorch Dataset for Next-Token Prediction.
    Chunks tokenized text into overlapping sequences of length `seq_len`.
    """
    def __init__(self, text: str, tokenizer: Tokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Tokenize the entire raw text corpus into integer IDs
        self.tokens = self.tokenizer.encode(text)

        # Check if the text is long enough for at least one training sequence
        if len(self.tokens) <= seq_len:
            raise ValueError(
                f"Dataset text length ({len(self.tokens)} tokens) must be "
                f"greater than context length seq_len ({seq_len} tokens)."
            )

    def __len__(self) -> int:
        # Total number of sequence we can extract
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx: int):
        # Extract input chunk of length seq_len
        x = self.tokens[idx : idx + self.seq_len]
        # Extract target chunk shifted by 1 token
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]

        return x, y


def create_dataloader(text: str, tokenizer: Tokenizer, batch_size: int = 4,
                      seq_len: int = 8, shuffle: bool = True) -> DataLoader:
    """
    Helper function to build a PyTorch DataLoader for batched training.
    """
    dataset = TextDataset(text, tokenizer, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


# --- Quick Self-Test Execution ---
if __name__ == "__main__":
    tokenizer = Tokenizer("gpt2")
    sample_text = (
        "Building a Large Language Model from scratch requires understanding "
        "tokenization, datasets, neural network architectures, and training loops. "
        "Step by step, we construct every component in PyTorch."
    )

    # Context window of 8 tokens, batch size of 2 
    batch_size = 2
    seq_len = 8

    dataloader = create_dataloader(sample_text, tokenizer, batch_size=batch_size, seq_len=seq_len, shuffle=False)

    # Inspect the first batch
    for inputs, targets in dataloader:
        print(f"Inputs Shape  (x): {inputs.shape}")   # Expected: [batch_size, seq_len]
        print(f"Targets Shape (y): {targets.shape}")  # Expected: [batch_size, seq_len]
        print("\nFirst Sample in Batch:")
        print(f"  Input Tokens  (x): {inputs[0].tolist()}")
        print(f"  Target Tokens (y): {targets[0].tolist()}")
        print(f"  Decoded Input    : '{tokenizer.decode(inputs[0])}'")
        print(f"  Decoded Target   : '{tokenizer.decode(targets[0])}'")
        break
