import sys
from pathlib import Path
import math 
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in Python path for standalone execution
sys.path.append(str(Path(__file__).resolve().parent.parent))


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention block.
    """
    def __init__(self, d_model: int, n_heads: int, seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Key, Query, Value projections combined into a single linear layer
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model)

        # Causal mask to ensure attention is only applied to past and current positions
        # Lower-triangular matrix of ones registered as a buffer (not a trainable parameter)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
        ) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch size, Sequence length, Embedding dimension(d_model)

        # Calculate Query, Key, Value vectors for all heads in batch
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)

        # Reshape to (B, n_heads, T, head_dim) for multi-head parallel execution
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention: (Q @ K^T) / sqrt(head_dim)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply Causal Mask (replace 0s in mask with -infinity so softmax zeros them out)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        # Weighted combination of Value vectors
        y = att @ v # Shape: (B, n_heads, T, head_dim)
        
        # Re-assemble all head outputs side-by-side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output linear projection
        return self.c_proj(y)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Standard GPT architecture expands hidden layer size by 4x
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single Transformer Decoder Block with Pre-LayerNorm architecture.
    """
    def __init__(self, d_model: int, n_heads: int, seq_len: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, seq_len)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN pattern with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    Full GPT Decoder Language Model.
    """
    def __init__(self, vocab_size: int = 50257, d_model: int = 256, 
                 n_heads: int = 4, n_layers: int = 4, seq_len: int = 64):
        super().__init__()
        self.seq_len = seq_len

        # Token & Positional Embedding Lookup Tables
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)

        # Stack of N Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, seq_len) for _ in range(n_layers)
        ])

        # Final Layer Normalization & Language Model Linear Head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and final output projection
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        assert T <= self.seq_len, f"Cannot forward sequence length {T}, max context length is {self.seq_len}"

        # Generate positional indices [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Sum token embeddings and positional embeddings
        tok_emb = self.token_embedding(idx) # Shape: (B, T, d_model)
        pos_emb = self.position_embedding(pos) # Shape: (T, d_model)
        x = tok_emb + pos_emb

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply final norm and compute output logits over vocabulary
        x = self.ln_f(x)
        logits = self.lm_head(x) # Shape: (B, T, vocab_size)

        return logits


# --- Quick Verification Script ---
if __name__ == "__main__":
    # Small test model hyper-parameters
    vocab_size = 50257
    d_model = 128
    n_heads = 4
    n_layers = 2
    seq_len = 32

    model = GPT(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers, seq_len=seq_len)
    
    # Create dummy batch of token IDs: batch_size=2, sequence_length=16
    dummy_input = torch.randint(0, vocab_size, (2, 16))
    
    # Forward pass through model
    logits = model(dummy_input)

    # Calculate total parameter count
    total_params = sum(p.numel() for p in model.parameters())

    print("--- Model Architecture Test ---")
    print(f"Input Shape  : {dummy_input.shape}")
    print(f"Logits Shape : {logits.shape}")  # Expected: [2, 16, 50257]
    print(f"Total Model Parameters: {total_params:,}")