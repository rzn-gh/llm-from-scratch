import tiktoken
import torch

class Tokenizer:
    """
    Tokenizer class wrapping OpenAI's tiktoken library.
    Handles encoding string text to token IDs and decoding token IDs back to text.
    """
    def __init__(self, model_name: str = "gpt2"):
        # load the predefined PBE encoding scheme (GPT-2 default vocabulary size = 50,257)
        self.encoder = tiktoken.get_encoding(model_name)
        self.vocab_size = self.encoder.n_vocab

    def encode(self, text: str) -> torch.Tensor:
        """
        Converts a text string into a 1D PyTorch Tensor of token IDs.

        Args:
            text (str): Input text sequence.

        Returns:
            torch.Tensor: Tensor of integer token IDs.
        """
        # tiktoken.encode returns a list of integers
        tokens = self.encoder.encode(text, allowed_special={"<|endoftext|>"})
        # Convert list to a 1D PyTorch tensor of type 64-bit integer
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        """
        Converts a PyTorch Tensor or list of token IDs back into string text.

        Args: 
            tokens (torch.Tensor or list). Input token IDs.

        Returns:
            str: Decoded string text.
        """   
        # Convert PyTorch tensor to a Python list if necessary 
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        return self.encoder.decode(tokens)

#--- Quick Self-Test Execution ---
if __name__ == "__main__":
    tokenizer = Tokenizer("gpt2")
    sample_text = "Hello, world! Building an LLM from scratch."
    
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Vocabulary Size : {tokenizer.vocab_size}")
    print(f"Original Text   : '{sample_text}'")
    print(f"Encoded Tokens  : {encoded}")
    print(f"Decoded Text    : '{decoded}'")
    print(f"Token Count     : {len(encoded)}")
    