"""Language Model component of the hybrid system."""

import torch
import torch.nn as nn


class LanguageModel(nn.Module):
    """
    Language Model component that handles text understanding and generation.
    Designed to be lightweight and Windows 7 compatible.
    """
    
    def __init__(self, vocab_size=10000, embedding_dim=256, hidden_dim=512, num_layers=4):
        """
        Initialize the Language Model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: Hidden dimension size
            num_layers: Number of transformer layers
        """
        super(LanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        self.pos_embedding = nn.Embedding(512, embedding_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the language model.
        
        Args:
            input_ids: Tensor of token indices [batch_size, seq_len]
            attention_mask: Optional attention mask
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        token_embeds = self.embedding(input_ids)
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.pos_embedding(positions)
        
        # Combine embeddings
        embeds = token_embeds + pos_embeds
        
        # Transform
        embeds = embeds.transpose(0, 1)  # [seq_len, batch_size, embedding_dim]
        transformed = self.transformer(embeds, src_key_padding_mask=attention_mask)
        transformed = transformed.transpose(0, 1)  # [batch_size, seq_len, embedding_dim]
        
        # Project to vocabulary
        logits = self.output_proj(transformed)
        
        return logits
    
    def generate(self, prompt_ids, max_length=100, temperature=1.0):
        """
        Generate text autoregressively.
        
        Args:
            prompt_ids: Starting token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            generated_ids: Generated token sequence
        """
        self.eval()
        with torch.no_grad():
            current_ids = prompt_ids.clone()
            
            for _ in range(max_length):
                logits = self.forward(current_ids)
                next_token_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Stop if EOS token (assume token 2)
                if next_token.item() == 2:
                    break
                    
        return current_ids
    
    def encode_text(self, input_ids):
        """
        Encode text to latent representation.
        
        Args:
            input_ids: Token IDs
            
        Returns:
            encoding: Latent representation
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        token_embeds = self.embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.pos_embedding(positions)
        embeds = token_embeds + pos_embeds
        
        # Transform
        embeds = embeds.transpose(0, 1)
        transformed = self.transformer(embeds)
        
        # Pool (mean pooling)
        encoding = transformed.mean(dim=0)
        
        return encoding
