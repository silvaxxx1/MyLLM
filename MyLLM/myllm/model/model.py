import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from config import Config

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        t = torch.arange(seq_len, device=x.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, None, :]

def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [batch, seq_len, n_head, head_dim]
    xcos, xsin = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    rope_cos, rope_sin = rope_cache.cos()[..., None], rope_cache.sin()[..., None]
    x_out = torch.stack([
        xcos * rope_cos - xsin * rope_sin,
        xsin * rope_cos + xcos * rope_sin
    ], dim=-1)
    return x_out.flatten(-2)

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with support for both standard and grouped-query attention"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        
        # Key, Query, Value projections
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Rotary embeddings if needed
        self.rotary = config.rotary_percentage > 0
        if self.rotary:
            self.rope = RotaryEmbedding(self.head_dim)
            
        # Grouped-query attention setup
        self.n_kv_heads = getattr(config, 'n_kv_heads', config.n_head)
        self.n_kv_groups = self.n_head // self.n_kv_heads if self.n_kv_heads else 1

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        
        # Linear projections
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape and transpose for attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings if enabled
        if self.rotary:
            rope_cache = self.rope(x, seq_len=T)
            q = apply_rotary_pos_emb(q, rope_cache)
            k = apply_rotary_pos_emb(k, rope_cache)
        
        # Grouped-query attention handling
        if self.n_kv_heads < self.n_head:
            k = k.repeat_interleave(self.n_kv_groups, dim=1)
            v = v.repeat_interleave(self.n_kv_groups, dim=1)
        
        # Compute attention with flash attention when available
        if hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch 2.0's native flash attention
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # Manual implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attention_mask is not None:
                att = att.masked_fill(attention_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = F.dropout(att, p=self.dropout, training=self.training)
            y = att @ v
        
        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class FeedForward(nn.Module):
    """Feed-forward network with configurable activation function"""
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
        # Set activation function
        self.activation = F.gelu if config.activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with configurable components"""
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ff = FeedForward(config)
        
        # Normalization layers
        norm_class = RMSNorm if config.norm_class_name == "RMSNorm" else nn.LayerNorm
        self.norm1 = norm_class(config.n_embd, eps=getattr(config, 'norm_eps', 1e-5))
        self.norm2 = norm_class(config.n_embd, eps=getattr(config, 'norm_eps', 1e-5))
        
        # Parallel attention for LLaMA models
        self.parallel_residual = getattr(config, 'parallel_residual', False)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.parallel_residual:
            # LLaMA-style parallel residual connections
            norm_x = self.norm1(x)
            h = x + self.attn(norm_x, attention_mask)
            out = h + self.ff(self.norm2(h))
        else:
            # GPT-style sequential residual connections
            h = x + self.attn(self.norm1(x), attention_mask)
            out = h + self.ff(self.norm2(h))
        return out

class TransformerDecoder(nn.Module):
    """Unified Transformer Decoder supporting GPT-2, LLaMA-2, and LLaMA-3 architectures"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Position embeddings (only for GPT-2 style models)
        if not getattr(config, 'rotary_percentage', 0):
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        
        # Final layer normalization
        norm_class = RMSNorm if config.norm_class_name == "RMSNorm" else nn.LayerNorm
        self.norm = norm_class(config.n_embd, eps=getattr(config, 'norm_eps', 1e-5))
        
        # Output projection
        self.proj = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.size()
        assert T <= self.config.block_size, f"Input sequence length ({T}) exceeds maximum length ({self.config.block_size})"

        # Get token embeddings
        x = self.tok_emb(input_ids)
        
        # Add positional embeddings if not using rotary
        if not getattr(self.config, 'rotary_percentage', 0):
            pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
            x = x + self.pos_emb(pos)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
            
        # Final normalization and projection
        x = self.norm(x)
        logits = self.proj(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0
    ) -> torch.LongTensor:
        """Generate text tokens using various sampling strategies"""
        for _ in range(max_length - input_ids.size(1)):
            # Truncate sequence if needed
            input_ids_cond = input_ids if input_ids.size(1) <= self.config.block_size else input_ids[:, -self.config.block_size:]
            
            # Get predictions
            logits = self(input_ids_cond)
            logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(input_ids.size(1)):
                    logits[torch.arange(logits.size(0)), input_ids[:, i]] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
        return input_ids

    def load_pretrained(self, checkpoint_path: str, strict: bool = True):
        """Load pretrained weights from a checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        self.load_state_dict(state_dict, strict=strict)
        return self
    


# ...existing code...

if __name__ == "__main__":
    # Create a model instance using GPT-2 small configuration
    config = Config.from_name("gpt2-medium")
    model = TransformerDecoder(config)
    
    # Move model to available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Example input (replace with your tokenizer)
    input_ids = torch.tensor([[1, 2, 3]]).to(device)  # Your tokenized input
    
    # Generate text with sampling parameters
    output = model.generate(
        input_ids=input_ids,
        max_length=50,          # Shorter sequence for testing
        temperature=0.7,
        top_k=50,              # Add top-k filtering
        top_p=0.9,
        repetition_penalty=1.2  # Add repetition penalty
    )
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Generated sequence: {output.tolist()}")
