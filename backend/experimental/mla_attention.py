import torch
import torch.nn as nn
import math

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim, max_seq_len=4096):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = embed_dim // num_heads
        
        # Compression matrices
        self.key_compression = nn.Linear(embed_dim, latent_dim)
        self.value_compression = nn.Linear(embed_dim, latent_dim)
        self.query_compression = nn.Linear(embed_dim, latent_dim)
        
        # KV cache
        self.kv_cache = None
        self.cache_enabled = False
        
        # Rotary Positional Embedding (RoPE)
        self.register_buffer('freqs', self._compute_frequencies(max_seq_len))
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def _compute_frequencies(self, max_seq_len):
        freqs = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        return freqs.unsqueeze(0).unsqueeze(0)
        
    def _apply_rope(self, x):
        seq_len = x.size(1)
        sin = torch.sin(self.freqs[:, :seq_len])
        cos = torch.cos(self.freqs[:, :seq_len])
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        
    def enable_cache(self):
        """Enable KV caching for inference"""
        self.cache_enabled = True
        
    def disable_cache(self):
        """Disable KV caching"""
        self.cache_enabled = False
        self.clear_cache()
        
    def clear_cache(self):
        """Clear the KV cache"""
        self.kv_cache = None

    def forward(self, query, key, value, attn_mask=None):
        # Compress inputs
        query = self.query_compression(query)
        
        # Use cached keys/values if available
        if self.cache_enabled and self.kv_cache is not None:
            key, value = self.kv_cache
        else:
            key = self.key_compression(key)
            value = self.value_compression(value)
            if self.cache_enabled:
                self.kv_cache = (key, value)
        
        # Apply RoPE
        query = self._apply_rope(query)
        key = self._apply_rope(key)
        
        # Split into heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        
        # Combine heads and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.embed_dim)
        return self.out_proj(attn_output)
