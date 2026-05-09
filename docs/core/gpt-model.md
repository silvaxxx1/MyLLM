# GPT Model

**File:** `myllm/model.py`
**Import:** `from myllm.model import GPT`

Pure PyTorch decoder-only transformer. Covers GPT-2, GPT-NeoX, LLaMA 1/2/3,
and other variants through `ModelConfig` switches — no subclassing needed.

---

## Architecture

```
GPT (nn.Module)
├── transformer.wte        nn.Embedding(vocab_size, n_embd)
├── transformer.wpe        nn.Embedding(block_size, n_embd)   # learned pos only
├── transformer.drop       nn.Dropout(dropout)
├── transformer.h          nn.ModuleList of Block × n_layer
│   └── Block
│       ├── ln_1           LayerNorm | RMSNorm
│       ├── attn           CausalSelfAttention
│       ├── ln_2           LayerNorm | RMSNorm
│       └── mlp            GptMLP | LLaMAMLP | GemmaMLP | …
├── transformer.ln_f       Final norm
└── lm_head                nn.Linear(n_embd, vocab_size, bias=False)
                           Weights tied to wte if weight_tying=True
```

**Forward pass:**
```
idx (B, T)
→ wte(idx) + wpe(pos)              # token + position embeddings
→ drop
→ for block in h: x = block(x)    # sequential or parallel residuals
→ ln_f(x)
→ lm_head(x)                       # (B, T, vocab_size)
```

---

## CausalSelfAttention

Implements MHA / MQA / GQA depending on `n_query_groups`:

| `n_query_groups` | Mode | Description |
|-----------------|------|-------------|
| `== n_head` | MHA | Standard multi-head attention |
| `== 1` | MQA | Single KV pair shared across all heads |
| `1 < n_query_groups < n_head` | GQA | Grouped-query attention |

**RoPE:** Applied to Q and K when `use_rope=True`.
Percentage of dimensions receiving RoPE is controlled by `rotary_percentage`.

**Forward signature:**
```python
def forward(
    self,
    x: torch.Tensor,        # (B, T, C)
    use_cache: bool = False,
    pos_offset: int = 0,    # token position index for KV cache
) -> torch.Tensor            # (B, T, C)
```

### KV Cache

Pre-allocated on first call to `model.initialize_kv_cache()`:

```python
model.initialize_kv_cache(
    batch_size: int,
    max_seq_len: int,
    dtype: torch.dtype,
)
```

Internal tensors:
```python
self.cache_k = torch.zeros(batch_size, max_seq_len, n_head, head_size)
self.cache_v = torch.zeros(batch_size, max_seq_len, n_head, head_size)
```

During generation with `use_cache=True`:
- New K/V written at `pos_offset`
- Attention computed over `cache_k[:, :pos_offset+1]`, `cache_v[:, :pos_offset+1]`

Reset before each new generation sequence:
```python
model.reset_cache()
```

---

## MLP Variants

### GptMLP (GPT-2)
```
x → fc (n_embd → 4×n_embd) → GELU → proj (4×n_embd → n_embd)
```

### GptNeoxMLP
Same as GptMLP but uses `intermediate_size` instead of `mlp_ratio`.

### LLaMAMLP (SwiGLU)
```
gate_proj(x) × silu(up_proj(x)) → down_proj
```
Gate projection controls what information flows through.
No bias; three separate weight matrices.

### GemmaMLP
```
gate_proj(x) × gelu(up_proj(x)) → down_proj
```
Same structure as LLaMA MLP but uses GeLU instead of SiLU.

### LLaMAMoE
Sparse mixture-of-experts wrapper around `LLaMAMLP`.
A router selects top-k experts per token. (Experimental)

---

## Normalisation

Controlled by `norm_class_name`:

| Value | Class | Used by |
|-------|-------|---------|
| `"LayerNorm"` | `nn.LayerNorm` | GPT-2, GPT-NeoX |
| `"RMSNorm"` | `nn.RMSNorm` | LLaMA 1/2/3, Gemma, Mistral |

Epsilon: `norm_eps` (default `1e-5`).

---

## Residual connection patterns

**Sequential (GPT-2 default, `parallel_residual=False`):**
```
x = x + attn(ln_1(x))
x = x + mlp(ln_2(x))
```

**Parallel (LLaMA style, `parallel_residual=True`):**
```
h = ln_1(x)
x = x + attn(h) + mlp(h)
```

When `shared_attention_norm=True` with parallel residual,
`ln_1` is shared between attention and MLP paths.

---

## Weight tying

When `weight_tying=True` (GPT-2 default):
```python
self.lm_head.weight = self.transformer.wte.weight
```

This halves the parameter count for the vocabulary projection and is standard
for GPT-2 style models. LLaMA sets `weight_tying=False`.

---

## Tensor shape annotations

All operations in `model.py` are annotated with shapes:
```python
# x: (B, T, C)
x = self.attn(x)    # (B, T, C)
```

Where:
- `B` = batch size
- `T` = sequence length
- `C` = `n_embd` (channel/hidden dimension)
- `H` = `n_head`
- `HS` = `head_size` = `C // H`

---

## Instantiation

```python
from myllm.model import GPT
from myllm import ModelConfig

config = ModelConfig.from_name('gpt2-small')
model = GPT(config)
model.eval()

import torch
x = torch.randint(0, config.vocab_size, (1, 10))
logits = model(x)           # (1, 10, 50257)
```

---

## Parameter count

```python
n_params = sum(p.numel() for p in model.parameters())
# gpt2-small: ~124M
# gpt2-medium: ~335M
# gpt2-large: ~774M
# gpt2-xl: ~1.5B
```
