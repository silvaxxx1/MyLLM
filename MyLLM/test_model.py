import pytest
import torch
import torch.nn as nn
from model import GPT, Block, CausalSelfAttention, RMSNorm, GptMLP, LLaMAMLP, pre_compute_freq, apply_rope
from config import Config

# Set random seed for reproducibility
torch.manual_seed(42)

# Fixture for a basic GPT-2 small configuration
@pytest.fixture
def gpt2_config():
    return Config.from_name("gpt2-small")

# Fixture for a LLaMA-like configuration
@pytest.fixture
def llama_config():
    return Config.from_name("llama2-7b")

# Test Config class
def test_config_initialization(gpt2_config):
    assert gpt2_config.name == "gpt2-small"
    assert gpt2_config.block_size == 1024
    assert gpt2_config.vocab_size == 50257
    assert gpt2_config.n_layer == 12
    assert gpt2_config.n_head == 12
    assert gpt2_config.n_embd == 768
    assert gpt2_config.padded_vocab_size == 50257

def test_config_validation(gpt2_config):
    # Should pass validation
    gpt2_config.validate()
    
    # Test invalid n_embd and n_head combination
    gpt2_config.n_embd = 767  # Not divisible by n_head=12
    with pytest.raises(AssertionError):
        gpt2_config.validate()

def test_config_from_name():
    config = Config.from_name("llama2-7b")
    assert config.name == "llama2-7b"
    assert config.block_size == 4096
    assert config.norm_class_name == "RMSNorm"
    assert config.mlp_class_name == "LLaMAMLP"
    assert config.rotary_percentage == 1.0

# Test GPT model initialization and forward pass
def test_gpt_initialization(gpt2_config):
    model = GPT(gpt2_config)
    assert isinstance(model.lm_head, nn.Linear)
    assert isinstance(model.wte, nn.Embedding)
    assert isinstance(model.wpe, nn.Embedding)  # GPT-2 specific
    assert len(model.transformer) == gpt2_config.n_layer
    assert isinstance(model.ln_f, nn.Module)

def test_gpt_forward(gpt2_config):
    model = GPT(gpt2_config)
    batch_size, seq_len = 2, 64
    x = torch.randint(0, gpt2_config.vocab_size, (batch_size, seq_len))
    logits = model(x)
    assert logits.shape == (batch_size, seq_len, gpt2_config.n_embd)

def test_gpt_forward_exceeds_block_size(gpt2_config):
    model = GPT(gpt2_config)
    batch_size, seq_len = 2, gpt2_config.block_size + 1
    x = torch.randint(0, gpt2_config.vocab_size, (batch_size, seq_len))
    with pytest.raises(ValueError):
        model(x)

# Test Block
def test_block_initialization(gpt2_config):
    block = Block(gpt2_config, block_idx=0)
    assert isinstance(block.norm1, nn.Module)
    assert isinstance(block.attn, CausalSelfAttention)
    assert isinstance(block.mlp, nn.Module)

def test_block_forward(gpt2_config):
    block = Block(gpt2_config, block_idx=0)
    batch_size, seq_len, n_embd = 2, 64, gpt2_config.n_embd
    x = torch.randn(batch_size, seq_len, n_embd)
    output = block(x)
    assert output.shape == (batch_size, seq_len, n_embd)

# Test CausalSelfAttention
def test_causal_self_attention_initialization(gpt2_config):
    attn = CausalSelfAttention(gpt2_config, block_idx=0)
    assert isinstance(attn.qkv, nn.Linear)
    assert isinstance(attn.proj, nn.Linear)

def test_causal_self_attention_forward(gpt2_config):
    attn = CausalSelfAttention(gpt2_config, block_idx=0)
    batch_size, seq_len, n_embd = 2, 64, gpt2_config.n_embd
    x = torch.randn(batch_size, seq_len, n_embd)
    output = attn(x)
    assert output.shape == (batch_size, seq_len, n_embd)

# Test RoPE (Rotary Position Embeddings)
def test_pre_compute_freq(gpt2_config):
    gpt2_config.use_rope = True
    freqs = pre_compute_freq(gpt2_config)
    assert freqs.shape == (gpt2_config.block_size, gpt2_config.n_embd // gpt2_config.n_head // 2)
    assert torch.is_complex(freqs)

def test_apply_rope(gpt2_config):
    gpt2_config.use_rope = True
    batch_size, n_head, seq_len, head_dim = 2, gpt2_config.n_head, 64, gpt2_config.n_embd // gpt2_config.n_head
    x = torch.randn(batch_size, n_head, seq_len, head_dim)
    freqs = pre_compute_freq(gpt2_config)
    output = apply_rope(x, freqs)
    assert output.shape == (batch_size, n_head, seq_len, head_dim)

# Test RMSNorm
def test_rms_norm():
    size = 768
    norm = RMSNorm(size=size, eps=1e-6)
    batch_size, seq_len = 2, 64
    x = torch.randn(batch_size, seq_len, size)
    output = norm(x)
    assert output.shape == (batch_size, seq_len, size)

# Test MLP variants
def test_gpt_mlp(gpt2_config):
    mlp = GptMLP(gpt2_config)
    batch_size, seq_len, n_embd = 2, 64, gpt2_config.n_embd
    x = torch.randn(batch_size, seq_len, n_embd)
    output = mlp(x)
    assert output.shape == (batch_size, seq_len, n_embd)

def test_llama_mlp(llama_config):
    mlp = LLaMAMLP(llama_config)
    batch_size, seq_len, n_embd = 2, 64, llama_config.n_embd
    x = torch.randn(batch_size, seq_len, n_embd)
    output = mlp(x)
    assert output.shape == (batch_size, seq_len, n_embd)

# Test LLaMA-specific features
def test_llama_parallel_residual(llama_config):
    block = Block(llama_config, block_idx=0)
    batch_size, seq_len, n_embd = 2, 64, llama_config.n_embd
    x = torch.randn(batch_size, seq_len, n_embd)
    output = block(x)
    assert output.shape == (batch_size, seq_len, n_embd)

def test_llama_unsupported_config(llama_config):
    llama_config.parallel_residual = False
    llama_config.shared_attention_norm = True
    with pytest.raises(NotImplementedError):
        Block(llama_config, block_idx=0)

# Test device compatibility
def test_gpt_on_gpu(gpt2_config):
    if torch.cuda.is_available():
        model = GPT(gpt2_config).cuda()
        batch_size, seq_len = 2, 64
        x = torch.randint(0, gpt2_config.vocab_size, (batch_size, seq_len)).cuda()
        logits = model(x)
        assert logits.shape == (batch_size, seq_len, gpt2_config.n_embd)
        assert logits.device.type == "cuda"

if __name__ == "__main__":
    pytest.main(["-v"])