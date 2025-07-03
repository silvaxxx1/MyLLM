import pytest
import torch
import os
import json
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import your LLM class
from api import LLM 
from model import GPT
from config import Config  # Replace 'your_module' with actual module name

@pytest.fixture
def sample_config():
    return Config(
        vocab_size=50257,
        block_size=1024,
        n_layer=2,  # Smaller for testing
        n_head=2,
        n_embd=128,  # Smaller embedding size
        dropout=0.1,
        bias=True,
        tokenizer_name="gpt2"
    )

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def mock_hf_model():
    model = MagicMock(spec=AutoModelForCausalLM)
    model.generate.return_value = torch.tensor([[1, 2, 3]])
    return model

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=AutoTokenizer)
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.return_tensors = "pt"
    tokenizer.__call__.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    tokenizer.batch_decode.return_value = ["Mock generated text"]
    return tokenizer

def test_llm_initialization(sample_config):
    llm = LLM(sample_config)
    assert llm.model is None
    assert llm.tokenizer is None
    assert llm.device == torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert not llm.is_initialized
    assert not llm.is_loaded

def test_load_config_from_dict():
    config_dict = {
        "vocab_size": 1000,
        "block_size": 512,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 128
    }
    llm = LLM(config_dict)
    assert isinstance(llm.config, Config)
    assert llm.config.vocab_size == 1000

def test_load_config_from_file(sample_config, temp_dir):
    config_path = Path(temp_dir) / "config.json"
    with open(config_path, 'w') as f:
        json.dump(sample_config.__dict__, f)
    
    llm = LLM(str(config_path))
    assert isinstance(llm.config, Config)

def test_initialize_model(sample_config):
    llm = LLM(sample_config)
    llm.initialize()
    
    assert llm.model is not None
    assert isinstance(llm.model, GPT)
    assert llm.is_initialized
    assert not llm.is_loaded
    assert llm.tokenizer is not None

def test_load_huggingface_model(sample_config, mock_hf_model, mock_tokenizer):
    with patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=mock_hf_model), \
         patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        
        llm = LLM(sample_config)
        llm.load("gpt2")
        
        assert llm.model is not None
        assert llm.tokenizer is not None
        assert llm.is_loaded
        assert not llm.is_initialized

def test_generate_text(sample_config, mock_hf_model, mock_tokenizer):
    with patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=mock_hf_model), \
         patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        
        llm = LLM(sample_config)
        llm.load("gpt2")
        
        result = llm.generate("Test prompt", max_length=10)
        assert result == "Mock generated text"

def test_generate_text_with_custom_model(sample_config):
    llm = LLM(sample_config)
    llm.initialize()
    
    # Mock the model's forward pass
    llm.model = MagicMock()
    llm.model.return_value = torch.randn(1, 3, sample_config.vocab_size)  # Mock logits
    
    result = llm.generate("Test prompt", max_length=5)
    assert isinstance(result, str)

def test_generation_errors(sample_config):
    llm = LLM(sample_config)
    
    # Test not initialized/loaded
    with pytest.raises(RuntimeError):
        llm.generate("Test prompt")
    
    # Test no tokenizer
    llm.initialize()
    llm.tokenizer = None
    with pytest.raises(RuntimeError):
        llm.generate("Test prompt")

def test_save_model(sample_config, temp_dir):
    llm = LLM(sample_config)
    llm.initialize()
    
    save_path = Path(temp_dir) / "saved_model"
    llm.save(str(save_path))
    
    assert (save_path / "config.json").exists()
    assert (save_path / "pytorch_model.bin").exists()
    assert (save_path / "tokenizer.json").exists()

def test_device_management(sample_config):
    llm = LLM(sample_config)
    llm.initialize()
    
    original_device = llm.device
    llm.to('cpu')
    assert llm.device == torch.device('cpu')
    assert next(llm.model.parameters()).device == torch.device('cpu')
    
    llm.to(original_device)
    assert llm.device == original_device

def test_memory_usage(sample_config):
    llm = LLM(sample_config)
    llm.initialize()
    
    mem_info = llm.get_memory_usage()
    if torch.cuda.is_available():
        assert 'allocated_gb' in mem_info
    else:
        assert 'error' in mem_info

def test_parameter_count(sample_config):
    llm = LLM(sample_config)
    llm.initialize()
    
    count = llm._count_parameters()
    assert count > 0
    assert isinstance(count, int)

def test_custom_generate_method(sample_config):
    llm = LLM(sample_config)
    llm.initialize()
    
    # Mock model forward pass
    llm.model = MagicMock()
    llm.model.return_value = torch.randn(1, 3, sample_config.vocab_size)  # Mock logits
    
    input_ids = torch.tensor([[1, 2, 3]])
    output = llm._custom_generate(input_ids, max_new_tokens=5)
    assert isinstance(output, torch.Tensor)
    assert output.shape[1] > input_ids.shape[1]  # Should have generated tokens

def test_weight_initialization(sample_config):
    llm = LLM(sample_config)
    llm.initialize(pretrained_weights=True)
    
    # Check some weights
    for name, param in llm.model.named_parameters():
        if 'weight' in name:
            if 'ln' not in name:  # Skip layer norms
                assert torch.std(param).item() > 0  # Should be initialized

def test_load_from_checkpoint(sample_config, temp_dir):
    # Create dummy checkpoint
    checkpoint_path = Path(temp_dir) / "checkpoint.pth"
    torch.save({"model_state_dict": {}}, checkpoint_path)
    
    llm = LLM(sample_config)
    llm.load(str(checkpoint_path), from_hf=False)
    
    assert llm.is_loaded

def test_transfer_hf_weights(sample_config, mock_hf_model):
    llm = LLM(sample_config)
    llm.model = MagicMock()
    llm.model.state_dict.return_value = {"transformer.block_0.attn.qkv.weight": torch.randn(128, 128)}
    
    mock_hf_model.state_dict.return_value = {"transformer.h.0.attn.c_attn.weight": torch.randn(128, 128)}
    
    llm._transfer_hf_weights(mock_hf_model)
    # Should have attempted to transfer weights
    assert llm.model.load_state_dict.called

def test_repr(sample_config):
    llm = LLM(sample_config)
    repr_str = repr(llm)
    assert "LLM" in repr_str
    assert "device=" in repr_str
    
    llm.initialize()
    repr_str = repr(llm)
    assert "initialized" in repr_str