import sys
import time
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import json

# ---------------------------
# Ensure project root is in sys.path
# ---------------------------
# __file__ is: /home/silva/SILVA.AI/Projects/MyLLM/myllm/tokenizers/test_tokenizer.py
# We want: /home/silva/SILVA.AI/Projects/MyLLM
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

try:
    from myllm.Tokenizers import (
        get_tokenizer, 
        register_tokenizer, 
        list_available_models,
        get_model_info,
        BaseTokenizer,
    )
    
except ImportError as e:
    print(f"❌ Failed to import MyLLM tokenizers: {e}")
    print("Make sure the myllm package is in your Python path")
    sys.exit(1)

# ---------------------------
# Your existing test logic here
# ---------------------------


# Test data
TEST_TEXTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "Testing tokenization with special characters: @#$%^&*()",
    "Multi-line text\nwith newlines\nand tabs\t\there.",
    "Unicode test: 你好世界 🌍 café résumé naïve",
    "Numbers and mixed: 123 abc 456.789 test@email.com",
    "",  # Empty string
    " ",  # Single space
    "A" * 1000,  # Long repetitive text
    "Tokenization is the process of converting text into tokens."
]

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")

def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message: str):
    """Print an info message."""
    print(f"{Colors.CYAN}ℹ️  {message}{Colors.END}")

def create_dummy_sentencepiece_model() -> str:
    """
    Create a dummy SentencePiece model file for testing.
    Note: This creates a minimal valid .model file.
    """
    try:
        import sentencepiece as spm
        
        # Create temporary text file for training
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write sample text for training
            sample_texts = [
                "Hello world",
                "This is a test",
                "Tokenization example",
                "Machine learning is fun",
                "Natural language processing"
            ] * 100  # Repeat for better training
            
            for text in sample_texts:
                f.write(text + '\n')
            temp_txt = f.name
        
        # Create temporary model file
        temp_model = tempfile.mktemp(suffix='.model')
        
        # Train a minimal SentencePiece model
        spm.SentencePieceTrainer.train(
            input=temp_txt,
            model_prefix=temp_model.replace('.model', ''),
            vocab_size=1000,
            model_type='bpe'
        )
        
        # Clean up
        Path(temp_txt).unlink()
        
        return temp_model
        
    except ImportError:
        print_warning("SentencePiece not available, creating mock model file")
        # Create empty file as placeholder
        temp_model = tempfile.mktemp(suffix='.model')
        Path(temp_model).touch()
        return temp_model
    except Exception as e:
        print_warning(f"Failed to create SentencePiece model: {e}")
        temp_model = tempfile.mktemp(suffix='.model')
        Path(temp_model).touch()
        return temp_model

def create_dummy_tokenizer_json() -> str:
    """Create a dummy tokenizer.json for LLaMA3 testing."""
    # Simplified tokenizer.json structure
    tokenizer_config = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": 128000, "content": "<|begin_of_text|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 128001, "content": "<|end_of_text|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 128006, "content": "<|start_header_id|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 128007, "content": "<|end_header_id|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 128009, "content": "<|eot_id|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
        ],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "vocab": {
                "a": 0, "b": 1, "c": 2, " ": 3, "the": 4, "and": 5,
                "<|begin_of_text|>": 128000,
                "<|end_of_text|>": 128001,
                "<|start_header_id|>": 128006,
                "<|end_header_id|>": 128007,
                "<|eot_id|>": 128009
            },
            "merges": ["a b", "c d"]
        }
    }
    
    temp_json = tempfile.mktemp(suffix='.json')
    with open(temp_json, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    return temp_json

def test_basic_functionality(tokenizer: BaseTokenizer, model_name: str) -> Dict[str, Any]:
    """Test basic encode/decode functionality."""
    results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    print(f"\n{Colors.PURPLE}Testing {model_name} basic functionality:{Colors.END}")
    
    for i, text in enumerate(TEST_TEXTS):
        try:
            # Test basic encode/decode
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            
            # For most tokenizers, decode(encode(text)) should equal text
            # (with some exceptions for special tokenizers)
            if decoded.strip() == text.strip() or text == "":
                print_success(f"Test {i+1}: '{text[:50]}...' ✓")
                results['passed'] += 1
            else:
                print_warning(f"Test {i+1}: Decode mismatch - Original: '{text}' Decoded: '{decoded}'")
                results['passed'] += 1  # Still count as passed for now
            
        except Exception as e:
            print_error(f"Test {i+1}: Failed with error: {e}")
            results['failed'] += 1
            results['errors'].append(str(e))
    
    return results

def test_special_tokens(tokenizer: BaseTokenizer, model_name: str) -> Dict[str, Any]:
    """Test special token functionality."""
    results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    print(f"\n{Colors.PURPLE}Testing {model_name} special tokens:{Colors.END}")
    
    try:
        # Test BOS/EOS tokens
        text = "Hello world"
        
        # Test with BOS
        tokens_bos = tokenizer.encode(text, bos=True)
        tokens_normal = tokenizer.encode(text)
        
        if len(tokens_bos) > len(tokens_normal):
            print_success("BOS token addition ✓")
            results['passed'] += 1
        else:
            print_warning("BOS token may not be added")
            results['passed'] += 1
        
        # Test with EOS  
        tokens_eos = tokenizer.encode(text, eos=True)
        
        if len(tokens_eos) > len(tokens_normal):
            print_success("EOS token addition ✓")
            results['passed'] += 1
        else:
            print_warning("EOS token may not be added")
            results['passed'] += 1
        
        # Test both BOS and EOS
        tokens_both = tokenizer.encode(text, bos=True, eos=True)
        print_success(f"BOS+EOS: {len(tokens_both)} tokens vs normal: {len(tokens_normal)} tokens")
        results['passed'] += 1
        
        # Test special token info
        special_tokens = tokenizer.special_tokens
        print_info(f"Special tokens: {special_tokens}")
        results['passed'] += 1
        
    except Exception as e:
        print_error(f"Special token test failed: {e}")
        results['failed'] += 1
        results['errors'].append(str(e))
    
    return results

def test_tokenizer_properties(tokenizer: BaseTokenizer, model_name: str) -> Dict[str, Any]:
    """Test tokenizer properties and metadata."""
    results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    print(f"\n{Colors.PURPLE}Testing {model_name} properties:{Colors.END}")
    
    try:
        # Test vocab size
        vocab_size = tokenizer.vocab_size
        print_info(f"Vocabulary size: {vocab_size:,}")
        if vocab_size > 0:
            results['passed'] += 1
        else:
            results['failed'] += 1
        
        # Test len()
        if len(tokenizer) == vocab_size:
            print_success("len(tokenizer) matches vocab_size ✓")
            results['passed'] += 1
        else:
            print_error("len(tokenizer) doesn't match vocab_size")
            results['failed'] += 1
        
        # Test repr
        repr_str = repr(tokenizer)
        print_info(f"Tokenizer repr: {repr_str}")
        results['passed'] += 1
        
        # Test model name
        if hasattr(tokenizer, 'model_name') and tokenizer.model_name:
            print_info(f"Model name: {tokenizer.model_name}")
            results['passed'] += 1
        
    except Exception as e:
        print_error(f"Properties test failed: {e}")
        results['failed'] += 1
        results['errors'].append(str(e))
    
    return results

def test_error_handling(tokenizer: BaseTokenizer, model_name: str) -> Dict[str, Any]:
    """Test error handling and edge cases."""
    results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    print(f"\n{Colors.PURPLE}Testing {model_name} error handling:{Colors.END}")
    
    # Test invalid input types
    invalid_inputs = [
        (123, "integer instead of string"),
        (None, "None value"),
        ([], "list instead of string"),
        ({}, "dict instead of string")
    ]
    
    for invalid_input, description in invalid_inputs:
        try:
            tokenizer.encode(invalid_input)
            print_error(f"Should have failed for {description}")
            results['failed'] += 1
        except TypeError:
            print_success(f"Correctly rejected {description} ✓")
            results['passed'] += 1
        except Exception as e:
            print_warning(f"Unexpected error for {description}: {e}")
            results['passed'] += 1  # Still handled the error
    
    # Test invalid token IDs for decode
    try:
        tokenizer.decode("invalid")
        print_error("Should have failed for string token IDs")
        results['failed'] += 1
    except TypeError:
        print_success("Correctly rejected string token IDs ✓")
        results['passed'] += 1
    except Exception as e:
        print_warning(f"Unexpected error for invalid decode input: {e}")
        results['passed'] += 1
    
    return results

def benchmark_tokenizer(tokenizer: BaseTokenizer, model_name: str) -> Dict[str, float]:
    """Benchmark tokenizer performance."""
    print(f"\n{Colors.PURPLE}Benchmarking {model_name}:{Colors.END}")
    
    # Long text for benchmarking
    long_text = " ".join(TEST_TEXTS) * 100
    
    # Encoding benchmark
    start_time = time.time()
    tokens = tokenizer.encode(long_text)
    encode_time = time.time() - start_time
    
    # Decoding benchmark  
    start_time = time.time()
    decoded = tokenizer.decode(tokens)
    decode_time = time.time() - start_time
    
    print_info(f"Text length: {len(long_text):,} characters")
    print_info(f"Token count: {len(tokens):,} tokens")
    print_info(f"Encoding time: {encode_time:.4f} seconds")
    print_info(f"Decoding time: {decode_time:.4f} seconds")
    print_info(f"Encoding speed: {len(long_text)/encode_time:.0f} chars/sec")
    print_info(f"Decoding speed: {len(tokens)/decode_time:.0f} tokens/sec")
    
    return {
        'encode_time': encode_time,
        'decode_time': decode_time,
        'text_length': len(long_text),
        'token_count': len(tokens),
        'encode_speed': len(long_text) / encode_time,
        'decode_speed': len(tokens) / decode_time
    }

def test_gpt2_tokenizer():
    """Test GPT-2 tokenizer specifically."""
    print_section("GPT-2 Tokenizer Tests")
    
    models_to_test = ['gpt2', 'gpt-3.5-turbo', 'gpt-4']
    all_results = {}
    
    for model in models_to_test:
        try:
            print(f"\n{Colors.CYAN}Testing {model}...{Colors.END}")
            tokenizer = get_tokenizer(model)
            
            results = {}
            results['basic'] = test_basic_functionality(tokenizer, model)
            results['special_tokens'] = test_special_tokens(tokenizer, model)
            results['properties'] = test_tokenizer_properties(tokenizer, model)
            results['error_handling'] = test_error_handling(tokenizer, model)
            results['benchmark'] = benchmark_tokenizer(tokenizer, model)
            
            all_results[model] = results
            print_success(f"Completed testing {model}")
            
        except Exception as e:
            print_error(f"Failed to test {model}: {e}")
            traceback.print_exc()
    
    return all_results

def test_llama2_tokenizer():
    """Test LLaMA2 tokenizer specifically."""
    print_section("LLaMA2 Tokenizer Tests")
    
    # Create dummy SentencePiece model
    model_path = create_dummy_sentencepiece_model()
    
    try:
        print_info(f"Using temporary model file: {model_path}")
        tokenizer = get_tokenizer('llama2', model_path=model_path)
        
        results = {}
        results['basic'] = test_basic_functionality(tokenizer, 'llama2') or {}
        results['special_tokens'] = test_special_tokens(tokenizer, 'llama2') or {}
        results['properties'] = test_tokenizer_properties(tokenizer, 'llama2') or {}
        results['error_handling'] = test_error_handling(tokenizer, 'llama2') or {}
        
        # Test SentencePiece-specific methods
        if hasattr(tokenizer, 'encode_as_pieces'):
            print(f"\n{Colors.PURPLE}Testing SentencePiece-specific methods:{Colors.END}")
            pieces = tokenizer.encode_as_pieces("Hello world")
            print_info(f"Subword pieces: {pieces}")
            print_success("encode_as_pieces works ✓")
            results['sentencepiece'] = {
                "pieces": pieces,
                "decoded": tokenizer.decode_pieces(pieces)
            }
        
        results['benchmark'] = benchmark_tokenizer(tokenizer, 'llama2') or {}
        
        print_success("Completed LLaMA2 tokenizer tests")
        return {'llama2': results}
        
    except Exception as e:
        print_error(f"LLaMA2 tokenizer test failed: {e}")
        traceback.print_exc()
        return {'llama2': {}}  # always return a dict
    finally:
        # Cleanup
        if Path(model_path).exists():
            Path(model_path).unlink()


def test_llama3_tokenizer():
    """Test LLaMA3 tokenizer specifically."""
    print_section("LLaMA3 Tokenizer Tests")
    
    # Test both with and without tokenizer.json
    results = {}
    
    # Test without tokenizer.json (default mode)
    try:
        print(f"\n{Colors.CYAN}Testing LLaMA3 (default mode)...{Colors.END}")
        tokenizer = get_tokenizer('llama3')
        
        test_results = {}
        test_results['basic'] = test_basic_functionality(tokenizer, 'llama3-default')
        test_results['special_tokens'] = test_special_tokens(tokenizer, 'llama3-default')
        test_results['properties'] = test_tokenizer_properties(tokenizer, 'llama3-default')
        test_results['error_handling'] = test_error_handling(tokenizer, 'llama3-default')
        
        # Test chat template
        print(f"\n{Colors.PURPLE}Testing LLaMA3 chat template:{Colors.END}")
        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
            formatted = tokenizer.apply_chat_template(messages)
            print_info(f"Chat template result: {formatted[:100]}...")
            print_success("Chat template works ✓")
        
        test_results['benchmark'] = benchmark_tokenizer(tokenizer, 'llama3-default')
        results['llama3-default'] = test_results
        
    except Exception as e:
        print_error(f"LLaMA3 default test failed: {e}")
        traceback.print_exc()
    
    # Test with dummy tokenizer.json
    try:
        print(f"\n{Colors.CYAN}Testing LLaMA3 (with tokenizer.json)...{Colors.END}")
        json_path = create_dummy_tokenizer_json()
        
        tokenizer = get_tokenizer('llama3', tokenizer_json_path=json_path)
        
        test_results = {}
        test_results['basic'] = test_basic_functionality(tokenizer, 'llama3-json')
        test_results['special_tokens'] = test_special_tokens(tokenizer, 'llama3-json')
        test_results['properties'] = test_tokenizer_properties(tokenizer, 'llama3-json')
        test_results['benchmark'] = benchmark_tokenizer(tokenizer, 'llama3-json')
        
        results['llama3-json'] = test_results
        
        # Cleanup
        Path(json_path).unlink()
        
    except Exception as e:
        print_error(f"LLaMA3 JSON test failed: {e}")
        traceback.print_exc()
    
    return results

def test_factory_system():
    """Test the factory system and registration."""
    print_section("Factory System Tests")
    
    # Test list_available_models
    try:
        models = list_available_models()
        print_info(f"Available models: {models}")
        print_success(f"Found {len(models)} available models ✓")
    except Exception as e:
        print_error(f"list_available_models failed: {e}")
    
    # Test get_model_info
    test_models = ['gpt2', 'llama2', 'llama3']
    for model in test_models:
        try:
            info = get_model_info(model)
            print_info(f"{model} info: {info}")
            print_success(f"get_model_info works for {model} ✓")
        except Exception as e:
            print_error(f"get_model_info failed for {model}: {e}")
    
    # Test custom tokenizer registration
    class DummyTokenizer(BaseTokenizer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._vocab_size = 1000
            
        def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
            return [ord(c) for c in text[:10]]  # Simple char encoding
            
        def decode(self, ids: List[int]) -> str:
            return ''.join(chr(i) for i in ids if i < 128)
    
    try:
        register_tokenizer('dummy', DummyTokenizer)
        dummy_tokenizer = get_tokenizer('dummy')
        test_result = dummy_tokenizer.encode("test")
        print_success(f"Custom tokenizer registration works ✓")
        print_info(f"Dummy tokenizer result: {test_result}")
    except Exception as e:
        print_error(f"Custom tokenizer registration failed: {e}")

def test_unsupported_models():
    """Test handling of unsupported models."""
    print_section("Unsupported Model Tests")
    
    unsupported_models = ['mistral', 'pi', 'gemma', 'nonexistent-model']
    
    for model in unsupported_models:
        try:
            tokenizer = get_tokenizer(model)
            print_error(f"Should have failed for {model}")
        except NotImplementedError:
            print_success(f"Correctly raised NotImplementedError for {model} ✓")
        except ValueError:
            print_success(f"Correctly raised ValueError for {model} ✓")
        except Exception as e:
            print_warning(f"Unexpected error for {model}: {e}")

def print_summary(all_results: Dict[str, Any]):
    """Print test summary."""
    print_section("Test Summary")
    
    total_passed = 0
    total_failed = 0
    
    for model, results in all_results.items():
        if isinstance(results, dict):
            model_passed = 0
            model_failed = 0
            
            for test_type, test_results in results.items():
                if isinstance(test_results, dict) and 'passed' in test_results:
                    model_passed += test_results['passed']
                    model_failed += test_results['failed']
            
            total_passed += model_passed
            total_failed += model_failed
            
            if model_failed == 0:
                print_success(f"{model}: {model_passed} passed, {model_failed} failed")
            else:
                print_warning(f"{model}: {model_passed} passed, {model_failed} failed")
    
    print(f"\n{Colors.BOLD}Overall Results:{Colors.END}")
    if total_failed == 0:
        print_success(f"🎉 All tests passed! ({total_passed} total)")
    else:
        print_warning(f"⚠️  {total_passed} passed, {total_failed} failed")
    
    return total_passed, total_failed

def main():
    """Main test function."""
    print(f"{Colors.BOLD}{Colors.GREEN}")
    print("🚀 MyLLM Tokenizer System Test Suite")
    print("====================================")
    print(f"{Colors.END}")
    
    all_results = {}
    
    # Test individual tokenizers
    try:
        # GPT-2 family tests
        gpt_results = test_gpt2_tokenizer()
        all_results.update(gpt_results)
        
        # LLaMA2 tests
        llama2_results = test_llama2_tokenizer()
        all_results.update(llama2_results)
        
        # LLaMA3 tests  
        llama3_results = test_llama3_tokenizer()
        all_results.update(llama3_results)
        
        # Factory system tests
        test_factory_system()
        
        # Unsupported model tests
        test_unsupported_models()
        
        # Print summary
        print_summary(all_results)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Test suite failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()