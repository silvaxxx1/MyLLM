# Enhanced version combining the best of both approaches

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, Optional, Union, List
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelSource(Enum):
    HUGGINGFACE = "huggingface"
    LOCAL = "local" 
    SCRATCH = "scratch"

@dataclass
class GenerationConfig:
    """Enhanced generation configuration"""
    max_length: int = 100
    max_new_tokens: Optional[int] = None
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = False
    use_cache: bool = True  # Added from first implementation

class EnhancedLLMAutomator:
    """
    Best-of-both-worlds LLM wrapper combining:
    - LLMAutomator's architecture and UX
    - First implementation's advanced features
    """
    
    def __init__(self, config: Union[Dict, str, Path]):
        self.config = self._load_config(config)
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.model_source = None
        self.initialized = False
        self.loaded = False
        self._generation_stats = {}
        
        logger.info(f"Enhanced LLM Automator initialized on {self.device}")
    
    def _load_config(self, config_input: Union[Dict, str, Path]) -> Dict:
        """Load configuration from various sources"""
        if isinstance(config_input, dict):
            return config_input
        
        config_path = Path(config_input)
        if config_path.is_dir():
            config_path = config_path / "config.json"
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _get_device(self) -> torch.device:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    
    def setup_model(self, 
                   model_source: ModelSource = ModelSource.HUGGINGFACE,
                   model_identifier: Optional[str] = None,
                   trust_remote_code: bool = True) -> 'EnhancedLLMAutomator':
        """Setup model with enhanced error handling"""
        try:
            if model_source == ModelSource.HUGGINGFACE:
                self._setup_huggingface_model(
                    model_identifier or self.config.get('model_name', 'gpt2'),
                    trust_remote_code
                )
            elif model_source == ModelSource.LOCAL:
                self._setup_local_model(model_identifier or self.config.get('model_path'))
            elif model_source == ModelSource.SCRATCH:
                self._setup_scratch_model()
            
            self.model_source = model_source
            logger.info(f"✅ Model ready: {self._count_parameters():,} parameters")
            return self
            
        except Exception as e:
            logger.error(f"❌ Model setup failed: {str(e)}")
            raise
    
    def _setup_huggingface_model(self, model_name: str, trust_remote_code: bool) -> None:
        """Enhanced HuggingFace model loading with fallback"""
        logger.info(f"🔄 Loading HuggingFace model: {model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimal settings
            torch_dtype = torch.float16 if 'cuda' in str(self.device) else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            if not torch.cuda.is_available() or not hasattr(self.model, 'hf_device_map'):
                self.model = self.model.to(self.device)
            
            self.loaded = True
            
        except Exception as e:
            logger.warning(f"Direct HF loading failed: {e}")
            # Fallback to custom architecture + weight transfer
            self._setup_custom_with_hf_weights(model_name)
    
    def _setup_custom_with_hf_weights(self, model_name: str) -> None:
        """Fallback: Load HF weights into custom architecture"""
        logger.info("🔄 Loading HF weights into custom architecture...")
        
        # Initialize custom model
        from model import GPT  # Your custom model
        self.model = GPT(self.config).to(self.device)
        
        # Load HF model for weight extraction
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Transfer weights (enhanced from first implementation)
        self._transfer_weights(hf_model.state_dict(), self.model.state_dict())
        self.loaded = True
    
    def _setup_local_model(self, model_path: str) -> None:
        """Enhanced local model loading"""
        logger.info(f"🔄 Loading local model: {model_path}")
        model_path = Path(model_path)
        
        # Try HuggingFace format first
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
            self.loaded = True
            return
        except Exception:
            logger.info("Not HF format, trying custom loading...")
        
        # Custom format loading
        from model import GPT
        self.model = GPT(self.config).to(self.device)
        
        # Load weights
        model_file = self._find_model_file(model_path)
        if model_file.suffix == '.safetensors':
            from safetensors.torch import load_file
            state_dict = load_file(model_file)
        else:
            state_dict = torch.load(model_file, map_location='cpu')
        
        self.model.load_state_dict(state_dict, strict=False)
        self.loaded = True
    
    def _setup_scratch_model(self) -> None:
        """Initialize model from scratch"""
        logger.info("🔄 Initializing model from scratch...")
        
        from model import GPT
        self.model = GPT(self.config).to(self.device)
        self._init_weights()
        
        # Load tokenizer if specified
        if 'tokenizer_name' in self.config:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer_name'])
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.initialized = True
    
    def generate_text(self,
                     prompt: Union[str, List[str]],
                     generation_config: Optional[Union[GenerationConfig, Dict]] = None,
                     return_dict: bool = False,
                     show_progress: bool = True) -> Union[str, List[str], Dict]:
        """
        Enhanced text generation with progress tracking and stats
        """
        if not self._validate_readiness():
            raise RuntimeError("Model not ready for generation")
        
        # Process config
        if generation_config is None:
            gen_config = GenerationConfig()
        elif isinstance(generation_config, dict):
            gen_config = GenerationConfig(**generation_config)
        else:
            gen_config = generation_config
        
        # Handle batch prompts
        prompts = [prompt] if isinstance(prompt, str) else prompt
        
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get('max_length', 1024)
        ).to(self.device)
        
        # Calculate max_new_tokens
        if gen_config.max_new_tokens is None:
            gen_config.max_new_tokens = gen_config.max_length - inputs.input_ids.shape[1]
        
        # Generate with progress
        start_time = time.time()
        progress_bar = tqdm(total=gen_config.max_new_tokens, desc="Generating", disable=not show_progress)
        
        try:
            with torch.no_grad():
                if hasattr(self.model, 'generate'):
                    # Use HuggingFace generate
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=gen_config.max_new_tokens,
                        temperature=gen_config.temperature,
                        top_k=gen_config.top_k,
                        top_p=gen_config.top_p,
                        do_sample=gen_config.do_sample,
                        num_return_sequences=gen_config.num_return_sequences,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=gen_config.use_cache,
                        repetition_penalty=gen_config.repetition_penalty,
                        length_penalty=gen_config.length_penalty,
                        early_stopping=gen_config.early_stopping,
                        return_dict_in_generate=return_dict
                    )
                else:
                    # Use custom generation
                    outputs = self._custom_generate(inputs.input_ids, gen_config)
                
                progress_bar.update(gen_config.max_new_tokens)
        finally:
            progress_bar.close()
        
        # Process outputs
        if return_dict and hasattr(outputs, 'sequences'):
            sequences = outputs.sequences
        else:
            sequences = outputs if isinstance(outputs, torch.Tensor) else outputs
        
        # Decode
        new_tokens = sequences[:, inputs.input_ids.shape[1]:]
        generated_texts = self.tokenizer.batch_decode(
            new_tokens, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Update stats
        generation_time = time.time() - start_time
        self._generation_stats = {
            'last_generation_time': generation_time,
            'tokens_per_second': gen_config.max_new_tokens / generation_time,
            'total_generations': self._generation_stats.get('total_generations', 0) + 1
        }
        
        # Format output
        if isinstance(prompt, str) and gen_config.num_return_sequences == 1:
            result = generated_texts[0]
        else:
            result = generated_texts
        
        if return_dict:
            return {
                'generated_text': result,
                'sequences': sequences,
                'generation_time': generation_time,
                'tokens_per_second': self._generation_stats['tokens_per_second'],
                'config': asdict(gen_config)
            }
        
        return result
    
    def _custom_generate(self, input_ids: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        """Custom generation for non-HF models (from first implementation)"""
        # Implementation from first LLM class's _custom_generate method
        # ... (detailed implementation would go here)
        pass
    
    def _transfer_weights(self, source_dict: Dict, target_dict: Dict) -> None:
        """Enhanced weight transfer with better mapping"""
        logger.info("🔄 Transferring weights...")
        
        # Create parameter mapping (enhanced from first implementation)
        mapping = self._create_param_mapping(source_dict, target_dict)
        
        transferred = 0
        for target_name, source_name in mapping.items():
            if source_name in source_dict and target_name in target_dict:
                source_param = source_dict[source_name]
                target_param = target_dict[target_name]
                
                if source_param.shape == target_param.shape:
                    target_dict[target_name].copy_(source_param)
                    transferred += 1
                    logger.debug(f"✅ {source_name} -> {target_name}")
                else:
                    logger.warning(f"❌ Shape mismatch: {source_name} {source_param.shape} -> {target_name} {target_param.shape}")
        
        logger.info(f"✅ Transferred {transferred} parameters")
        self.model.load_state_dict(target_dict)
    
    def _create_param_mapping(self, source_dict: Dict, target_dict: Dict) -> Dict:
        """Create parameter name mapping between models"""
        mapping = {}
        
        for target_name in target_dict.keys():
            # Direct mapping first
            if target_name in source_dict:
                mapping[target_name] = target_name
                continue
            
            # Common transformations
            source_name = target_name
            source_name = source_name.replace('transformer.block_', 'transformer.h.')
            source_name = source_name.replace('.norm1.', '.ln_1.')
            source_name = source_name.replace('.norm2.', '.ln_2.')
            source_name = source_name.replace('.attn.qkv.', '.attn.c_attn.')
            source_name = source_name.replace('.attn.proj.', '.attn.c_proj.')
            source_name = source_name.replace('.mlp.fc.', '.mlp.c_fc.')
            source_name = source_name.replace('.mlp.proj.', '.mlp.c_proj.')
            
            if source_name in source_dict:
                mapping[target_name] = source_name
        
        return mapping
    
    def _find_model_file(self, model_path: Path) -> Path:
        """Find model weights file"""
        for file in ["pytorch_model.bin", "model.safetensors", "model.bin"]:
            if (model_path / file).exists():
                return model_path / file
        raise FileNotFoundError(f"No model file found in {model_path}")
    
    def _init_weights(self) -> None:
        """Initialize model weights"""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _validate_readiness(self) -> bool:
        """Validate model is ready for generation"""
        if not (self.loaded or self.initialized):
            logger.error("❌ Model not loaded or initialized")
            return False
        if self.tokenizer is None:
            logger.error("❌ Tokenizer not available")
            return False
        return True
    
    def _count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        stats = {
            'device': str(self.device),
            'model_source': self.model_source.value if self.model_source else None,
            'parameters': self._count_parameters() if self.model else 0,
            'status': 'loaded' if self.loaded else 'initialized' if self.initialized else 'empty'
        }
        stats.update(self._generation_stats)
        
        if torch.cuda.is_available():
            stats['gpu_memory'] = {
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'reserved_gb': torch.cuda.memory_reserved() / 1e9
            }
        
        return stats
    
    def save_model(self, save_path: str, save_format: str = "pytorch") -> None:
        """Save model with multiple format support"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving model to {save_path}")
        
        # Save config
        with open(save_path / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save model weights
        if save_format == "safetensors":
            try:
                from safetensors.torch import save_file
                save_file(self.model.state_dict(), save_path / "model.safetensors")
            except ImportError:
                logger.warning("safetensors not available, using pytorch format")
                torch.save(self.model.state_dict(), save_path / "pytorch_model.bin")
        else:
            torch.save(self.model.state_dict(), save_path / "pytorch_model.bin")
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
        
        logger.info("✅ Model saved successfully")
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"EnhancedLLMAutomator(device={stats['device']}, "
                f"source={stats['model_source']}, "
                f"params={stats['parameters']:,}, "
                f"status={stats['status']})")

# Usage example
if __name__ == "__main__":
    # Simple usage
    automator = EnhancedLLMAutomator({})
    automator.setup_model(ModelSource.HUGGINGFACE, "gpt2")
    
    output = automator.generate_text(
        "The future of AI is",
        {"max_new_tokens": 50, "temperature": 0.8}
    )
    
    print(output)
    print(automator.get_stats())