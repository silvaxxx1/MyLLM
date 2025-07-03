#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Tuple
import json
import os
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import argparse
import yaml
from dataclasses import dataclass, asdict
import time
from tqdm import tqdm
import warnings
import sys
from enum import Enum

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelSource(Enum):
    """Enum for model source types"""
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    SCRATCH = "scratch"

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
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

class LLMAutomator:
    """
    Automated LLM wrapper with enhanced features:
    - Configuration management
    - Error handling
    - Progress tracking
    - Model lifecycle management
    """
    
    def __init__(self, config: Union[Dict, str, Path]):
        """
        Initialize the LLM automator
        
        Args:
            config: Can be:
                - Path to config file (JSON/YAML)
                - Dictionary with configuration
                - Path to directory with config.json
        """
        self.config = self._load_config(config)
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.model_source = None
        self.initialized = False
        self.loaded = False
        
        logger.info(f"LLM Automator initialized on device: {self.device}")
    
    def _load_config(self, config_input: Union[Dict, str, Path]) -> Dict:
        """Load configuration from various sources"""
        if isinstance(config_input, dict):
            return config_input
        
        config_path = Path(config_input)
        
        # Handle directory case
        if config_path.is_dir():
            config_path = config_path / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"No config.json found in {config_path.parent}")
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load based on file extension
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                return json.load(f)
        elif config_path.suffix in ('.yaml', '.yml'):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def _get_device(self) -> torch.device:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def setup_model(self, 
                   model_source: ModelSource = ModelSource.HUGGINGFACE,
                   model_identifier: Optional[str] = None,
                   init_pretrained: bool = False) -> None:
        """
        Setup the model based on source
        
        Args:
            model_source: Where to get the model from
            model_identifier: Name/path for the model
            init_pretrained: For scratch init, whether to use pretrained-like init
        """
        try:
            if model_source == ModelSource.HUGGINGFACE:
                if not model_identifier:
                    model_identifier = self.config.get('model_name', 'gpt2')
                self._setup_huggingface_model(model_identifier)
            elif model_source == ModelSource.LOCAL:
                if not model_identifier:
                    model_identifier = self.config.get('model_path')
                self._setup_local_model(model_identifier)
            elif model_source == ModelSource.SCRATCH:
                self._setup_scratch_model(init_pretrained)
            else:
                raise ValueError(f"Unknown model source: {model_source}")
            
            self.model_source = model_source
            logger.info(f"Model setup complete from {model_source.value}")
            
        except Exception as e:
            logger.error(f"Failed to setup model: {str(e)}")
            raise
    
    def _setup_huggingface_model(self, model_name: str) -> None:
        """Setup model from HuggingFace Hub"""
        logger.info(f"Loading HuggingFace model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate settings
        torch_dtype = torch.float16 if 'cuda' in str(self.device) else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).to(self.device)
        
        self.loaded = True
    
    def _setup_local_model(self, model_path: str) -> None:
        """Setup model from local path"""
        logger.info(f"Loading local model from: {model_path}")
        
        model_path = Path(model_path)
        
        # First try to load as complete HuggingFace model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if 'cuda' in str(self.device) else torch.float32
            ).to(self.device)
            self.loaded = True
            return
        except Exception as e:
            logger.warning(f"Failed to load as HF model: {str(e)}")
        
        # Fall back to custom loading
        if (model_path / "config.json").exists():
            with open(model_path / "config.json", 'r') as f:
                model_config = json.load(f)
            
            # Update our config with model-specific settings
            self.config.update(model_config)
        
        # Initialize model architecture
        from model import GPT  # Import here to avoid circular imports
        self.model = GPT(self.config).to(self.device)
        
        # Load weights
        model_file = self._find_model_file(model_path)
        if model_file.suffix == '.safetensors':
            from safetensors.torch import load_file
            state_dict = load_file(model_file)
        else:
            state_dict = torch.load(model_file, map_location='cpu')
        
        self.model.load_state_dict(state_dict)
        self.loaded = True
        
        # Try to load tokenizer
        if (model_path / "tokenizer.json").exists():
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {str(e)}")
    
    def _find_model_file(self, model_path: Path) -> Path:
        """Find the model weights file in a directory"""
        possible_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "model.bin",
            "weights.bin"
        ]
        
        for file in possible_files:
            if (model_path / file).exists():
                return model_path / file
        
        raise FileNotFoundError(f"No model weights file found in {model_path}")
    
    def _setup_scratch_model(self, init_pretrained: bool) -> None:
        """Initialize model from scratch"""
        logger.info("Initializing new model from scratch")
        
        from model import GPT  # Import here to avoid circular imports
        self.model = GPT(self.config).to(self.device)
        
        # Initialize weights
        if init_pretrained:
            self._init_pretrained_weights()
        else:
            self._init_random_weights()
        
        self.initialized = True
        
        # Initialize tokenizer if specified
        if 'tokenizer_name' in self.config:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer_name'])
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {str(e)}")
    
    def generate_text(self,
                     prompt: str,
                     generation_config: Optional[Union[GenerationConfig, Dict]] = None,
                     return_dict: bool = False) -> Union[str, Dict]:
        """
        Generate text from prompt with given configuration
        
        Args:
            prompt: Input text prompt
            generation_config: Configuration for generation
            return_dict: Whether to return full generation info
            
        Returns:
            Generated text or full generation info
        """
        if not self._validate_generation_readiness():
            raise RuntimeError("Model not ready for generation")
        
        # Process generation config
        if generation_config is None:
            gen_config = GenerationConfig()
        elif isinstance(generation_config, dict):
            gen_config = GenerationConfig(**generation_config)
        else:
            gen_config = generation_config
        
        # Prepare inputs
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get('max_length', 1024)
        ).to(self.device)
        
        # Calculate max_new_tokens if not provided
        if gen_config.max_new_tokens is None:
            gen_config.max_new_tokens = gen_config.max_length - inputs.input_ids.shape[1]
        
        # Generate with progress tracking
        start_time = time.time()
        
        with torch.no_grad(), tqdm(total=gen_config.max_new_tokens, desc="Generating") as pbar:
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
                repetition_penalty=gen_config.repetition_penalty,
                length_penalty=gen_config.length_penalty,
                early_stopping=gen_config.early_stopping,
                return_dict_in_generate=return_dict
            )
            pbar.update(gen_config.max_new_tokens)
        
        # Process outputs
        if return_dict and hasattr(outputs, 'sequences'):
            sequences = outputs.sequences
        else:
            sequences = outputs if isinstance(outputs, torch.Tensor) else outputs.sequences
        
        # Remove input tokens from output
        new_tokens = sequences[:, inputs.input_ids.shape[1]:] if len(sequences.shape) > 1 else sequences[inputs.input_ids.shape[1]:]
        
        # Decode to text
        generated_texts = self.tokenizer.batch_decode(
            new_tokens, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Prepare return value
        result = generated_texts[0] if gen_config.num_return_sequences == 1 else generated_texts
        
        if return_dict:
            return {
                'generated_text': result,
                'sequences': sequences,
                'scores': getattr(outputs, 'scores', None),
                'input_ids': inputs.input_ids,
                'time_elapsed': time.time() - start_time,
                'config': asdict(gen_config)
            }
        
        return result
    
    def _validate_generation_readiness(self) -> bool:
        """Check if model is ready for generation"""
        if not (self.loaded or self.initialized):
            logger.error("Model not initialized or loaded")
            return False
        
        if self.tokenizer is None:
            logger.error("Tokenizer not available")
            return False
        
        return True
    
    def save_model(self, 
                  save_path: str,
                  save_tokenizer: bool = True,
                  save_format: str = "pytorch") -> None:
        """
        Save model to disk
        
        Args:
            save_path: Directory to save to
            save_tokenizer: Whether to save tokenizer
            save_format: Format to save ('pytorch' or 'safetensors')
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {save_path}")
        
        # Save config
        with open(save_path / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save model
        if save_format == "safetensors":
            try:
                from safetensors.torch import save_file
                save_file(self.model.state_dict(), save_path / "model.safetensors")
            except ImportError:
                logger.warning("safetensors not available, falling back to pytorch format")
                torch.save(self.model.state_dict(), save_path / "pytorch_model.bin")
        else:
            torch.save(self.model.state_dict(), save_path / "pytorch_model.bin")
        
        # Save tokenizer
        if save_tokenizer and self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
    
    def _init_pretrained_weights(self) -> None:
        """Initialize weights similar to pretrained models"""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _init_random_weights(self) -> None:
        """Initialize weights randomly"""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.xavier_uniform_(module.weight)
    
    def __repr__(self) -> str:
        return (f"LLMAutomator(device={self.device}, "
                f"model_source={self.model_source.value if self.model_source else 'none'}, "
                f"status={'loaded' if self.loaded else 'initialized' if self.initialized else 'empty'})")

def main():
    """Command line interface for the LLM Automator"""
    parser = argparse.ArgumentParser(description="LLM Model Automator")
    
    # Model configuration
    parser.add_argument('--config', type=str, help="Path to config file")
    parser.add_argument('--model-source', type=str, choices=['huggingface', 'local', 'scratch'],
                       default='huggingface', help="Where to load model from")
    parser.add_argument('--model-identifier', type=str, 
                       help="Model name (HF) or path (local) for loading")
    
    # Generation options
    parser.add_argument('--prompt', type=str, help="Text prompt for generation")
    parser.add_argument('--max-length', type=int, default=100, help="Maximum length of generated text")
    parser.add_argument('--temperature', type=float, default=1.0, help="Sampling temperature")
    parser.add_argument('--top-p', type=float, default=0.9, help="Top-p sampling value")
    parser.add_argument('--top-k', type=int, default=50, help="Top-k sampling value")
    
    # Output options
    parser.add_argument('--output', type=str, help="File to save output to")
    parser.add_argument('--save-model', type=str, help="Directory to save model to")
    parser.add_argument('--save-format', choices=['pytorch', 'safetensors'], 
                       default='pytorch', help="Format to save model in")
    
    args = parser.parse_args()
    
    try:
        # Initialize automator
        automator = LLMAutomator(args.config or {})
        
        # Setup model
        automator.setup_model(
            model_source=ModelSource(args.model_source),
            model_identifier=args.model_identifier
        )
        
        # Generate text if prompt provided
        if args.prompt:
            gen_config = {
                'max_length': args.max_length,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'top_k': args.top_k
            }
            
            output = automator.generate_text(args.prompt, gen_config)
            
            # Handle output
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output)
                logger.info(f"Output saved to {args.output}")
            else:
                print("\nGenerated Text:")
                print("-" * 50)
                print(output)
                print("-" * 50)
        
        # Save model if requested
        if args.save_model:
            automator.save_model(args.save_model, save_format=args.save_format)
            logger.info(f"Model saved to {args.save_model}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
    # usage example in python script (no cli)
    from config import Config
    config = Config.from_name("gpt2-small")
    automator = LLMAutomator(config)
    automator.setup_model(model_source=ModelSource.LOCAL, model_identifier="/path/to/local/model")
    output = automator.generate_text("Hello, world!", {"max_length": 50, "temperature": 0.8, "top_p": 0.9})
