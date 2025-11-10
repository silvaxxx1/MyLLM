from .model_registry import MODEL_REGISTRY
from .weight_mappers import WEIGHT_MAPPERS
from typing import Dict, List
import os
from typing import Optional
from myllm.Configs import ModelConfig
import torch
import gc


class ModelLoader:
    """Unified loader for all model families with your architecture"""
    
    def __init__(self, cache_dir: str = "./models"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def load(
        self,
        model_variant: str,
        device: str = "cuda",
        model_family: Optional[str] = None,
        custom_config: Optional[ModelConfig] = None,
        torch_dtype: Optional[torch.dtype] = None,  # 🆕
        low_cpu_mem_usage: bool = True  # 🆕
    ):
        """
        Load any supported model variant with memory optimizations.
        
        Args:
            model_variant: e.g., "gpt2-small", "llama2-7b"
            device: Target device
            model_family: Optional, auto-detected if not provided
            custom_config: Optional custom config
            torch_dtype: Data type for model (float32, float16, bfloat16)
            low_cpu_mem_usage: Load weights incrementally to save memory
        
        Returns:
            Tuple of (model, config) with loaded weights
        """
        from myllm.model import GPT
        from .download_weight import download_safetensors, load_safetensors, Spinner
        
        # Auto-detect model family
        if model_family is None:
            model_family = self._detect_family(model_variant)
        
        if model_family not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model family: {model_family}")
        
        family = MODEL_REGISTRY[model_family]
        
        if model_variant not in family.variants:
            raise ValueError(
                f"Unknown variant '{model_variant}' for family '{model_family}'. "
                f"Available: {list(family.variants.keys())}"
            )
        
        spec = family.variants[model_variant]
        
        # Get or create config
        if custom_config is None:
            config = ModelConfig.from_name(spec.config_name)
            print(f"📋 Using config: {spec.config_name}")
        else:
            config = custom_config
            print(f"📋 Using custom config")
        
        # 🆕 Estimate memory requirements
        dtype = torch_dtype or torch.float32
        memory_estimate = config.estimate_memory(batch_size=1, dtype=dtype)
        print(f"📊 Estimated memory: {memory_estimate['total_gb']:.2f} GB")
        print(f"   Parameters: {memory_estimate['parameters_gb']:.2f} GB")
        print(f"   Activations: {memory_estimate['activations_gb']:.2f} GB")
        
        # Handle authentication
        headers = None
        if family.requires_auth:
            token = os.getenv(family.token_env_var)
            if not token:
                raise RuntimeError(
                    f"Model requires authentication. Set {family.token_env_var} environment variable."
                )
            headers = {"Authorization": f"Bearer {token}"}
        
        # Download weights
        filename = f"model-{model_variant}.safetensors"
        print(f"⬇️  Downloading {model_variant}...")
        
        filepath = download_safetensors(
            filename,
            self.cache_dir,
            spec.url,
            expected_size=spec.expected_size
        )
        
        # 🆕 Load weights to CPU first if low_cpu_mem_usage
        print(f"📂 Loading weights from disk...")
        if low_cpu_mem_usage:
            params = load_safetensors(filepath, device="cpu")
        else:
            params = load_safetensors(filepath)
        
        # 🆕 Initialize model on CPU first
        print(f"🏗️  Initializing model architecture...")
        model = GPT(config).to("cpu")
        
        # 🆕 Convert model to target dtype before loading weights
        if torch_dtype is not None:
            print(f"🔧 Converting model to {torch_dtype}...")
            model = model.to(dtype=torch_dtype)
        
        # Get appropriate mapper
        mapper_name = spec.weight_mapper or family.default_mapper
        if mapper_name not in WEIGHT_MAPPERS:
            raise ValueError(f"Unknown weight mapper: {mapper_name}")
        
        mapper = WEIGHT_MAPPERS[mapper_name]
        
        # 🆕 Map weights with memory optimization
        print(f"🔄 Mapping weights using {mapper_name}...")
        with Spinner(f"Loading {model_variant} weights"):
            model = mapper.map_weights(
                model, 
                params, 
                config, 
                device,
                low_memory=low_cpu_mem_usage,  # 🆕
                torch_dtype=torch_dtype  # 🆕
            )
        
        # 🆕 Clear memory after loading
        del params
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✅ Successfully loaded {model_variant}!")
        return model, config
    
    def _detect_family(self, model_variant: str) -> str:
        """Auto-detect model family from variant name"""
        for family_name, family in MODEL_REGISTRY.items():
            if model_variant in family.variants:
                return family_name
        raise ValueError(f"Could not detect model family for variant: {model_variant}")
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models grouped by family"""
        return {
            family_name: list(family.variants.keys())
            for family_name, family in MODEL_REGISTRY.items()
        }