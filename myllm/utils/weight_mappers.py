"""
Weight Mapping Utilities for Different Model Architectures

Provides mappers for converting pretrained checkpoint weights from various
sources (HuggingFace, etc.) to the universal model architecture.

Supported Model Families:
- GPT-2
- LLaMA/Llama2/Llama3
- Mistral
- Phi
- Gemma
"""

import torch
import torch.nn as nn 
from typing import Dict, Optional
from abc import ABC, abstractmethod
from tqdm import tqdm


class BaseWeightMapper(ABC):
    """Base class for weight mapping strategies."""
    
    @abstractmethod
    def map_weights(
        self, 
        model: nn.Module, 
        params: dict, 
        config, 
        device: str,
        low_memory: bool = True,
        torch_dtype: torch.dtype = None
    ) -> nn.Module:
        """Map checkpoint weights to model parameters."""
        pass 

    def safe_copy(
        self, 
        name: str, 
        dest, 
        params: dict,
        device: str = "cpu",
        clear_after: bool = True,
        transpose: bool = False
    ):
        """
        Safely copy tensor from checkpoint to model parameter.
        
        Args:
            name: Parameter name in checkpoint
            dest: Destination parameter in model
            params: Checkpoint parameter dictionary
            device: Device for intermediate operations (unused, kept for compatibility)
            clear_after: Whether to clear tensor after copying
            transpose: Whether to transpose 2D tensors
        """
        tensor = params.get(name)
        if tensor is None or dest is None:
            return
        
        # Handle transposition
        if transpose and len(tensor.shape) == 2:
            tensor = tensor.t().contiguous()
        elif len(tensor.shape) == 2 and tensor.shape != dest.shape:
            if tensor.shape == (dest.shape[1], dest.shape[0]):
                tensor = tensor.t().contiguous()
        
        # Match destination device/dtype and copy
        tensor = tensor.to(dest.device, dtype=dest.dtype, non_blocking=True)
        with torch.no_grad():
            dest.data.copy_(tensor)
        
        # Clean up if requested
        if clear_after:
            del tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class GPT2WeightMapperV2(BaseWeightMapper):
    """Weight mapper for GPT-2 style models."""

    def map_weights(
        self, 
        model: nn.Module, 
        params: dict, 
        config, 
        device: str,
        low_memory: bool = True,
        torch_dtype: torch.dtype = None
    ) -> nn.Module:
        """Map HuggingFace GPT-2 weights to model architecture."""
        model.cpu()

        # Token & position embeddings
        self.safe_copy("wte.weight", model.wte.weight, params, clear_after=low_memory)
        if hasattr(model, "wpe") and model.wpe is not None:
            self.safe_copy("wpe.weight", model.wpe.weight, params, clear_after=low_memory)

        num_blocks = self.count_blocks(params, "h.")

        with tqdm(total=num_blocks, desc="Loading GPT-2 blocks", unit="block") as pbar:
            for i in range(num_blocks):
                prefix = f"h.{i}"
                block = model.transformer[f"block_{i}"]

                # QKV weights
                qkv_weight = params.get(f"{prefix}.attn.c_attn.weight")
                if qkv_weight is not None:
                    qkv_weight_t = qkv_weight.t().contiguous()
                    q, k, v = qkv_weight_t.chunk(3, dim=0)
                    qkv_full = torch.cat([q, k, v], dim=0)
                    temp = {f"{prefix}.attn.c_attn.weight.qkv": qkv_full}
                    self.safe_copy(
                        f"{prefix}.attn.c_attn.weight.qkv", 
                        block.attn.qkv.weight, temp, clear_after=low_memory
                    )
                    del temp, qkv_full, q, k, v, qkv_weight_t

                # QKV bias
                qkv_bias = params.get(f"{prefix}.attn.c_attn.bias")
                if qkv_bias is not None:
                    q, k, v = qkv_bias.chunk(3, dim=0)
                    qkv_bias_full = torch.cat([q, k, v], dim=0)
                    temp = {f"{prefix}.attn.c_attn.bias.qkv": qkv_bias_full}
                    self.safe_copy(
                        f"{prefix}.attn.c_attn.bias.qkv", 
                        block.attn.qkv.bias, temp, clear_after=low_memory
                    )
                    del temp, qkv_bias_full, q, k, v

                # Attention projection
                self.safe_copy(
                    f"{prefix}.attn.c_proj.weight", 
                    block.attn.proj.weight, params, 
                    clear_after=low_memory, transpose=True
                )
                self.safe_copy(
                    f"{prefix}.attn.c_proj.bias", 
                    block.attn.proj.bias, params, clear_after=low_memory
                )

                # MLP
                self.safe_copy(
                    f"{prefix}.mlp.c_fc.weight", 
                    block.mlp.fc.weight, params, 
                    clear_after=low_memory, transpose=True
                )
                self.safe_copy(
                    f"{prefix}.mlp.c_fc.bias", 
                    block.mlp.fc.bias, params, clear_after=low_memory
                )
                self.safe_copy(
                    f"{prefix}.mlp.c_proj.weight", 
                    block.mlp.proj.weight, params, 
                    clear_after=low_memory, transpose=True
                )
                self.safe_copy(
                    f"{prefix}.mlp.c_proj.bias", 
                    block.mlp.proj.bias, params, clear_after=low_memory
                )

                # LayerNorm
                self.safe_copy(
                    f"{prefix}.ln_1.weight", 
                    block.norm1.weight, params, clear_after=False
                )
                self.safe_copy(
                    f"{prefix}.ln_1.bias", 
                    block.norm1.bias, params, clear_after=low_memory
                )
                self.safe_copy(
                    f"{prefix}.ln_2.weight", 
                    block.norm2.weight, params, clear_after=False
                )
                self.safe_copy(
                    f"{prefix}.ln_2.bias", 
                    block.norm2.bias, params, clear_after=low_memory
                )

                pbar.update(1)

        # Final LayerNorm
        self.safe_copy("ln_f.weight", model.ln_f.weight, params, clear_after=False)
        self.safe_copy("ln_f.bias", model.ln_f.bias, params, clear_after=low_memory)

        # Weight tying
        if hasattr(model, "lm_head") and getattr(config, 'weight_tying', True):
            with torch.no_grad():
                model.lm_head.weight.data.copy_(model.wte.weight.data)

        # Move to target device
        model.cpu()
        model.to(device=device, dtype=torch_dtype)
        torch.cuda.empty_cache()

        return model

    def count_blocks(self, params: dict, pattern: str) -> int:
        """Count transformer blocks in checkpoint."""
        return len([k for k in params if "h." in k and "attn.c_attn.weight" in k])


class LlamaWeightMapper(BaseWeightMapper):
    """Weight mapper for LLaMA/Llama2/Llama3 style models."""
    
    def map_weights(
        self, 
        model: nn.Module, 
        params: dict, 
        config, 
        device: str,
        low_memory: bool = True,
        torch_dtype: torch.dtype = None
    ) -> nn.Module:
        if not low_memory:
            model.to("cpu")
        
        # Token embeddings
        self.safe_copy(
            "model.embed_tokens.weight", model.wte.weight, 
            params, device="cpu", clear_after=low_memory
        )
        
        num_blocks = self.count_blocks(params, ".self_attn.q_proj.weight")
        
        with tqdm(total=num_blocks, desc="Loading LLaMA blocks", unit="block") as pbar:
            for i in range(num_blocks):
                prefix = f"model.layers.{i}"
                block = model.transformer[f"block_{i}"]
                
                # Attention (separate Q, K, V)
                q_weight = params.get(f"{prefix}.self_attn.q_proj.weight")
                k_weight = params.get(f"{prefix}.self_attn.k_proj.weight")
                v_weight = params.get(f"{prefix}.self_attn.v_proj.weight")
                
                if q_weight is not None and k_weight is not None and v_weight is not None:
                    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                    with torch.no_grad():
                        block.attn.qkv.weight.copy_(
                            qkv_weight.to(block.attn.qkv.weight.device)
                        )
                    
                    if low_memory:
                        del q_weight, k_weight, v_weight, qkv_weight
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Attention projection
                self.safe_copy(
                    f"{prefix}.self_attn.o_proj.weight", 
                    block.attn.proj.weight, params, 
                    device="cpu", clear_after=low_memory
                )
                
                # MLP (SwiGLU)
                self.safe_copy(
                    f"{prefix}.mlp.gate_proj.weight", 
                    block.mlp.fc_1.weight, params, 
                    device="cpu", clear_after=low_memory
                )
                self.safe_copy(
                    f"{prefix}.mlp.up_proj.weight", 
                    block.mlp.fc_2.weight, params, 
                    device="cpu", clear_after=low_memory
                )
                self.safe_copy(
                    f"{prefix}.mlp.down_proj.weight", 
                    block.mlp.proj.weight, params, 
                    device="cpu", clear_after=low_memory
                )
                
                # RMSNorm
                self.safe_copy(
                    f"{prefix}.input_layernorm.weight", 
                    block.norm1.weight, params, 
                    device="cpu", clear_after=low_memory
                )
                self.safe_copy(
                    f"{prefix}.post_attention_layernorm.weight", 
                    block.norm2.weight, params, 
                    device="cpu", clear_after=low_memory
                )
                
                pbar.update(1)
        
        # Final RMSNorm
        self.safe_copy(
            "model.norm.weight", model.ln_f.weight, 
            params, device="cpu", clear_after=low_memory
        )
        
        # LM head
        if hasattr(model, "lm_head"):
            self.safe_copy(
                "lm_head.weight", model.lm_head.weight, 
                params, device="cpu", clear_after=low_memory
            )
        
        if low_memory:
            model.to(device)
        
        return model

    def count_blocks(self, params: dict, pattern: str) -> int:
        """Count transformer blocks in checkpoint."""
        return len([k for k in params if pattern in k])


class MistralWeightMapper(LlamaWeightMapper):
    """Mistral uses same structure as LLaMA."""
    pass


class PhiWeightMapper(BaseWeightMapper):
    """Weight mapper for Microsoft Phi models."""
    
    def map_weights(
        self, 
        model: nn.Module, 
        params: dict, 
        config, 
        device: str,
        low_memory: bool = True,
        torch_dtype: torch.dtype = None
    ) -> nn.Module:
        if not low_memory:
            model.to("cpu")
        
        self.safe_copy(
            "transformer.embd.wte.weight", model.wte.weight, 
            params, device="cpu", clear_after=low_memory
        )
        
        num_blocks = self.count_blocks(params, ".mixer.Wqkv.weight")
        
        with tqdm(total=num_blocks, desc="Loading Phi blocks", unit="block") as pbar:
            for i in range(num_blocks):
                prefix = f"transformer.h.{i}"
                block = model.transformer[f"block_{i}"]
                
                # Combined Wqkv
                self.safe_copy(
                    f"{prefix}.mixer.Wqkv.weight", 
                    block.attn.qkv.weight, params, 
                    device="cpu", clear_after=low_memory
                )
                self.safe_copy(
                    f"{prefix}.mixer.Wqkv.bias", 
                    block.attn.qkv.bias, params, 
                    device="cpu", clear_after=low_memory
                )
                self.safe_copy(
                    f"{prefix}.mixer.out_proj.weight", 
                    block.attn.proj.weight, params, 
                    device="cpu", clear_after=low_memory
                )
                self.safe_copy(
                    f"{prefix}.mixer.out_proj.bias", 
                    block.attn.proj.bias, params, 
                    device="cpu", clear_after=low_memory
                )
                
                # MLP
                self.safe_copy(
                    f"{prefix}.mlp.fc1.weight", 
                    block.mlp.fc.weight, params, 
                    device="cpu", clear_after=low_memory
                )
                self.safe_copy(
                    f"{prefix}.mlp.fc1.bias", 
                    block.mlp.fc.bias, params, 
                    device="cpu", clear_after=low_memory
                )
                self.safe_copy(
                    f"{prefix}.mlp.fc2.weight", 
                    block.mlp.proj.weight, params, 
                    device="cpu", clear_after=low_memory
                )
                self.safe_copy(
                    f"{prefix}.mlp.fc2.bias", 
                    block.mlp.proj.bias, params, 
                    device="cpu", clear_after=low_memory
                )
                
                # LayerNorm
                self.safe_copy(
                    f"{prefix}.ln.weight", 
                    block.norm1.weight, params, 
                    device="cpu", clear_after=low_memory
                )
                self.safe_copy(
                    f"{prefix}.ln.bias", 
                    block.norm1.bias, params, 
                    device="cpu", clear_after=low_memory
                )
                
                pbar.update(1)
        
        self.safe_copy(
            "transformer.ln_f.weight", model.ln_f.weight, 
            params, device="cpu", clear_after=low_memory
        )
        self.safe_copy(
            "transformer.ln_f.bias", model.ln_f.bias, 
            params, device="cpu", clear_after=low_memory
        )
        self.safe_copy(
            "lm_head.linear.weight", model.lm_head.weight, 
            params, device="cpu", clear_after=low_memory
        )
        self.safe_copy(
            "lm_head.linear.bias", model.lm_head.bias, 
            params, device="cpu", clear_after=low_memory
        )
        
        if low_memory:
            model.to(device)
        
        return model

    def count_blocks(self, params: dict, pattern: str) -> int:
        """Count transformer blocks in checkpoint."""
        return len([k for k in params if pattern in k])


class GemmaWeightMapper(BaseWeightMapper):
    """Weight mapper for Google Gemma models."""
    
    def map_weights(
        self, 
        model: nn.Module, 
        params: dict, 
        config, 
        device: str,
        low_memory: bool = True,
        torch_dtype: torch.dtype = None
    ) -> nn.Module:
        if not low_memory:
            model.to("cpu")
        
        self.safe_copy(
            "model.embed_tokens.weight", model.wte.weight, 
            params, device="cpu", clear_after=low_memory
        )
        
        num_blocks = self.count_blocks(params, ".self_attn.q_proj.weight")
        
        with tqdm(total=num_blocks, desc="Loading Gemma blocks", unit="block") as pbar:
            for i in range(num_blocks):
                prefix = f"model.layers.{i}"
                block = model.transformer[f"block_{i}"]
                
                # Attention
                q_weight = params.get(f"{prefix}.self_attn.q_proj.weight")
                k_weight = params.get(f"{prefix}.self_attn.k_proj.weight")
                v_weight = params.get(f"{prefix}.self_attn.v_proj.weight")
                
                if q_weight is not None and k_weight is not None and v_weight is not None:
                    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                    with torch.no_grad():
                        block.attn.qkv.weight.copy_(
                            qkv_weight.to(block.attn.qkv.weight.device)
                        )
                    
                    if low_memory:
                        del q_weight, k_weight, v_weight, qkv_weight
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                self.safe_copy(
                    f"{prefix}.self_attn.o_proj.weight", 
                    block.attn.proj.weight, params, 
                    device="cpu", clear_after=low_memory
                )
                
                # MLP (GELU instead of SiLU)
                self.safe_copy(
                    f"{prefix}.mlp.gate_proj.weight", 
                    block.mlp.fc_1.weight, params, 
                    device="cpu", clear_after=low_memory
                )
                self.safe_copy(
                    f"{prefix}.mlp.up_proj.weight", 
                    block.mlp.fc_2.weight, params, 
                    device="cpu", clear_after=low_memory
                )
                self.safe_copy(
                    f"{prefix}.mlp.down_proj.weight", 
                    block.mlp.proj.weight, params, 
                    device="cpu", clear_after=low_memory
                )
                
                # RMSNorm
                self.safe_copy(
                    f"{prefix}.input_layernorm.weight", 
                    block.norm1.weight, params, 
                    device="cpu", clear_after=low_memory
                )
                self.safe_copy(
                    f"{prefix}.post_attention_layernorm.weight", 
                    block.norm2.weight, params, 
                    device="cpu", clear_after=low_memory
                )
                
                pbar.update(1)
        
        self.safe_copy(
            "model.norm.weight", model.ln_f.weight, 
            params, device="cpu", clear_after=low_memory
        )
        
        if hasattr(model, "lm_head"):
            self.safe_copy(
                "lm_head.weight", model.lm_head.weight, 
                params, device="cpu", clear_after=low_memory
            )
        
        if low_memory:
            model.to(device)
        
        return model

    def count_blocks(self, params: dict, pattern: str) -> int:
        """Count transformer blocks in checkpoint."""
        return len([k for k in params if pattern in k])


# Weight mapper registry
WEIGHT_MAPPERS: Dict[str, BaseWeightMapper] = {
    "gpt2_mapper": GPT2WeightMapperV2(),
    "llama_mapper": LlamaWeightMapper(),
    "mistral_mapper": MistralWeightMapper(),
    "phi_mapper": PhiWeightMapper(),
    "gemma_mapper": GemmaWeightMapper(),
}