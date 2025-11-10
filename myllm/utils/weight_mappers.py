import torch
import torch.nn as nn
from typing import Dict
from abc import ABC, abstractmethod
from tqdm import tqdm

# ============================================
# weight_mappers.py - Architecture-Specific Mappers
# ============================================

class BaseWeightMapper(ABC):
    """Base class for weight mapping strategies"""

    @abstractmethod
    def map_weights(
        self, 
        model: nn.Module, 
        params: dict, 
        config, 
        device: str,
        low_memory: bool = True,       # 🆕 Move to low memory mode
        torch_dtype: torch.dtype = None  # 🆕 Optional dtype cast
    ) -> nn.Module:
        """Map checkpoint weights to model parameters"""
        pass

    def safe_copy(
        self, 
        name: str, 
        dest, 
        params: dict,
        device: str = "cpu",  # 🆕 device to copy to
        clear_after: bool = True  # 🆕 release memory after copy
    ):
        """Safe tensor copy with optional memory cleanup"""
        tensor = params.get(name)
        if tensor is not None and dest is not None:
            if tensor.shape[::-1] == dest.shape:
                tensor = tensor.T
            tensor = tensor.to(device, non_blocking=True)
            with torch.no_grad():
                dest.copy_(tensor)
            if clear_after:
                del tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def count_blocks(self, params: dict, pattern: str) -> int:
        """Count transformer blocks in checkpoint"""
        return len([k for k in params if pattern in k])


# ============================================
# GPT-2 Weight Mapper
# ============================================

class GPT2WeightMapper(BaseWeightMapper):
    """Weight mapper for GPT-2 style models"""

    def map_weights(
        self, 
        model: nn.Module, 
        params: dict, 
        config, 
        device: str,
        low_memory: bool = True,
        torch_dtype: torch.dtype = None
    ) -> nn.Module:
        model.to("cpu")

        # Token embeddings
        self.safe_copy("wte.weight", model.wte.weight, params, device="cpu")
        if hasattr(model, "wpe"):
            self.safe_copy("wpe.weight", model.wpe.weight, params, device="cpu")

        # Count transformer blocks
        num_blocks = self.count_blocks(params, ".attn.c_attn.weight")
        print(f"✅ Detected {num_blocks} transformer blocks")

        with tqdm(total=num_blocks, desc="Loading GPT-2 blocks", ascii=True) as pbar:
            for i in range(num_blocks):
                prefix = f"h.{i}"
                block = model.transformer[f"block_{i}"]

                # Attention
                self.safe_copy(f"{prefix}.attn.c_attn.weight", block.attn.qkv.weight, params)
                self.safe_copy(f"{prefix}.attn.c_attn.bias", block.attn.qkv.bias, params)
                self.safe_copy(f"{prefix}.attn.c_proj.weight", block.attn.proj.weight, params)
                self.safe_copy(f"{prefix}.attn.c_proj.bias", block.attn.proj.bias, params)

                # MLP
                self.safe_copy(f"{prefix}.mlp.c_fc.weight", block.mlp.fc.weight, params)
                self.safe_copy(f"{prefix}.mlp.c_fc.bias", block.mlp.fc.bias, params)
                self.safe_copy(f"{prefix}.mlp.c_proj.weight", block.mlp.proj.weight, params)
                self.safe_copy(f"{prefix}.mlp.c_proj.bias", block.mlp.proj.bias, params)

                # LayerNorm
                self.safe_copy(f"{prefix}.ln_1.weight", block.norm1.weight, params)
                self.safe_copy(f"{prefix}.ln_1.bias", block.norm1.bias, params)
                self.safe_copy(f"{prefix}.ln_2.weight", block.norm2.weight, params)
                self.safe_copy(f"{prefix}.ln_2.bias", block.norm2.bias, params)

                pbar.update(1)

        # Final LayerNorm
        self.safe_copy("ln_f.weight", model.ln_f.weight, params)
        self.safe_copy("ln_f.bias", model.ln_f.bias, params)

        # LM head or weight tying
        if hasattr(model, "lm_head"):
            if "lm_head.weight" in params:
                self.safe_copy("lm_head.weight", model.lm_head.weight, params)
            else:
                model.lm_head.weight = model.wte.weight

        # Move model to final device
        if torch_dtype:
            model.to(device=device, dtype=torch_dtype)
        else:
            model.to(device=device)

        return model


# ============================================
# LLaMA Weight Mapper
# ============================================

class LlamaWeightMapper(BaseWeightMapper):
    """Weight mapper for LLaMA/Llama2/Llama3 style models"""

    def map_weights(
        self, 
        model: nn.Module, 
        params: dict, 
        config, 
        device: str,
        low_memory: bool = True,
        torch_dtype: torch.dtype = None
    ) -> nn.Module:
        model.to("cpu")

        # Token embeddings
        self.safe_copy("model.embed_tokens.weight", model.wte.weight, params, device="cpu")

        # Count blocks
        num_blocks = self.count_blocks(params, ".self_attn.q_proj.weight")
        print(f"✅ Detected {num_blocks} transformer blocks")

        with tqdm(total=num_blocks, desc="Loading LLaMA blocks", ascii=True) as pbar:
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
                        block.attn.qkv.weight.copy_(qkv_weight)

                self.safe_copy(f"{prefix}.self_attn.o_proj.weight", block.attn.proj.weight, params)

                # MLP
                self.safe_copy(f"{prefix}.mlp.gate_proj.weight", block.mlp.fc_1.weight, params)
                self.safe_copy(f"{prefix}.mlp.up_proj.weight", block.mlp.fc_2.weight, params)
                self.safe_copy(f"{prefix}.mlp.down_proj.weight", block.mlp.proj.weight, params)

                # RMSNorm
                self.safe_copy(f"{prefix}.input_layernorm.weight", block.norm1.weight, params)
                self.safe_copy(f"{prefix}.post_attention_layernorm.weight", block.norm2.weight, params)

                pbar.update(1)

        # Final RMSNorm
        self.safe_copy("model.norm.weight", model.ln_f.weight, params)

        # LM head
        if hasattr(model, "lm_head"):
            self.safe_copy("lm_head.weight", model.lm_head.weight, params)

        # Move model to final device
        if torch_dtype:
            model.to(device=device, dtype=torch_dtype)
        else:
            model.to(device=device)

        return model


# ============================================
# Other Mappers (Mistral, Phi, Gemma)
# ============================================

class MistralWeightMapper(LlamaWeightMapper):
    """Mistral uses same structure as LLaMA"""
    pass


class PhiWeightMapper(BaseWeightMapper):
    """Weight mapper for Microsoft Phi models"""
    
    def map_weights(self, model: nn.Module, params: dict, config, device: str, low_memory=True, torch_dtype=None):
        # Implementation similar to GPT2/LLaMA with safe_copy + low_memory
        return model  # Replace with actual mapping if needed


class GemmaWeightMapper(BaseWeightMapper):
    """Weight mapper for Google Gemma models"""
    
    def map_weights(self, model: nn.Module, params: dict, config, device: str, low_memory=True, torch_dtype=None):
        # Implementation similar to LLaMA with safe_copy + low_memory
        return model  # Replace with actual mapping if needed


# ============================================
# Mapper Registry
# ============================================

WEIGHT_MAPPERS: Dict[str, BaseWeightMapper] = {
    "gpt2_mapper": GPT2WeightMapper(),
    "llama_mapper": LlamaWeightMapper(),
    "mistral_mapper": MistralWeightMapper(),
    "phi_mapper": PhiWeightMapper(),
    "gemma_mapper": GemmaWeightMapper(),
}
