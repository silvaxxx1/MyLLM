import torch.nn as nn 
from typing import Dict
from abc import ABC, abstractmethod
from tqdm import tqdm
import torch


class BaseWeightMapper(ABC):
    """Base class for weight mapping strategies"""
    
    @abstractmethod
    def map_weights(
        self, 
        model: nn.Module, 
        params: dict, 
        config, 
        device: str,
        low_memory: bool = True,  # 🆕
        torch_dtype: torch.dtype = None  # 🆕
    ) -> nn.Module:
        """Map checkpoint weights to model parameters"""
        pass
    
    def safe_copy(
        self, 
        name: str, 
        dest, 
        params: dict,
        device: str = "cpu",  # 🆕
        clear_after: bool = True  # 🆕
    ):
        """Safe tensor copy with shape matching and memory cleanup"""
        tensor = params.get(name)
        if tensor is not None and dest is not None:
            # Handle transposition if needed
            if tensor.shape[::-1] == dest.shape:
                tensor = tensor.T
            
            # 🆕 Move to device incrementally
            tensor = tensor.to(device, non_blocking=True)
            
            with torch.no_grad():
                dest.copy_(tensor)
            
            # 🆕 Clear memory immediately after copying
            if clear_after:
                del tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def count_blocks(self, params: dict, pattern: str) -> int:
        """Count transformer blocks in checkpoint"""
        return len([k for k in params if pattern in k])


class GPT2WeightMapper(BaseWeightMapper):
    """Weight mapper for GPT-2 style models"""
    
    def map_weights(
        self, 
        model: nn.Module, 
        params: dict, 
        config, 
        device: str,
        low_memory: bool = True,  # 🆕
        torch_dtype: torch.dtype = None  # 🆕
    ) -> nn.Module:
        # Model stays on CPU during loading if low_memory
        if not low_memory:
            model.to("cpu")
        
        # Token embeddings
        self.safe_copy("wte.weight", model.wte.weight, params, device="cpu", clear_after=low_memory)
        
        # Position embeddings (GPT-2 uses learned positions)
        if hasattr(model, "wpe"):
            self.safe_copy("wpe.weight", model.wpe.weight, params, device="cpu", clear_after=low_memory)
        
        # Count blocks
        num_blocks = self.count_blocks(params, ".attn.c_attn.weight")
        print(f"✅ Detected {num_blocks} transformer blocks")
        
        with tqdm(total=num_blocks, desc="Loading GPT-2 blocks", ascii=True) as pbar:
            for i in range(num_blocks):
                prefix = f"h.{i}"
                block = model.transformer[f"block_{i}"]
                
                # Attention (combined QKV projection)
                self.safe_copy(f"{prefix}.attn.c_attn.weight", block.attn.qkv.weight, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.attn.c_attn.bias", block.attn.qkv.bias, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.attn.c_proj.weight", block.attn.proj.weight, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.attn.c_proj.bias", block.attn.proj.bias, params, device="cpu", clear_after=low_memory)
                
                # MLP
                self.safe_copy(f"{prefix}.mlp.c_fc.weight", block.mlp.fc.weight, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.mlp.c_fc.bias", block.mlp.fc.bias, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.mlp.c_proj.weight", block.mlp.proj.weight, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.mlp.c_proj.bias", block.mlp.proj.bias, params, device="cpu", clear_after=low_memory)
                
                # LayerNorm
                self.safe_copy(f"{prefix}.ln_1.weight", block.norm1.weight, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.ln_1.bias", block.norm1.bias, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.ln_2.weight", block.norm2.weight, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.ln_2.bias", block.norm2.bias, params, device="cpu", clear_after=low_memory)
                
                pbar.update(1)
        
        # Final LayerNorm
        self.safe_copy("ln_f.weight", model.ln_f.weight, params, device="cpu", clear_after=low_memory)
        self.safe_copy("ln_f.bias", model.ln_f.bias, params, device="cpu", clear_after=low_memory)
        
        # LM head (or tie to embeddings)
        if hasattr(model, "lm_head"):
            if "lm_head.weight" in params:
                self.safe_copy("lm_head.weight", model.lm_head.weight, params, device="cpu", clear_after=low_memory)
            else:
                # Weight tying
                model.lm_head.weight = model.wte.weight
        
        # 🆕 Move to final device
        if low_memory:
            print(f"📦 Moving model to {device}...")
        model.to(device)
        
        return model


class LlamaWeightMapper(BaseWeightMapper):
    """Weight mapper for LLaMA/Llama2/Llama3 style models"""
    
    def map_weights(
        self, 
        model: nn.Module, 
        params: dict, 
        config, 
        device: str,
        low_memory: bool = True,  # 🆕
        torch_dtype: torch.dtype = None  # 🆕
    ) -> nn.Module:
        if not low_memory:
            model.to("cpu")
        
        # Token embeddings
        self.safe_copy("model.embed_tokens.weight", model.wte.weight, params, device="cpu", clear_after=low_memory)
        
        # Count blocks
        num_blocks = self.count_blocks(params, ".self_attn.q_proj.weight")
        print(f"✅ Detected {num_blocks} transformer blocks")
        
        with tqdm(total=num_blocks, desc="Loading LLaMA blocks", ascii=True) as pbar:
            for i in range(num_blocks):
                prefix = f"model.layers.{i}"
                block = model.transformer[f"block_{i}"]
                
                # Attention - LLaMA has separate Q, K, V projections
                q_weight = params.get(f"{prefix}.self_attn.q_proj.weight")
                k_weight = params.get(f"{prefix}.self_attn.k_proj.weight")
                v_weight = params.get(f"{prefix}.self_attn.v_proj.weight")
                
                if q_weight is not None and k_weight is not None and v_weight is not None:
                    # Combine Q, K, V into single tensor
                    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                    with torch.no_grad():
                        block.attn.qkv.weight.copy_(qkv_weight.to(block.attn.qkv.weight.device))
                    
                    # 🆕 Clear memory
                    if low_memory:
                        del q_weight, k_weight, v_weight, qkv_weight
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Output projection
                self.safe_copy(f"{prefix}.self_attn.o_proj.weight", block.attn.proj.weight, params, device="cpu", clear_after=low_memory)
                
                # MLP - LLaMA uses SwiGLU (gate + up + down)
                self.safe_copy(f"{prefix}.mlp.gate_proj.weight", block.mlp.fc_1.weight, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.mlp.up_proj.weight", block.mlp.fc_2.weight, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.mlp.down_proj.weight", block.mlp.proj.weight, params, device="cpu", clear_after=low_memory)
                
                # RMSNorm
                self.safe_copy(f"{prefix}.input_layernorm.weight", block.norm1.weight, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.post_attention_layernorm.weight", block.norm2.weight, params, device="cpu", clear_after=low_memory)
                
                pbar.update(1)
        
        # Final RMSNorm
        self.safe_copy("model.norm.weight", model.ln_f.weight, params, device="cpu", clear_after=low_memory)
        
        # LM head
        if hasattr(model, "lm_head"):
            self.safe_copy("lm_head.weight", model.lm_head.weight, params, device="cpu", clear_after=low_memory)
        
        # 🆕 Move to final device
        if low_memory:
            print(f"📦 Moving model to {device}...")
        model.to(device)
        
        return model


class MistralWeightMapper(LlamaWeightMapper):
    """Mistral uses same structure as LLaMA"""
    pass


class PhiWeightMapper(BaseWeightMapper):
    """Weight mapper for Microsoft Phi models"""
    
    def map_weights(
        self, 
        model: nn.Module, 
        params: dict, 
        config, 
        device: str,
        low_memory: bool = True,  # 🆕
        torch_dtype: torch.dtype = None  # 🆕
    ) -> nn.Module:
        if not low_memory:
            model.to("cpu")
        
        # Phi uses slightly different naming
        self.safe_copy("transformer.embd.wte.weight", model.wte.weight, params, device="cpu", clear_after=low_memory)
        
        num_blocks = self.count_blocks(params, ".mixer.Wqkv.weight")
        print(f"✅ Detected {num_blocks} transformer blocks")
        
        with tqdm(total=num_blocks, desc="Loading Phi blocks", ascii=True) as pbar:
            for i in range(num_blocks):
                prefix = f"transformer.h.{i}"
                block = model.transformer[f"block_{i}"]
                
                # Phi uses combined Wqkv
                self.safe_copy(f"{prefix}.mixer.Wqkv.weight", block.attn.qkv.weight, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.mixer.Wqkv.bias", block.attn.qkv.bias, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.mixer.out_proj.weight", block.attn.proj.weight, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.mixer.out_proj.bias", block.attn.proj.bias, params, device="cpu", clear_after=low_memory)
                
                # MLP
                self.safe_copy(f"{prefix}.mlp.fc1.weight", block.mlp.fc.weight, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.mlp.fc1.bias", block.mlp.fc.bias, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.mlp.fc2.weight", block.mlp.proj.weight, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.mlp.fc2.bias", block.mlp.proj.bias, params, device="cpu", clear_after=low_memory)
                
                # LayerNorm
                self.safe_copy(f"{prefix}.ln.weight", block.norm1.weight, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.ln.bias", block.norm1.bias, params, device="cpu", clear_after=low_memory)
                
                pbar.update(1)
        
        self.safe_copy("transformer.ln_f.weight", model.ln_f.weight, params, device="cpu", clear_after=low_memory)
        self.safe_copy("transformer.ln_f.bias", model.ln_f.bias, params, device="cpu", clear_after=low_memory)
        self.safe_copy("lm_head.linear.weight", model.lm_head.weight, params, device="cpu", clear_after=low_memory)
        self.safe_copy("lm_head.linear.bias", model.lm_head.bias, params, device="cpu", clear_after=low_memory)
        
        # 🆕 Move to final device
        if low_memory:
            print(f"📦 Moving model to {device}...")
        model.to(device)
        
        return model


class GemmaWeightMapper(BaseWeightMapper):
    """Weight mapper for Google Gemma models"""
    
    def map_weights(
        self, 
        model: nn.Module, 
        params: dict, 
        config, 
        device: str,
        low_memory: bool = True,  # 🆕
        torch_dtype: torch.dtype = None  # 🆕
    ) -> nn.Module:
        if not low_memory:
            model.to("cpu")
        
        # Gemma embeddings
        self.safe_copy("model.embed_tokens.weight", model.wte.weight, params, device="cpu", clear_after=low_memory)
        
        num_blocks = self.count_blocks(params, ".self_attn.q_proj.weight")
        print(f"✅ Detected {num_blocks} transformer blocks")
        
        with tqdm(total=num_blocks, desc="Loading Gemma blocks", ascii=True) as pbar:
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
                        block.attn.qkv.weight.copy_(qkv_weight.to(block.attn.qkv.weight.device))
                    
                    # 🆕 Clear memory
                    if low_memory:
                        del q_weight, k_weight, v_weight, qkv_weight
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                self.safe_copy(f"{prefix}.self_attn.o_proj.weight", block.attn.proj.weight, params, device="cpu", clear_after=low_memory)
                
                # Gemma MLP uses GELU instead of SiLU
                self.safe_copy(f"{prefix}.mlp.gate_proj.weight", block.mlp.fc_1.weight, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.mlp.up_proj.weight", block.mlp.fc_2.weight, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.mlp.down_proj.weight", block.mlp.proj.weight, params, device="cpu", clear_after=low_memory)
                
                # RMSNorm
                self.safe_copy(f"{prefix}.input_layernorm.weight", block.norm1.weight, params, device="cpu", clear_after=low_memory)
                self.safe_copy(f"{prefix}.post_attention_layernorm.weight", block.norm2.weight, params, device="cpu", clear_after=low_memory)
                
                pbar.update(1)
        
        self.safe_copy("model.norm.weight", model.ln_f.weight, params, device="cpu", clear_after=low_memory)
        
        if hasattr(model, "lm_head"):
            self.safe_copy("lm_head.weight", model.lm_head.weight, params, device="cpu", clear_after=low_memory)
        
        # 🆕 Move to final device
        if low_memory:
            print(f"📦 Moving model to {device}...")
        model.to(device)
        
        return model


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
                