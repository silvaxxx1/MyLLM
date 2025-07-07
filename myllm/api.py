import os
import torch
import torch.nn as nn
from model import GPT
from config import Config
from utils.download_weight import (
    download_safetensors,
    load_safetensors,
    load_gpt2_weights_meta,
    get_gpt2_safetensors_url,
    Spinner
)

# generation_config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class GenerationConfig:
    max_length: int = 20
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    do_sample: bool = True
    use_kv_cache: bool = True
    repetition_penalty: float = 1.0
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None


class LLM(nn.Module):
    def __init__(self, config: Config = None, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.config = config
        self.model = None
        if config is not None:
            self.init_from_config(config)

    def init_from_config(self, config: Config, train_mode: bool = True):
        self.config = config
        self.model = GPT(config).to(self.device)
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

    def load(self, model_variant: str, model_family: str = "gpt2", cache_dir: str = "./models"):
        if model_family != "gpt2":
            raise NotImplementedError(f"Loading weights for {model_family} not implemented.")

        if self.model is None:
            if self.config is None:
                raise RuntimeError("Config must be set before loading weights.")
            self.model = GPT(self.config).to(self.device)

        filename = f"model-{model_variant}.safetensors"
        url = get_gpt2_safetensors_url(model_variant)

        filepath = download_safetensors(filename, cache_dir, url)
        params = load_safetensors(filepath)

        with Spinner("Assigning weights to model"):
            # Load weights into existing model instance
            self.model = load_gpt2_weights_meta(self.model, self.config, params, device=self.device)

    def save(self, save_path: str):
        if self.model is None:
            raise RuntimeError("Model not initialized. Nothing to save.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, generation_config: GenerationConfig, verbose: bool = False):
        max_length = generation_config.max_length
        temperature = generation_config.temperature
        top_k = generation_config.top_k
        top_p = generation_config.top_p
        do_sample = generation_config.do_sample
        use_cache = generation_config.use_kv_cache
        repetition_penalty = generation_config.repetition_penalty
        eos_token_id = generation_config.eos_token_id

        self.model.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        self.model.reset_cache()

        if use_cache:
            max_seq_len = input_ids.size(1) + max_length
            self.model.initialize_kv_cache(batch_size=batch_size, max_seq_len=max_seq_len, dtype=torch.float32)

        generated = input_ids.to(device)

        if use_cache:
            _ = self.model(generated, use_cache=False, pos_offset=0)

        def _top_k_logits(logits, k):
            if k == 0:
                return logits
            values, _ = torch.topk(logits, k)
            min_values = values[:, -1].unsqueeze(1)
            return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

        def _top_p_logits(logits, p):
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            return logits.masked_fill(indices_to_remove, float('-inf'))

        for step in range(max_length - input_ids.size(1)):
            if use_cache:
                input_tokens = generated[:, -1:].to(device)
                pos_offset = self.model.transformer["block_0"].attn.kv_cache.size
            else:
                input_tokens = generated.to(device)
                pos_offset = 0

            logits = self.model(input_tokens, use_cache=use_cache, pos_offset=pos_offset)
            logits = logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                for i in range(logits.size(0)):
                    for prev_token in set(generated[i].tolist()):
                        logits[i, prev_token] /= repetition_penalty

            if top_k is not None:
                logits = _top_k_logits(logits, top_k)
            if top_p is not None:
                logits = _top_p_logits(logits, top_p)

            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat((generated, next_token), dim=1)

            if verbose:
                print(f"[Step {step+1}] Token ID: {next_token.squeeze().tolist()}")
                if use_cache:
                    cache_size = self.model.transformer["block_0"].attn.kv_cache.size
                    print(f"[Step {step+1}] KV Cache Size: {cache_size}")
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    print(f"[Step {step+1}] EOS token generated.")
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

