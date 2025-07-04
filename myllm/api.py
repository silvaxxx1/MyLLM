import os
import torch
import torch.nn as nn
from model import GPT
from config import Config
from utils.download_weight import download_safetensors, load_safetensors, load_gpt2_weights, get_gpt2_safetensors_url , Spinner 

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

    def load(self, model_variant, model_family="gpt2", cache_dir="./models", efficient=True):

        if self.model is None:
            if self.config is None:
                raise RuntimeError("Config must be set before loading weights.")
            self.model = GPT(self.config).to(self.device)

        if model_family == "gpt2":
            filename = f"model-{model_variant}.safetensors"
            url = get_gpt2_safetensors_url(model_variant)
            
            filepath = download_safetensors(filename, cache_dir, url)

            params = load_safetensors(filepath)
            
            with Spinner("Assigning weights to model"):
                load_gpt2_weights(self.model, params)
        else:
            raise NotImplementedError(f"Loading weights for {model_family} not implemented.")


    def save(self, save_path: str):
        if self.model is None:
            raise RuntimeError("Model not initialized. Nothing to save.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, generation_config: dict = None):
        if self.model is None:
            raise RuntimeError("Model not initialized. Call init_from_config or load first.")

        max_length = generation_config.get("max_length", 50) if generation_config else 50
        temperature = generation_config.get("temperature", 1.0) if generation_config else 1.0
        top_k = generation_config.get("top_k", None) if generation_config else None
        top_p = generation_config.get("top_p", None) if generation_config else None

        self.model.eval()
        generated = input_ids.to(self.device)

        for _ in range(max_length):
            outputs = self.model(generated)  # (B, T, vocab_size)
            logits = outputs[:, -1, :] / temperature

            if top_k is not None:
                logits = self._top_k_logits(logits, top_k)
            if top_p is not None:
                logits = self._top_p_logits(logits, top_p)

            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_tokens), dim=1)

        return generated

    def _top_k_logits(self, logits, k):
        if k == 0:
            return logits
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1].unsqueeze(1)
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

    def _top_p_logits(self, logits, p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[:, 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits
