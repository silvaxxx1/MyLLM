import torch 
from typing import Optional, List
from dataclasses import dataclass 


@dataclass
class GenerationConfig:
    max_length: int = 20
    max_new_tokens: Optional[int] = None
    min_length: Optional[int] = None
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    typical_p: Optional[float] = None
    do_sample: bool = True
    use_kv_cache: bool = True
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: Optional[int] = None
    early_stopping: bool = True
    eos_token_ids: Optional[List[int]] = None
    pad_token_id: Optional[int] = None
    return_tokens: bool = True
    return_logprobs: bool = False
    output_scores: bool = False
    output_attentions: bool = False
    output_hidden_states: bool = False



@torch.no_grad()
def generate(self, input_ids: torch.Tensor, generation_config: GenerationConfig):
        model = self.model.eval()
        B, T = input_ids.shape
        device = input_ids.device
        generated = input_ids.clone()

        if generation_config.use_kv_cache:
            model.reset_cache()
            model.initialize_kv_cache(batch_size=B, max_seq_len=T + generation_config.max_length)
            _ = model(input_ids, use_cache=False, pos_offset=0)

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

        logprobs = []
        for step in range(generation_config.max_length):
            if generation_config.use_kv_cache:
                input_token = generated[:, -1:]
                pos_offset = model.transformer["block_0"].attn.kv_cache.size
            else:
                input_token = generated
                pos_offset = 0

            logits = model(input_token, use_cache=generation_config.use_kv_cache, pos_offset=pos_offset)
            logits = logits[:, -1, :] / generation_config.temperature

            if generation_config.repetition_penalty != 1.0:
                for i in range(B):
                    for prev_token in set(generated[i].tolist()):
                        logits[i, prev_token] /= generation_config.repetition_penalty

            if generation_config.top_k is not None:
                logits = _top_k_logits(logits, generation_config.top_k)
            if generation_config.top_p is not None:
                logits = _top_p_logits(logits, generation_config.top_p)

            probs = torch.softmax(logits, dim=-1)
            if generation_config.do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            if generation_config.return_logprobs:
                logprob = torch.log(probs.gather(1, next_token))
                logprobs.append(logprob)

            generated = torch.cat([generated, next_token], dim=1)

            if generation_config.eos_token_ids and all((next_token == eos).all() for eos in generation_config.eos_token_ids):
                break

        output = {"tokens": generated}
        if generation_config.return_logprobs:
            output["logprobs"] = torch.cat(logprobs, dim=1)
        return output

def generate_text(self, prompt: str, tokenizer, generation_config: GenerationConfig):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output = self.generate(input_ids, generation_config)
        tokens = output["tokens"][0]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        return {
            "text": text,
            "tokens": tokens.tolist() if generation_config.return_tokens else None,
            "logprobs": output.get("logprobs", None)
        }
