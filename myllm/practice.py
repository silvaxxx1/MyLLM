import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from config import Config 
from model import GPT 
from api import LLM 
from transformers import GPT2Tokenizer 

device = "cpu"
model_variant = "gpt2-small"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is defined
config = Config.from_name(model_variant)
llm = LLM(config=config, device=device)
llm.load(model_variant=model_variant, model_family="gpt2")


def top_k_filter(logits, k):
    val , _ = torch.topk(logits, k) 
    return torch.where(logits < val[:, -1].unsqueeze(1), torch.full_like(logits, float('-inf')), logits) 

def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[:, 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, float('-inf'))


@torch.no_grad()
def gen(model, tokenizer, prompt, max_len, temperature=1.0, top_k=25, top_p=0.95):
    # Encode the prompt
    prompt = tokenizer.encode(prompt, return_tensors="pt").to(device)
    model.model.eval()  # Important: inner model
    gen = prompt.clone()

    for _ in range(max_len):
        output = model.model(gen)  # Corrected line
        logits = output[:, -1, :] 

        logits = logits / temperature
        logits = top_k_filter(logits, k=top_k)
        logits = top_p_logits(logits, p=top_p)

        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        gen = torch.cat((gen, next_token), dim=1)

    return gen



prompt = "Everything moves you forward."
input_ids = gen(llm, tokenizer, prompt, max_len=10, temperature=1.0, top_k=50, top_p=0.95)
ouptut_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print(input_ids.shape)
print(ouptut_text)