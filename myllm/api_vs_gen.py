import os
import time
import torch
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer
from config import Config
from api import LLM as BasicLLM
from gen import LLM as OptimizedLLM, GenerationConfig
import csv

# ----------------------------
# Path Setup
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "Assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

csv_path = os.path.join(ASSETS_DIR, "benchmark_results.csv")
png_path = os.path.join(ASSETS_DIR, "benchmark_plot.png")

# ----------------------------
# Setup
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
prompt = "The future of AI is"
batch_prompts = [prompt] * 8

print("🚀 Initializing models with pretrained weights...")
preset_config = Config.from_name("gpt2-small")
basic_llm = BasicLLM(config=preset_config, device=device)
optimized_llm = OptimizedLLM(config=preset_config, device=device)

basic_llm.load(model_variant="gpt2-small", model_family="gpt2")
optimized_llm.load(model_variant="gpt2-small", model_family="gpt2")

# ----------------------------
# Benchmark Function
# ----------------------------
def benchmark(llm, gen_config, name: str, batch: bool = False):
    torch.cuda.empty_cache()
    print(f"\n⚡ Benchmarking: {name} | Batch mode: {'ON' if batch else 'OFF'}")
    
    if batch:
        encoded = [tokenizer.encode(p, return_tensors="pt").to(device) for p in batch_prompts]
        max_len = max(t.shape[1] for t in encoded)
        input_ids = [torch.cat([t, torch.full((1, max_len - t.shape[1]), gen_config.pad_token_id, device=device)], dim=1) for t in encoded]
        input_ids = torch.cat(input_ids, dim=0)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    start = time.time()
    with torch.no_grad():
        output = llm.generate(input_ids, gen_config)
    end = time.time()

    total_tokens = output["tokens"].numel()
    elapsed = end - start
    speed = total_tokens / elapsed

    print(f"⏱ Time taken: {elapsed:.3f}s, Tokens/sec: {speed:.2f}")
    return elapsed, speed

# ----------------------------
# Run Benchmarks
# ----------------------------
results = {
    "Basic LLM (Single Input)": benchmark(
        basic_llm,
        GenerationConfig(max_length=20, pad_token_id=tokenizer.pad_token_id or 0),
        "Basic LLM", batch=False
    )
}

# Optimization Variants
gen_variants = {
    "Optimized LLM - No Optimizations": GenerationConfig(
        max_length=20, use_optimized_sampler=False,
        use_kv_cache=False, apply_repetition_penalty=False,
        apply_top_k_sampling=False, apply_top_p_sampling=False,
        pad_token_id=tokenizer.pad_token_id or 0
    ),
    "Optimized LLM - KV Cache Only": GenerationConfig(
        max_length=20, use_optimized_sampler=False,
        use_kv_cache=True, apply_repetition_penalty=False,
        apply_top_k_sampling=False, apply_top_p_sampling=False,
        pad_token_id=tokenizer.pad_token_id or 0
    ),
    "Optimized LLM - Repetition Penalty Only": GenerationConfig(
        max_length=20, use_optimized_sampler=True,
        use_kv_cache=False, apply_repetition_penalty=True,
        repetition_penalty=1.2,
        apply_top_k_sampling=False, apply_top_p_sampling=False,
        pad_token_id=tokenizer.pad_token_id or 0
    ),
    "Optimized LLM - Top-K Sampling Only": GenerationConfig(
        max_length=20, use_optimized_sampler=True,
        use_kv_cache=False, apply_repetition_penalty=False,
        apply_top_k_sampling=True, top_k=50,
        apply_top_p_sampling=False,
        pad_token_id=tokenizer.pad_token_id or 0
    ),
    "Optimized LLM - Top-P Sampling Only": GenerationConfig(
        max_length=20, use_optimized_sampler=True,
        use_kv_cache=False, apply_repetition_penalty=False,
        apply_top_k_sampling=False,
        apply_top_p_sampling=True, top_p=0.95,
        pad_token_id=tokenizer.pad_token_id or 0
    ),
    "Optimized LLM - KV Cache + Repetition Penalty": GenerationConfig(
        max_length=20, use_optimized_sampler=True,
        use_kv_cache=True, apply_repetition_penalty=True,
        repetition_penalty=1.2,
        apply_top_k_sampling=False, apply_top_p_sampling=False,
        pad_token_id=tokenizer.pad_token_id or 0
    ),
    "Optimized LLM - KV Cache + Top-K + Top-P": GenerationConfig(
        max_length=20, use_optimized_sampler=True,
        use_kv_cache=True, apply_repetition_penalty=False,
        apply_top_k_sampling=True, top_k=50,
        apply_top_p_sampling=True, top_p=0.95,
        pad_token_id=tokenizer.pad_token_id or 0
    ),
    "Optimized LLM - All Optimizations": GenerationConfig(
        max_length=20, use_optimized_sampler=True,
        use_kv_cache=True, apply_repetition_penalty=True,
        repetition_penalty=1.2,
        apply_top_k_sampling=True, top_k=50,
        apply_top_p_sampling=True, top_p=0.95,
        pad_token_id=tokenizer.pad_token_id or 0
    ),
}

for name, cfg in gen_variants.items():
    results[name] = benchmark(optimized_llm, cfg, name, batch=False)

# ----------------------------
# Save Results to CSV
# ----------------------------
with open(csv_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Variant", "Time (s)", "Tokens/sec"])
    for name, (t, s) in results.items():
        writer.writerow([name, f"{t:.3f}", f"{s:.2f}"])

print(f"\n📁 Benchmark CSV saved to: {csv_path}")

# ----------------------------
# Print Table
# ----------------------------
print("\n📊 Benchmark Summary")
print("-" * 75)
print(f"{'Variant':<45} | {'Time (s)':>10} | {'Tokens/sec':>15}")
print("-" * 75)
for name, (t, s) in results.items():
    print(f"{name:<45} | {t:>10.3f} | {s:>15.2f}")
print("-" * 75)

# ----------------------------
# Plot Results
# ----------------------------
variants = list(results.keys())
times = [results[v][0] for v in variants]
speeds = [results[v][1] for v in variants]

fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:blue'
ax1.set_xlabel('Benchmark Variant')
ax1.set_ylabel('Time (s)', color=color)
ax1.bar(variants, times, color=color, alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(range(len(variants)))
ax1.set_xticklabels(variants, rotation=45, ha='right')
ax1.set_ylim(0, max(times) * 1.2)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Tokens per Second', color=color)
ax2.plot(range(len(variants)), speeds, color=color, marker='o', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, max(speeds) * 1.2)

plt.title("LLM Benchmark: Impact of Individual & Combined Optimizations")
plt.tight_layout()
plt.savefig(png_path)
plt.show()

print(f"📁 Plot saved to: {png_path}")
