import os
import time
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from myllm.api import LLM
from myllm.Configs import ModelConfig, GenerationConfig


class Benchmark:
    """
    Benchmarking suite for the LLM class.
    Tests different GenerationConfig strategies and reports performance metrics.
    """

    def __init__(self, model_name: str = "gpt2-medium", tokenizer_name: str = "gpt2",
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self._setup_directories()
        self._load_model()

    def _setup_directories(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.assets_dir = os.path.join(self.base_dir, "Assets")
        self.reports_dir = os.path.join(self.assets_dir, "Reports")
        os.makedirs(self.assets_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.assets_dir, f"benchmark_{self.timestamp}.csv")
        self.json_path = os.path.join(self.assets_dir, f"benchmark_{self.timestamp}.json")
        self.html_path = os.path.join(self.reports_dir, f"benchmark_{self.timestamp}.html")

    def _load_model(self):
        print(f"Loading {self.model_name}...")
        config = ModelConfig.from_name("gpt2-small")
        self.llm = LLM(config=config, device=self.device)
        self.llm.load(self.model_name, self.tokenizer_name)

        # Use HF tokenizer for encoding/decoding benchmark prompts
        from transformers import GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _encode(self, prompt: str, batch_size: int = 1) -> torch.Tensor:
        ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if batch_size > 1:
            ids = ids.expand(batch_size, -1)
        return ids.to(self.device)

    def run_config(self, name: str, gen_config: GenerationConfig,
                   prompt: str = "The future of AI is", batch_size: int = 1) -> dict:
        input_ids = self._encode(prompt, batch_size)
        gen_config.pad_token_id = self.tokenizer.pad_token_id

        # Warmup
        with torch.no_grad():
            self.llm.generate(input_ids, gen_config)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        t0 = time.perf_counter()

        with torch.no_grad():
            output = self.llm.generate(input_ids, gen_config)

        elapsed = time.perf_counter() - t0
        mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        total_tokens = output["tokens"].numel()
        tok_per_sec = total_tokens / elapsed
        mem_mb = (mem_after - mem_before) / 1024 / 1024
        sample = self.tokenizer.decode(output["tokens"][0], skip_special_tokens=True)[:100]

        result = {
            "name": name,
            "batch_size": batch_size,
            "tokens_per_second": tok_per_sec,
            "elapsed_time": elapsed,
            "total_tokens": total_tokens,
            "memory_used_mb": mem_mb,
            "timestamp": datetime.now().isoformat(),
            "sample_output": sample + "...",
        }
        self.results.append(result)
        print(f"  {name:<40} | {tok_per_sec:>8.1f} tok/s | {elapsed:.3f}s | {mem_mb:.1f} MB")
        return result

    def run_suite(self, prompt: str = "The future of AI is"):
        print("\n" + "=" * 70)
        print("LLM BENCHMARK SUITE")
        print("=" * 70)
        print(f"{'Config':<40} | {'tok/s':>8} | {'time':>6} | {'mem':>6}")
        print("-" * 70)

        # --- Sampling strategy comparison ---
        configs = {
            "Greedy (no sampling)": GenerationConfig(
                max_length=20, do_sample=False, use_kv_cache=True,
                use_optimized_sampler=False,
            ),
            "Top-K (k=50)": GenerationConfig(
                max_length=20, do_sample=True, use_kv_cache=True,
                use_optimized_sampler=True, apply_top_k_sampling=True,
                top_k=50, apply_top_p_sampling=False,
            ),
            "Top-P (p=0.95)": GenerationConfig(
                max_length=20, do_sample=True, use_kv_cache=True,
                use_optimized_sampler=True, apply_top_k_sampling=False,
                apply_top_p_sampling=True, top_p=0.95,
            ),
            "Top-K + Top-P": GenerationConfig(
                max_length=20, do_sample=True, use_kv_cache=True,
                use_optimized_sampler=True, apply_top_k_sampling=True,
                top_k=50, apply_top_p_sampling=True, top_p=0.95,
            ),
            "All opts + rep penalty": GenerationConfig(
                max_length=20, do_sample=True, use_kv_cache=True,
                use_optimized_sampler=True, apply_top_k_sampling=True,
                top_k=50, apply_top_p_sampling=True, top_p=0.95,
                apply_repetition_penalty=True, repetition_penalty=1.2,
            ),
            "No KV cache": GenerationConfig(
                max_length=20, do_sample=True, use_kv_cache=False,
                use_optimized_sampler=True, apply_top_k_sampling=True,
                top_k=50, apply_top_p_sampling=True, top_p=0.95,
            ),
        }
        for name, cfg in configs.items():
            self.run_config(name, cfg, prompt=prompt, batch_size=1)

        # --- Sequence length scaling ---
        print("\n--- Sequence length scaling ---")
        for length in [10, 20, 50, 100]:
            cfg = GenerationConfig(
                max_length=length, do_sample=True, use_kv_cache=True,
                use_optimized_sampler=True, apply_top_k_sampling=True,
                top_k=50, apply_top_p_sampling=True, top_p=0.95,
            )
            self.run_config(f"Length={length}", cfg, prompt=prompt, batch_size=1)

        # --- Batch size scaling ---
        print("\n--- Batch size scaling ---")
        for bs in [1, 2, 4, 8]:
            cfg = GenerationConfig(
                max_length=20, do_sample=True, use_kv_cache=True,
                use_optimized_sampler=True, apply_top_k_sampling=True,
                top_k=50, apply_top_p_sampling=True, top_p=0.95,
            )
            self.run_config(f"Batch={bs}", cfg, prompt=prompt, batch_size=bs)

    def analyze(self) -> pd.DataFrame:
        df = pd.DataFrame(self.results)
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"Configs tested   : {len(df)}")
        best = df.loc[df["tokens_per_second"].idxmax()]
        worst = df.loc[df["tokens_per_second"].idxmin()]
        print(f"Fastest          : {best['name']} ({best['tokens_per_second']:.1f} tok/s)")
        print(f"Slowest          : {worst['name']} ({worst['tokens_per_second']:.1f} tok/s)")
        print(f"Average speed    : {df['tokens_per_second'].mean():.1f} tok/s")
        ratio = best["tokens_per_second"] / worst["tokens_per_second"]
        print(f"Best/worst ratio : {ratio:.2f}x")
        return df

    def visualize(self, df: pd.DataFrame) -> str:
        print("\nGenerating plots...")
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("LLM Benchmark Results", fontsize=14, fontweight="bold")

        # 1. Overall throughput
        ax = axes[0, 0]
        sorted_df = df.sort_values("tokens_per_second")
        colors = plt.cm.RdYlGn(sorted_df["tokens_per_second"] / sorted_df["tokens_per_second"].max())
        ax.barh(range(len(sorted_df)), sorted_df["tokens_per_second"], color=colors)
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels([n[:35] for n in sorted_df["name"]], fontsize=8)
        ax.set_xlabel("Tokens / second")
        ax.set_title("Throughput Comparison")
        ax.grid(True, alpha=0.3)

        # 2. Batch size scaling
        ax = axes[0, 1]
        batch_df = df[df["name"].str.startswith("Batch=")]
        if not batch_df.empty:
            bs = batch_df["batch_size"].values
            sp = batch_df["tokens_per_second"].values
            ax.plot(bs, sp, "o-", linewidth=2, markersize=8)
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Tokens / second")
            ax.set_title("Batch Size Scaling")
            ax.set_xscale("log", base=2)
            ax.grid(True, alpha=0.3)

        # 3. Sequence length scaling
        ax = axes[1, 0]
        len_df = df[df["name"].str.startswith("Length=")]
        if not len_df.empty:
            lengths = len_df["total_tokens"].values
            speeds = len_df["tokens_per_second"].values
            ax.plot(lengths, speeds, "s-", linewidth=2, markersize=8, color="steelblue")
            ax.set_xlabel("Total Tokens Generated")
            ax.set_ylabel("Tokens / second")
            ax.set_title("Sequence Length Scaling")
            ax.grid(True, alpha=0.3)

        # 4. Memory vs throughput
        ax = axes[1, 1]
        sc = ax.scatter(df["memory_used_mb"], df["tokens_per_second"],
                        c=df["batch_size"], s=80, alpha=0.8, cmap="viridis")
        ax.set_xlabel("Memory Used (MB)")
        ax.set_ylabel("Tokens / second")
        ax.set_title("Memory vs Throughput")
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="Batch Size")

        plt.tight_layout()
        plot_path = os.path.join(self.assets_dir, f"benchmark_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.show()
        print(f"Plot saved: {plot_path}")
        return plot_path

    def save_html_report(self, df: pd.DataFrame) -> str:
        max_spd = df["tokens_per_second"].max()
        min_spd = df["tokens_per_second"].min()
        rows_html = ""
        for _, row in df.sort_values("tokens_per_second", ascending=False).iterrows():
            pct = row["tokens_per_second"] / max_spd * 100
            cls = "best" if row["tokens_per_second"] == max_spd else (
                  "worst" if row["tokens_per_second"] == min_spd else "")
            rows_html += (
                f'<tr class="{cls}"><td>{row["name"]}</td>'
                f'<td>{row["tokens_per_second"]:.1f}</td>'
                f'<td>{row["elapsed_time"]:.3f}</td>'
                f'<td>{row["memory_used_mb"]:.1f}</td>'
                f'<td>{row["batch_size"]}</td>'
                f'<td>{pct:.1f}%</td></tr>\n'
            )
        html = f"""<!DOCTYPE html>
<html>
<head>
  <title>LLM Benchmark - {self.timestamp}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
    .container {{ max-width: 1100px; margin: 0 auto; background: white; padding: 24px; border-radius: 10px; }}
    h1 {{ text-align: center; color: #333; }}
    .card {{ display: inline-block; margin: 8px; padding: 16px 24px; background: #f0f4f8;
             border-radius: 8px; text-align: center; }}
    .val {{ font-size: 1.8em; font-weight: bold; color: #2c3e50; }}
    .lbl {{ color: #7f8c8d; font-size: .9em; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 10px 12px; text-align: left; }}
    th {{ background: #3498db; color: white; }}
    tr:nth-child(even) {{ background: #f9f9f9; }}
    .best {{ background: #2ecc71 !important; color: white; }}
    .worst {{ background: #e74c3c !important; color: white; }}
  </style>
</head>
<body>
<div class="container">
  <h1>LLM Benchmark Report</h1>
  <p style="text-align:center;color:#888">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; model: {self.model_name} &nbsp;|&nbsp; device: {self.device}</p>
  <div>
    <div class="card"><div class="val">{len(df)}</div><div class="lbl">Configs Tested</div></div>
    <div class="card"><div class="val">{max_spd:.1f}</div><div class="lbl">Max tok/s</div></div>
    <div class="card"><div class="val">{df['tokens_per_second'].mean():.1f}</div><div class="lbl">Avg tok/s</div></div>
    <div class="card"><div class="val">{max_spd/min_spd:.2f}x</div><div class="lbl">Best/Worst Ratio</div></div>
  </div>
  <h2>Results</h2>
  <table>
    <thead><tr>
      <th>Config</th><th>tok/s</th><th>Time (s)</th>
      <th>Memory (MB)</th><th>Batch</th><th>Relative</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
</body>
</html>"""
        with open(self.html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML report saved: {self.html_path}")
        return self.html_path

    def save_data(self, df: pd.DataFrame):
        df.to_csv(self.csv_path, index=False)
        summary = {
            "timestamp": self.timestamp,
            "model": self.model_name,
            "device": self.device,
            "fastest": df.loc[df["tokens_per_second"].idxmax(), "name"],
            "fastest_tok_per_sec": df["tokens_per_second"].max(),
            "average_tok_per_sec": df["tokens_per_second"].mean(),
            "results": df.to_dict("records"),
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"CSV : {self.csv_path}")
        print(f"JSON: {self.json_path}")

    def run(self, prompt: str = "The future of AI is"):
        self.run_suite(prompt=prompt)
        df = self.analyze()
        self.visualize(df)
        self.save_html_report(df)
        self.save_data(df)
        print(f"\nDone. Report: {self.html_path}")
        return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MyLLM Benchmark")
    parser.add_argument("--model", default="gpt2-medium")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompt", default="The future of AI is")
    parser.add_argument("--mode", choices=["full", "analyze", "visualize", "report"],
                        default="full")
    parser.add_argument("--input", help="Path to existing CSV (for analyze/visualize/report modes)")
    args = parser.parse_args()

    bench = Benchmark(model_name=args.model, tokenizer_name=args.tokenizer, device=args.device)

    if args.mode == "full":
        bench.run(prompt=args.prompt)
    else:
        if not args.input:
            raise ValueError(f"--input required for mode '{args.mode}'")
        df = pd.read_csv(args.input)
        bench.results = df.to_dict("records")
        if args.mode == "analyze":
            bench.analyze()
        elif args.mode == "visualize":
            bench.visualize(df)
        elif args.mode == "report":
            bench.save_html_report(df)


if __name__ == "__main__":
    main()
