import os
import time
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import GPT2Tokenizer
from Configs.ModelConfig import ModelConfig
from legacy_api import LLM as BasicLLM
from api import LLM as OptimizedLLM, GenerationConfig
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedBenchmark:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.results = []
        self.setup_directories()
        self.load_models()
        
    def setup_directories(self):
        """Create directories for storing results"""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.assets_dir = os.path.join(self.base_dir, "Assets")
        self.reports_dir = os.path.join(self.assets_dir, "Reports")
        
        os.makedirs(self.assets_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # File paths
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.assets_dir, f"benchmark_results_{self.timestamp}.csv")
        self.json_path = os.path.join(self.assets_dir, f"benchmark_results_{self.timestamp}.json")
        self.html_path = os.path.join(self.reports_dir, f"benchmark_report_{self.timestamp}.html")
        
    def load_models(self):
        """Load both basic and optimized models"""
        print("🚀 Loading models...")
        preset_config = ModelConfig.from_name("gpt2-small")
        
        self.basic_llm = BasicLLM(config=preset_config, device=self.device)
        self.optimized_llm = OptimizedLLM(config=preset_config, device=self.device)
        
        self.basic_llm.load("gpt2-medium", "gpt2")
        self.optimized_llm.load("gpt2-medium", "gpt2")
        
    def benchmark_configuration(self, llm, gen_config, name, batch_size=1, test_prompt="The future of AI is"):
        """Benchmark a specific configuration"""
        torch.cuda.empty_cache()
        
        print(f"⚡ Benchmarking: {name} | Batch size: {batch_size}")
        
        # Prepare input
        if batch_size > 1:
            prompts = [test_prompt] * batch_size
            encoded = [self.tokenizer.encode(p, return_tensors="pt").to(self.device) for p in prompts]
            max_len = max(t.shape[1] for t in encoded)
            input_ids = [torch.cat([t, torch.full((1, max_len - t.shape[1]), gen_config.pad_token_id, device=self.device)], dim=1) for t in encoded]
            input_ids = torch.cat(input_ids, dim=0)
        else:
            input_ids = self.tokenizer.encode(test_prompt, return_tensors="pt").to(self.device)
        
        # Warmup run
        with torch.no_grad():
            _ = llm.generate(input_ids, gen_config)
        
        # Actual benchmark
        start_time = time.time()
        memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        with torch.no_grad():
            output = llm.generate(input_ids, gen_config)
            
        end_time = time.time()
        memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Calculate metrics
        total_tokens = output["tokens"].numel()
        elapsed_time = end_time - start_time
        tokens_per_second = total_tokens / elapsed_time
        memory_used = (memory_after - memory_before) / 1024 / 1024  # MB
        
        # Store results
        result = {
            'name': name,
            'batch_size': batch_size,
            'tokens_per_second': tokens_per_second,
            'elapsed_time': elapsed_time,
            'total_tokens': total_tokens,
            'memory_used_mb': memory_used,
            'timestamp': datetime.now().isoformat(),
            'sample_output': self.tokenizer.decode(output['tokens'][0], skip_special_tokens=True)[:100] + "..."
        }
        
        self.results.append(result)
        print(f"  ⏱ Time: {elapsed_time:.3f}s | 🚀 Speed: {tokens_per_second:.2f} tok/s | 🧠 Memory: {memory_used:.1f}MB")
        
        return result
        
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark with various configurations"""
        print("\n" + "="*80)
        print("🔥 COMPREHENSIVE LLM BENCHMARK")
        print("="*80)
        
        # Test prompts for different scenarios
        test_prompts = [
            "The future of AI is",
            "In a world where technology advances rapidly",
            "Machine learning algorithms can help us"
        ]
        
        batch_sizes = [1, 2, 4, 8, 16]
        
        # 1. Basic LLM baseline
        baseline_config = GenerationConfig(
            max_length=20, 
            pad_token_id=self.tokenizer.pad_token_id or 0
        )
        
        for prompt in test_prompts:
            self.benchmark_configuration(
                self.basic_llm, 
                baseline_config, 
                f"Basic LLM - {prompt[:20]}...", 
                batch_size=1,
                test_prompt=prompt
            )
        
        # 2. Individual optimization analysis
        optimization_configs = {
            "No Optimizations": GenerationConfig(
                max_length=20, use_optimized_sampler=False, use_kv_cache=False,
                apply_repetition_penalty=False, apply_top_k_sampling=False,
                apply_top_p_sampling=False, pad_token_id=self.tokenizer.pad_token_id or 0
            ),
            "KV Cache Only": GenerationConfig(
                max_length=20, use_optimized_sampler=False, use_kv_cache=True,
                apply_repetition_penalty=False, apply_top_k_sampling=False,
                apply_top_p_sampling=False, pad_token_id=self.tokenizer.pad_token_id or 0
            ),
            "Repetition Penalty Only": GenerationConfig(
                max_length=20, use_optimized_sampler=True, use_kv_cache=False,
                apply_repetition_penalty=True, repetition_penalty=1.2,
                apply_top_k_sampling=False, apply_top_p_sampling=False,
                pad_token_id=self.tokenizer.pad_token_id or 0
            ),
            "Top-K Sampling Only": GenerationConfig(
                max_length=20, use_optimized_sampler=True, use_kv_cache=False,
                apply_repetition_penalty=False, apply_top_k_sampling=True,
                top_k=50, apply_top_p_sampling=False,
                pad_token_id=self.tokenizer.pad_token_id or 0
            ),
            "Top-P Sampling Only": GenerationConfig(
                max_length=20, use_optimized_sampler=True, use_kv_cache=False,
                apply_repetition_penalty=False, apply_top_k_sampling=False,
                apply_top_p_sampling=True, top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id or 0
            ),
            "All Optimizations": GenerationConfig(
                max_length=20, use_optimized_sampler=True, use_kv_cache=True,
                apply_repetition_penalty=True, repetition_penalty=1.2,
                apply_top_k_sampling=True, top_k=50, apply_top_p_sampling=True,
                top_p=0.95, pad_token_id=self.tokenizer.pad_token_id or 0
            ),
        }
        
        # Test each optimization individually
        for opt_name, config in optimization_configs.items():
            self.benchmark_configuration(
                self.optimized_llm, 
                config, 
                f"Optimized - {opt_name}",
                batch_size=1
            )
        
        # 3. Batch size scaling analysis
        best_config = optimization_configs["All Optimizations"]
        for batch_size in batch_sizes:
            self.benchmark_configuration(
                self.optimized_llm,
                best_config,
                f"All Optimizations (Batch {batch_size})",
                batch_size=batch_size
            )
        
        # 4. Different sequence lengths
        length_configs = [10, 20, 50, 100]
        for length in length_configs:
            length_config = GenerationConfig(
                max_length=length, use_optimized_sampler=True, use_kv_cache=True,
                apply_repetition_penalty=True, repetition_penalty=1.2,
                apply_top_k_sampling=True, top_k=50, apply_top_p_sampling=True,
                top_p=0.95, pad_token_id=self.tokenizer.pad_token_id or 0
            )
            self.benchmark_configuration(
                self.optimized_llm,
                length_config,
                f"Length {length} tokens",
                batch_size=1
            )
    
    def analyze_results(self):
        """Perform comprehensive analysis of benchmark results"""
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("📊 BENCHMARK ANALYSIS")
        print("="*80)
        
        # Summary statistics
        print("\n🏆 PERFORMANCE SUMMARY")
        print("-" * 50)
        print(f"Total configurations tested: {len(self.results)}")
        print(f"Fastest configuration: {df.loc[df['tokens_per_second'].idxmax(), 'name']}")
        print(f"  Speed: {df['tokens_per_second'].max():.2f} tokens/sec")
        print(f"Slowest configuration: {df.loc[df['tokens_per_second'].idxmin(), 'name']}")
        print(f"  Speed: {df['tokens_per_second'].min():.2f} tokens/sec")
        print(f"Average speed: {df['tokens_per_second'].mean():.2f} tokens/sec")
        print(f"Speed improvement (max vs min): {((df['tokens_per_second'].max() / df['tokens_per_second'].min()) - 1) * 100:.1f}%")
        
        # Memory analysis
        print(f"\n🧠 MEMORY USAGE")
        print("-" * 50)
        print(f"Average memory usage: {df['memory_used_mb'].mean():.1f} MB")
        print(f"Peak memory usage: {df['memory_used_mb'].max():.1f} MB")
        print(f"Memory efficient config: {df.loc[df['memory_used_mb'].idxmin(), 'name']}")
        
        # Batch size analysis
        batch_results = df[df['name'].str.contains('Batch')]
        if not batch_results.empty:
            print(f"\n📈 BATCH SIZE SCALING")
            print("-" * 50)
            for batch_size in sorted(batch_results['batch_size'].unique()):
                batch_data = batch_results[batch_results['batch_size'] == batch_size]
                if not batch_data.empty:
                    speed = batch_data['tokens_per_second'].iloc[0]
                    efficiency = speed / batch_size
                    print(f"Batch {batch_size}: {speed:.1f} tok/s (efficiency: {efficiency:.1f} tok/s per item)")
        
        # Optimization impact
        print(f"\n⚡ OPTIMIZATION IMPACT")
        print("-" * 50)
        baseline_speed = df[df['name'].str.contains('Basic LLM')]['tokens_per_second'].mean()
        
        for opt_type in ['KV Cache Only', 'All Optimizations']:
            opt_results = df[df['name'].str.contains(opt_type)]
            if not opt_results.empty:
                opt_speed = opt_results['tokens_per_second'].mean()
                improvement = ((opt_speed / baseline_speed) - 1) * 100
                print(f"{opt_type}: {improvement:+.1f}% vs baseline")
        
        return df
    
    def create_visualizations(self, df):
        """Create comprehensive visualizations"""
        print("\n📊 Generating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create comprehensive plot
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Overall performance comparison
        ax1 = plt.subplot(2, 3, 1)
        sorted_df = df.sort_values('tokens_per_second', ascending=True)
        bars = ax1.barh(range(len(sorted_df)), sorted_df['tokens_per_second'])
        ax1.set_yticks(range(len(sorted_df)))
        ax1.set_yticklabels([name[:30] + '...' if len(name) > 30 else name for name in sorted_df['name']], fontsize=8)
        ax1.set_xlabel('Tokens per Second')
        ax1.set_title('Overall Performance Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Color bars by performance
        max_speed = sorted_df['tokens_per_second'].max()
        for i, bar in enumerate(bars):
            performance_ratio = sorted_df.iloc[i]['tokens_per_second'] / max_speed
            bar.set_color(plt.cm.RdYlGn(performance_ratio))
        
        # 2. Batch size scaling
        ax2 = plt.subplot(2, 3, 2)
        batch_data = df[df['name'].str.contains('Batch')]
        if not batch_data.empty:
            batch_sizes = sorted(batch_data['batch_size'].unique())
            batch_speeds = [batch_data[batch_data['batch_size'] == bs]['tokens_per_second'].mean() for bs in batch_sizes]
            ax2.plot(batch_sizes, batch_speeds, 'o-', linewidth=3, markersize=8)
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Tokens per Second')
            ax2.set_title('Batch Size Scaling')
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log', base=2)
        
        # 3. Memory usage vs performance
        ax3 = plt.subplot(2, 3, 3)
        scatter = ax3.scatter(df['memory_used_mb'], df['tokens_per_second'], 
                           c=df['batch_size'], s=100, alpha=0.7, cmap='viridis')
        ax3.set_xlabel('Memory Usage (MB)')
        ax3.set_ylabel('Tokens per Second')
        ax3.set_title('Memory vs Performance')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Batch Size')
        
        # 4. Optimization effectiveness
        ax4 = plt.subplot(2, 3, 4)
        opt_data = df[df['name'].str.contains('Only|No Optimizations|All Optimizations')]
        if not opt_data.empty:
            opt_names = [name.split(' - ')[1] if ' - ' in name else name for name in opt_data['name']]
            opt_speeds = opt_data['tokens_per_second'].values
            wedges, texts, autotexts = ax4.pie(opt_speeds, labels=opt_names, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Optimization Technique Distribution')
        
        # 5. Time vs tokens generated
        ax5 = plt.subplot(2, 3, 5)
        ax5.scatter(df['elapsed_time'], df['total_tokens'], alpha=0.7, s=100)
        ax5.set_xlabel('Time (seconds)')
        ax5.set_ylabel('Total Tokens Generated')
        ax5.set_title('Time vs Output Volume')
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance ranking
        ax6 = plt.subplot(2, 3, 6)
        top_10 = df.nlargest(10, 'tokens_per_second')
        bars = ax6.bar(range(len(top_10)), top_10['tokens_per_second'])
        ax6.set_xticks(range(len(top_10)))
        ax6.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in top_10['name']], 
                           rotation=45, ha='right', fontsize=8)
        ax6.set_ylabel('Tokens per Second')
        ax6.set_title('Top 10 Configurations')
        ax6.grid(True, alpha=0.3)
        
        # Color bars by rank
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.RdYlGn(1 - i / len(bars)))
        
        plt.tight_layout()
        plot_path = os.path.join(self.assets_dir, f"comprehensive_benchmark_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Comprehensive plot saved to: {plot_path}")
        
        return plot_path
    
    def generate_html_report(self, df):
        """Generate comprehensive HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Benchmark Report - {self.timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
                .header {{ text-align: center; color: #333; margin-bottom: 30px; }}
                .metric-card {{ display: inline-block; margin: 10px; padding: 20px; background: #f9f9f9; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ color: #7f8c8d; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .best {{ background-color: #2ecc71; color: white; }}
                .worst {{ background-color: #e74c3c; color: white; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 LLM Performance Benchmark Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{len(df)}</div>
                        <div class="metric-label">Configurations Tested</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{df['tokens_per_second'].max():.1f}</div>
                        <div class="metric-label">Max Speed (tok/s)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{df['tokens_per_second'].mean():.1f}</div>
                        <div class="metric-label">Average Speed (tok/s)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{((df['tokens_per_second'].max() / df['tokens_per_second'].min()) - 1) * 100:.0f}%</div>
                        <div class="metric-label">Max Improvement</div>
                    </div>
                </div>
                
                <h2>📊 Detailed Results</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Configuration</th>
                            <th>Tokens/sec</th>
                            <th>Time (s)</th>
                            <th>Memory (MB)</th>
                            <th>Batch Size</th>
                            <th>Relative Performance</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        max_speed = df['tokens_per_second'].max()
        min_speed = df['tokens_per_second'].min()
        
        for _, row in df.sort_values('tokens_per_second', ascending=False).iterrows():
            relative_perf = (row['tokens_per_second'] / max_speed) * 100
            css_class = 'best' if row['tokens_per_second'] == max_speed else ('worst' if row['tokens_per_second'] == min_speed else '')
            
            html_content += f"""
                        <tr class="{css_class}">
                            <td>{row['name']}</td>
                            <td>{row['tokens_per_second']:.2f}</td>
                            <td>{row['elapsed_time']:.3f}</td>
                            <td>{row['memory_used_mb']:.1f}</td>
                            <td>{row['batch_size']}</td>
                            <td>{relative_perf:.1f}%</td>
                        </tr>
            """
        
        html_content += """
                    </tbody>
                </table>
                
                <h2>🎯 Key Insights</h2>
                <ul>
                    <li><strong>Best Overall Performance:</strong> Look for configurations with highest tokens/sec</li>
                    <li><strong>Memory Efficiency:</strong> Balance performance with memory usage for your hardware</li>
                    <li><strong>Batch Processing:</strong> Larger batches improve throughput but require more memory</li>
                    <li><strong>Optimization Impact:</strong> KV caching provides the most significant speed improvement</li>
                </ul>
                
                <h2>🔧 Recommendations</h2>
                <ul>
                    <li>For maximum speed: Use all optimizations with appropriate batch size</li>
                    <li>For memory-constrained systems: Use KV cache only with batch size 1</li>
                    <li>For production: Monitor memory usage and adjust batch size accordingly</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(self.html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 HTML report saved to: {self.html_path}")
        return self.html_path
    
    def save_results(self, df):
        """Save results in multiple formats"""
        # Save as CSV
        df.to_csv(self.csv_path, index=False)
        print(f"💾 Results saved to CSV: {self.csv_path}")
        
        # Save as JSON for dashboard
        results_json = {
            'timestamp': self.timestamp,
            'summary': {
                'total_configs': len(df),
                'fastest_config': df.loc[df['tokens_per_second'].idxmax(), 'name'],
                'fastest_speed': df['tokens_per_second'].max(),
                'slowest_config': df.loc[df['tokens_per_second'].idxmin(), 'name'],
                'slowest_speed': df['tokens_per_second'].min(),
                'average_speed': df['tokens_per_second'].mean(),
                'max_improvement': ((df['tokens_per_second'].max() / df['tokens_per_second'].min()) - 1) * 100
            },
            'detailed_results': df.to_dict('records')
        }
        
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"💾 Results saved to JSON: {self.json_path}")
        
        return self.csv_path, self.json_path
    
    def run_full_benchmark(self):
        """Run the complete benchmark suite"""
        print("🎯 Starting comprehensive benchmark...")
        
        # Run benchmarks
        self.run_comprehensive_benchmark()
        
        # Analyze results
        df = self.analyze_results()
        
        # Create visualizations
        self.create_visualizations(df)
        
        # Generate HTML report
        self.generate_html_report(df)
        
        # Save results
        self.save_results(df)
        
        print("\n" + "="*80)
        print("✅ BENCHMARK COMPLETE!")
        print("="*80)
        print(f"📊 View results: {self.html_path}")
        print(f"📈 Charts saved in: {self.assets_dir}")
        print(f"📁 Data files: {self.csv_path}")
        print("="*80)
        
        return df
import argparse

 
# Usage CLI entrypoint
def main():
    parser = argparse.ArgumentParser(description="Run Enhanced GPT Benchmarking Suite")
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["full", "analyze", "visualize", "report"],
        default="full",
        help="Mode to run: full (default), analyze, visualize, report"
    )
    
    parser.add_argument(
        "--input", 
        type=str,
        help="Path to existing benchmark result CSV file (for analyze/visualize/report modes)"
    )
    
    args = parser.parse_args()
    
    benchmark = EnhancedBenchmark()

    if args.mode == "full":
        df = benchmark.run_full_benchmark()
        print("\n🏁 Full benchmark finished!")
        
    elif args.mode == "analyze":
        if not args.input:
            raise ValueError("Please provide --input path to CSV file for analysis.")
        df = pd.read_csv(args.input)
        benchmark.results = df.to_dict("records")
        benchmark.analyze_results()
        
    elif args.mode == "visualize":
        if not args.input:
            raise ValueError("Please provide --input path to CSV file for visualization.")
        df = pd.read_csv(args.input)
        benchmark.results = df.to_dict("records")
        benchmark.create_visualizations(df)
        
    elif args.mode == "report":
        if not args.input:
            raise ValueError("Please provide --input path to CSV file for report generation.")
        df = pd.read_csv(args.input)
        benchmark.results = df.to_dict("records")
        benchmark.generate_html_report(df)

if __name__ == "__main__":
    main()
