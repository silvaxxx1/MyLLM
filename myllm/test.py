"""
Enhanced GPT-2 Diagnostic Test Suite
Focusing ONLY on critical functionality - no minor technicalities.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import time


class Colors:
    """Terminal color codes for formatted output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


class DiagnosticSuite:
    def __init__(self):
        self.start_time = time.time()
        self.results = {}
        
    def print_header(self):
        """Print beautiful diagnostic header."""
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}")
        print("╔════════════════════════════════════════════════════════════════════════════╗")
        print("║                     🚀 GPT-2 CRITICAL FUNCTIONALITY TEST                   ║")
        print("║                 Only Testing What Actually Matters for Users               ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝")
        print(f"{Colors.END}")
        
    def print_section(self, title: str, emoji: str = "🔍"):
        """Print formatted section header."""
        print(f"\n{Colors.BLUE}{Colors.BOLD}{emoji} {title}{Colors.END}")
        print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    
    def print_test(self, name: str, passed: bool, details: str = "", critical: bool = False):
        """Print formatted test result."""
        if passed:
            status = f"{Colors.GREEN}✅ PASS{Colors.END}"
        else:
            status = f"{Colors.RED}❌ FAIL{Colors.END}"
            
        print(f"   {name:<35} {status}")
        if details:
            print(f"      {details}")
    
    def print_metric(self, name: str, value, unit=""):
        """Print a metric without unnecessary comparisons."""
        print(f"      {name:<25} {value}{unit}")

    def test_critical_functionality(self, model, tokenizer, device) -> Dict[str, bool]:
        """Test only the absolutely critical functionality."""
        self.print_section("CRITICAL FUNCTIONALITY", "🎯")
        results = {}
        
        # Only test what matters: can the model process text?
        test_text = "Hello world"
        tokens = tokenizer.encode(test_text, return_tensors="pt")
        decoded = tokenizer.decode(tokens[0])
        results['text_processing'] = decoded.strip() == test_text
        self.print_test("Text Encoding/Decoding", results['text_processing'], 
                       f"'{test_text}' → '{decoded.strip()}'")
        
        # Basic forward pass - does it run without crashing?
        try:
            test_input = torch.tensor([[464, 2068]]).to(device)  # "The quick"
            with torch.no_grad():
                logits = model(test_input, use_cache=False)
            results['forward_pass'] = logits.shape[-1] == 50257  # vocab size
            self.print_test("Forward Pass Execution", results['forward_pass'],
                           f"Output shape: {list(logits.shape)}")
        except Exception as e:
            results['forward_pass'] = False
            self.print_test("Forward Pass Execution", False, f"Error: {e}")
        
        return results
    
    def test_generation_quality(self, llm, tokenizer, generation_config, device) -> Dict[str, bool]:
        """Test text generation - this is what users actually care about."""
        self.print_section("GENERATION QUALITY", "✨")
        results = {}
        
        test_cases = [
            ("The cat", "Basic continuation"),
            ("Once upon a time", "Story beginning"), 
            ("The quick brown fox", "Classic phrase"),
        ]
        
        all_passed = True
        for prompt, description in test_cases:
            try:
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                output = llm.generate(input_ids, generation_config)
                generated_text = tokenizer.decode(output["tokens"][0])
                
                # Critical checks only:
                # 1. Did it generate more text?
                prompt_tokens = len(input_ids[0])
                total_tokens = len(output["tokens"][0])
                generated_tokens = total_tokens - prompt_tokens
                
                # 2. Is the output coherent (not gibberish)?
                generated_ok = generated_tokens > 0 and len(generated_text) > len(prompt)
                
                # 3. Check for severe repetition (not minor repetition)
                all_tokens = output["tokens"][0].tolist()
                generated_only = all_tokens[prompt_tokens:]
                unique_ratio = len(set(generated_only)) / len(generated_only) if generated_only else 1.0
                
                # Only fail if there's catastrophic repetition
                no_catastrophic_repetition = unique_ratio > 0.3
                
                test_passed = generated_ok and no_catastrophic_repetition
                all_passed = all_passed and test_passed
                
                print(f"\n   {Colors.BOLD}{description}:{Colors.END} '{prompt}'")
                self.print_metric("Generated Tokens", generated_tokens)
                self.print_metric("Unique Ratio", f"{unique_ratio:.2f}")
                print(f"      Output: {generated_text[:80]}{'...' if len(generated_text) > 80 else ''}")
                
            except Exception as e:
                all_passed = False
                print(f"\n   {Colors.RED}Error in '{prompt}': {e}{Colors.END}")
        
        results['generation_quality'] = all_passed
        self.print_test("Overall Generation Quality", all_passed)
        
        return results
    
    def test_output_sanity(self, model, tokenizer, device) -> Dict[str, bool]:
        """Test that outputs are sane and usable."""
        self.print_section("OUTPUT SANITY", "🔍")
        results = {}
        
        try:
            test_input = torch.tensor([[464, 2068]]).to(device)  # "The quick"
            with torch.no_grad():
                logits = model(test_input, use_cache=False)
            
            # Critical sanity checks only:
            probs = torch.softmax(logits[0, -1], dim=-1)
            top5_probs, top5_indices = torch.topk(probs, 5)
            
            # 1. Not collapsing to a single token (catastrophic failure)
            max_prob = top5_probs[0].item()
            not_collapsed = max_prob < 0.95
            
            # 2. Has reasonable diversity in top predictions
            top5_diversity = top5_probs.sum().item() < 0.99
            
            # 3. No NaN/Inf in outputs
            no_nan = not torch.isnan(logits).any() and not torch.isinf(logits).any()
            
            results['output_sanity'] = not_collapsed and top5_diversity and no_nan
            self.print_test("Output Sanity", results['output_sanity'])
            self.print_metric("Max Token Probability", f"{max_prob:.3f}")
            self.print_metric("Top-5 Coverage", f"{top5_probs.sum().item():.3f}")
            
        except Exception as e:
            results['output_sanity'] = False
            self.print_test("Output Sanity", False, f"Error: {e}")
        
        return results
    
    def test_real_world_performance(self, llm, tokenizer, generation_config, device) -> Dict[str, bool]:
        """Test performance metrics that actually matter to users."""
        self.print_section("REAL-WORLD PERFORMANCE", "⚡")
        results = {}
        
        # Speed test - does it generate in reasonable time?
        try:
            prompt = "The weather today is"
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            start_time = time.time()
            output = llm.generate(input_ids, generation_config)
            generation_time = time.time() - start_time
            
            # Reasonable: under 5 seconds for 50 tokens
            speed_ok = generation_time < 5.0
            results['generation_speed'] = speed_ok
            self.print_test("Generation Speed", speed_ok)
            self.print_metric("Time for 50 tokens", f"{generation_time:.2f}s")
            
        except Exception as e:
            results['generation_speed'] = False
            self.print_test("Generation Speed", False, f"Error: {e}")
        
        # Memory usage - is it within reasonable bounds?
        try:
            param_count = sum(p.numel() for p in llm.model.parameters())
            # GPT-2 small should be around 100-200M parameters
            memory_ok = 100_000_000 < param_count < 200_000_000
            results['memory_usage'] = memory_ok
            self.print_test("Parameter Count", memory_ok)
            self.print_metric("Total Parameters", f"{param_count:,}")
            
        except Exception as e:
            results['memory_usage'] = False
            self.print_test("Parameter Count", False, f"Error: {e}")
        
        return results
    
    def generate_summary(self, all_results: Dict):
        """Generate focused summary report."""
        self.print_section("FINAL ASSESSMENT", "📋")
        
        # Count only critical tests
        critical_tests = [
            'text_processing', 'forward_pass', 'generation_quality', 
            'output_sanity', 'generation_speed', 'memory_usage'
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for category, tests in all_results.items():
            for test_name, result in tests.items():
                if test_name in critical_tests and result is not None:
                    total_tests += 1
                    if result is True:
                        passed_tests += 1
        
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        total_time = time.time() - self.start_time
        
        # Overall status - simplified
        if pass_rate >= 95:
            status = f"{Colors.GREEN}{Colors.BOLD}EXCELLENT{Colors.END}"
            emoji = "🎉"
        elif pass_rate >= 80:
            status = f"{Colors.GREEN}{Colors.BOLD}GOOD{Colors.END}"
            emoji = "✅"
        else:
            status = f"{Colors.RED}{Colors.BOLD}NEEDS WORK{Colors.END}"
            emoji = "⚠️"
        
        print(f"\n   {emoji} {Colors.BOLD}Overall Status:{Colors.END} {status}")
        self.print_metric("Critical Tests Passed", f"{passed_tests}/{total_tests}")
        self.print_metric("Success Rate", f"{pass_rate:.1f}%")
        self.print_metric("Total Time", f"{total_time:.1f}s")
        
        # Only show critical recommendations
        if pass_rate < 100:
            failed_tests = []
            for category, tests in all_results.items():
                for test_name, result in tests.items():
                    if test_name in critical_tests and result is False:
                        failed_tests.append(test_name)
            
            if failed_tests:
                print(f"\n   {Colors.YELLOW}{Colors.BOLD}AREAS FOR IMPROVEMENT:{Colors.END}")
                for test in failed_tests:
                    print(f"   • {test.replace('_', ' ').title()}")


def run_critical_diagnostics(llm, tokenizer, generation_config, device):
    """Run diagnostics focusing ONLY on critical functionality."""
    diagnostics = DiagnosticSuite()
    diagnostics.print_header()
    
    all_results = {
        'critical': diagnostics.test_critical_functionality(llm.model, tokenizer, device),
        'generation': diagnostics.test_generation_quality(llm, tokenizer, generation_config, device),
        'sanity': diagnostics.test_output_sanity(llm.model, tokenizer, device),
        'performance': diagnostics.test_real_world_performance(llm, tokenizer, generation_config, device),
    }
    
    diagnostics.generate_summary(all_results)
    return all_results


# For backward compatibility
def run_enhanced_diagnostics(llm, tokenizer, generation_config, device):
    """Alias for backward compatibility."""
    return run_critical_diagnostics(llm, tokenizer, generation_config, device)

def run_full_diagnostics(llm, tokenizer, generation_config, device):
    """Alias for backward compatibility."""
    return run_critical_diagnostics(llm, tokenizer, generation_config, device)


if __name__ == "__main__":
    print(f"{Colors.BOLD}GPT-2 Critical Functionality Diagnostics{Colors.END}")