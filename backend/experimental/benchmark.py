"""
Benchmark script to measure MTP training and inference improvements
"""

import torch
import time
from typing import List, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt
from .mtp_accelerator import MTPTrainer, SpeculativeDecoder, PerformanceTracker

class BenchmarkDataset:
    """Generates synthetic data for benchmarking"""
    
    def __init__(self, vocab_size: int = 32000, seq_length: int = 512):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
    
    def generate_batch(self, batch_size: int) -> torch.Tensor:
        """Generate random token sequences"""
        return torch.randint(1, self.vocab_size, (batch_size, self.seq_length))

def run_training_benchmark(
    model,
    tokenizer,
    dataset: BenchmarkDataset,
    batch_sizes: List[int] = [1, 2, 4, 8],
    mtp_lengths: List[int] = [2, 4, 8],
    num_steps: int = 100
) -> dict:
    """
    Benchmark MTP training against baseline
    
    Args:
        model: The model to benchmark
        tokenizer: The tokenizer
        dataset: Benchmark dataset
        batch_sizes: List of batch sizes to test
        mtp_lengths: List of MTP sequence lengths to test
        num_steps: Number of training steps per configuration
        
    Returns:
        Dictionary containing benchmark results
    """
    results = {
        'batch_sizes': batch_sizes,
        'mtp_lengths': mtp_lengths,
        'configurations': []
    }
    
    tracker = PerformanceTracker()
    
    for batch_size in batch_sizes:
        for mtp_length in mtp_lengths:
            print(f"\nBenchmarking batch_size={batch_size}, mtp_length={mtp_length}")
            
            # Initialize trainer
            trainer = MTPTrainer(model, tokenizer, mtp_sequence_length=mtp_length)
            optimizer = torch.optim.Adam(model.parameters())
            
            # Training loop
            baseline_times = []
            mtp_times = []
            
            for step in range(num_steps):
                # Generate batch
                input_ids = dataset.generate_batch(batch_size)
                
                # Baseline training
                start_time = time.time()
                outputs = model(input_ids)
                loss = outputs.logits.mean()  # Simplified loss for benchmarking
                loss.backward()
                optimizer.step()
                baseline_time = time.time() - start_time
                baseline_times.append(baseline_time)
                tracker.track_training('baseline', step, loss.item(), baseline_time)
                
                # MTP training
                start_time = time.time()
                mtp_loss = trainer.train_step(input_ids, optimizer)
                mtp_time = time.time() - start_time
                mtp_times.append(mtp_time)
                tracker.track_training('mtp', step, mtp_loss, mtp_time)
            
            # Record results
            results['configurations'].append({
                'batch_size': batch_size,
                'mtp_length': mtp_length,
                'baseline_avg_time': sum(baseline_times) / len(baseline_times),
                'mtp_avg_time': sum(mtp_times) / len(mtp_times),
                'speedup': sum(baseline_times) / sum(mtp_times)
            })
    
    return results

def run_inference_benchmark(
    model,
    tokenizer,
    dataset: BenchmarkDataset,
    input_lengths: List[int] = [64, 128, 256],
    draft_steps: List[int] = [2, 4, 8],
    num_runs: int = 50
) -> dict:
    """
    Benchmark speculative decoding against baseline
    
    Args:
        model: The model to benchmark
        tokenizer: The tokenizer
        dataset: Benchmark dataset
        input_lengths: List of input sequence lengths to test
        draft_steps: List of draft steps to test
        num_runs: Number of inference runs per configuration
        
    Returns:
        Dictionary containing benchmark results
    """
    results = {
        'input_lengths': input_lengths,
        'draft_steps': draft_steps,
        'configurations': []
    }
    
    tracker = PerformanceTracker()
    
    for input_length in input_lengths:
        for num_draft_steps in draft_steps:
            print(f"\nBenchmarking input_length={input_length}, draft_steps={num_draft_steps}")
            
            # Initialize decoder
            decoder = SpeculativeDecoder(model, tokenizer, draft_steps=num_draft_steps)
            
            baseline_times = []
            spec_times = []
            
            for _ in range(num_runs):
                # Generate input
                input_ids = dataset.generate_batch(1)[:, :input_length]
                
                # Baseline generation
                start_time = time.time()
                with torch.no_grad():
                    baseline_output = model.generate(input_ids, max_length=input_length + 50)
                baseline_time = time.time() - start_time
                baseline_times.append(baseline_time)
                tracker.track_inference('baseline', len(baseline_output[0]), baseline_time)
                
                # Speculative decoding
                start_time = time.time()
                spec_output = decoder.generate(input_ids, max_length=input_length + 50)
                spec_time = time.time() - start_time
                spec_times.append(spec_time)
                tracker.track_inference('mtp', len(spec_output), spec_time)
            
            # Record results
            results['configurations'].append({
                'input_length': input_length,
                'draft_steps': num_draft_steps,
                'baseline_avg_time': sum(baseline_times) / len(baseline_times),
                'speculative_avg_time': sum(spec_times) / len(spec_times),
                'speedup': sum(baseline_times) / sum(spec_times)
            })
    
    return results

def plot_results(training_results: dict, inference_results: dict, save_dir: Path):
    """Generate plots from benchmark results"""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training plots
    plt.figure(figsize=(10, 6))
    for config in training_results['configurations']:
        plt.scatter(config['batch_size'], config['speedup'], 
                   label=f"MTP Length {config['mtp_length']}")
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup Factor')
    plt.title('MTP Training Speedup vs Batch Size')
    plt.legend()
    plt.savefig(save_dir / 'training_speedup.png')
    plt.close()
    
    # Inference plots
    plt.figure(figsize=(10, 6))
    for config in inference_results['configurations']:
        plt.scatter(config['input_length'], config['speedup'],
                   label=f"Draft Steps {config['draft_steps']}")
    plt.xlabel('Input Sequence Length')
    plt.ylabel('Speedup Factor')
    plt.title('Speculative Decoding Speedup vs Input Length')
    plt.legend()
    plt.savefig(save_dir / 'inference_speedup.png')
    plt.close()

def main():
    """Run benchmarks and save results"""
    # Initialize components
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    dataset = BenchmarkDataset()
    results_dir = Path("benchmark_results")
    
    print("\nRunning training benchmark...")
    training_results = run_training_benchmark(model, tokenizer, dataset)
    
    print("\nRunning inference benchmark...")
    inference_results = run_inference_benchmark(model, tokenizer, dataset)
    
    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "training_results.json", "w") as f:
        json.dump(training_results, f, indent=2)
    
    with open(results_dir / "inference_results.json", "w") as f:
        json.dump(inference_results, f, indent=2)
    
    # Generate plots
    plot_results(training_results, inference_results, results_dir)
    
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    main()
