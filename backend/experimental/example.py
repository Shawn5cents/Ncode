"""
Example script demonstrating MTP accelerator usage with a real model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from pathlib import Path
from .mtp_accelerator import MTPTrainer, SpeculativeDecoder, PerformanceTracker

def load_model_and_tokenizer():
    """Load pretrained model and tokenizer"""
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    return model, tokenizer

def prepare_sample_data(tokenizer, num_samples=10):
    """Prepare sample text for demonstration"""
    samples = [
        "The quick brown fox",
        "In a world where technology",
        "Once upon a time in",
        "The future of artificial intelligence",
        "Deep in the heart of",
        "On a bright summer morning",
        "As the sun set over",
        "The ancient civilization of",
        "Through the mists of time",
        "Beyond the boundaries of"
    ]
    
    # Tokenize samples
    encodings = tokenizer(
        samples[:num_samples],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    
    return encodings.input_ids

def demonstrate_mtp_training(model, tokenizer, input_ids):
    """Demonstrate MTP training process"""
    print("\nDemonstrating MTP Training...")
    
    # Initialize components
    trainer = MTPTrainer(
        model=model,
        tokenizer=tokenizer,
        mtp_sequence_length=4
    )
    optimizer = torch.optim.Adam(model.parameters())
    tracker = PerformanceTracker()
    
    # Training demonstration
    num_steps = 5
    for step in range(num_steps):
        print(f"\nStep {step + 1}/{num_steps}")
        
        # Baseline training
        start_time = time.time()
        outputs = model(input_ids)
        loss = outputs.logits.mean()
        loss.backward()
        optimizer.step()
        baseline_time = time.time() - start_time
        print(f"Baseline - Loss: {loss.item():.4f}, Time: {baseline_time:.4f}s")
        tracker.track_training('baseline', step, loss.item(), baseline_time)
        
        # MTP training
        start_time = time.time()
        mtp_loss = trainer.train_step(input_ids, optimizer)
        mtp_time = time.time() - start_time
        print(f"MTP - Loss: {mtp_loss:.4f}, Time: {mtp_time:.4f}s")
        tracker.track_training('mtp', step, mtp_loss, mtp_time)
    
    # Show summary
    summary = tracker.get_summary()
    print("\nTraining Summary:")
    print(f"Loss Reduction: {summary['training']['loss_reduction']}")

def demonstrate_speculative_decoding(model, tokenizer, input_ids):
    """Demonstrate speculative decoding for faster inference"""
    print("\nDemonstrating Speculative Decoding...")
    
    # Initialize components
    decoder = SpeculativeDecoder(
        model=model,
        tokenizer=tokenizer,
        draft_steps=4
    )
    tracker = PerformanceTracker()
    
    # Select a single input for demonstration
    input_text = "The future of artificial intelligence is"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    print(f"\nInput text: {input_text}")
    
    # Baseline generation
    start_time = time.time()
    with torch.no_grad():
        baseline_output = model.generate(
            input_ids,
            max_length=100,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id
        )
    baseline_time = time.time() - start_time
    baseline_text = tokenizer.decode(baseline_output[0])
    print(f"\nBaseline Output ({baseline_time:.2f}s):")
    print(baseline_text)
    tracker.track_inference('baseline', len(baseline_output[0]), baseline_time)
    
    # Speculative decoding
    start_time = time.time()
    spec_output = decoder.generate(input_ids, max_length=100)
    spec_time = time.time() - start_time
    spec_text = tokenizer.decode(torch.tensor(spec_output))
    print(f"\nSpeculative Output ({spec_time:.2f}s):")
    print(spec_text)
    tracker.track_inference('mtp', len(spec_output), spec_time)
    
    # Show summary
    summary = tracker.get_summary()
    print("\nInference Summary:")
    print(f"Speedup: {summary['inference']['speedup']}")

def main():
    """Run complete demonstration"""
    # Setup
    model, tokenizer = load_model_and_tokenizer()
    input_ids = prepare_sample_data(tokenizer)
    
    # Demonstrate training
    demonstrate_mtp_training(model, tokenizer, input_ids)
    
    # Demonstrate inference
    demonstrate_speculative_decoding(model, tokenizer, input_ids)

if __name__ == "__main__":
    main()
