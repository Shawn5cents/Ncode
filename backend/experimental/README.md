# Multi-Token Prediction (MTP) Accelerator

This experimental module provides implementations for improving model performance and inference speed through:
1. Multi-Token Prediction (MTP) training objective
2. Speculative decoding for faster inference

## Key Benefits

- **Improved Model Performance**: MTP training helps models learn better token dependencies by predicting multiple tokens at once
- **Faster Inference**: Speculative decoding reduces latency by making educated guesses about future tokens
- **Non-intrusive**: Can be tested alongside existing systems without disrupting them
- **Performance Tracking**: Built-in metrics to measure improvements in both training and inference

## Usage Examples

### Training with MTP

```python
from experimental.mtp_accelerator import MTPTrainer, PerformanceTracker

# Initialize trainer
trainer = MTPTrainer(
    model=your_model,
    tokenizer=your_tokenizer,
    mtp_sequence_length=4  # Number of tokens to predict at once
)

# Initialize optimizer and tracker
optimizer = torch.optim.Adam(model.parameters())
tracker = PerformanceTracker()

# Training loop
for epoch in range(num_epochs):
    start_time = time.time()
    
    for batch in dataloader:
        # Regular training step
        baseline_loss = regular_training_step(batch)
        tracker.track_training('baseline', epoch, baseline_loss, time.time() - start_time)
        
        # MTP training step
        mtp_loss = trainer.train_step(batch, optimizer)
        tracker.track_training('mtp', epoch, mtp_loss, time.time() - start_time)

# Get performance summary
summary = tracker.get_summary()
print(f"Training loss reduction: {summary['training']['loss_reduction']}")
```

### Inference with Speculative Decoding

```python
from experimental.mtp_accelerator import SpeculativeDecoder

# Initialize decoder
decoder = SpeculativeDecoder(
    model=your_model,
    tokenizer=your_tokenizer,
    draft_steps=4  # Number of tokens to speculatively generate
)

# Track inference performance
tracker = PerformanceTracker()

# Regular inference
start_time = time.time()
baseline_output = your_model.generate(input_ids, max_length=100)
tracker.track_inference('baseline', len(baseline_output), time.time() - start_time)

# Speculative decoding
start_time = time.time()
mtp_output = decoder.generate(input_ids, max_length=100)
tracker.track_inference('mtp', len(mtp_output), time.time() - start_time)

# Get performance summary
summary = tracker.get_summary()
print(f"Inference speedup: {summary['inference']['speedup']}")
```

## Implementation Details

### MTPTrainer

The `MTPTrainer` class implements the Multi-Token Prediction training objective:

1. For each position in the input sequence, predicts the next N tokens
2. Uses sliding windows to create input-output pairs
3. Computes loss across all predictions
4. Supports variable sequence lengths through padding

### SpeculativeDecoder

The `SpeculativeDecoder` class implements speculative decoding for faster inference:

1. Quickly generates draft tokens using simplified forward passes
2. Verifies predictions with a full model pass
3. Accepts verified tokens and continues generation
4. Falls back to regular generation on mismatches

### PerformanceTracker

The `PerformanceTracker` class provides metrics to measure improvements:

1. Tracks training metrics:
   - Loss reduction
   - Training duration
2. Tracks inference metrics:
   - Tokens per second
   - Speedup factor
3. Provides summary statistics for easy comparison

## Running Tests

The implementation includes comprehensive tests covering all components:

```bash
# Run all tests
pytest backend/experimental/tests/test_mtp.py

# Run specific test
pytest backend/experimental/tests/test_mtp.py::test_mtp_end_to_end
```

## Best Practices

1. **MTP Sequence Length**: Start with a small value (2-4) and increase based on your specific use case
2. **Draft Steps**: Balance between speed and accuracy - more steps mean faster inference but higher chance of mismatches
3. **Performance Monitoring**: Always use the PerformanceTracker to validate improvements
4. **Gradual Integration**: Test with a small subset of your data before full deployment

## Limitations

1. MTP training requires more memory due to predicting multiple tokens
2. Speculative decoding effectiveness depends on model prediction accuracy
3. Not all models may benefit equally from these techniques

## Future Improvements

1. Adaptive MTP sequence length based on input complexity
2. Dynamic draft steps based on prediction confidence
3. Distributed training support
4. More sophisticated verification strategies for speculative decoding
