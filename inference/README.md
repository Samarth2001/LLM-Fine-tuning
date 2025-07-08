# Inference Optimization

Performance optimization benchmarks for fine-tuned CodeGen models using quantization, compilation, and batch processing.

## Optimization Techniques

### 1. LoRA Weight Merging
- Merges LoRA adapter weights into base model
- Reduces inference overhead from adapter lookups
- Maintains model quality while improving speed

### 2. 8-bit Quantization
- Uses BitsAndBytesConfig for 8-bit inference
- Reduces memory usage significantly
- Maintains inference quality with minimal degradation

### 3. BetterTransformer
- PyTorch's optimized transformer implementation
- Faster attention computation
- Automatic kernel fusion optimizations

### 4. Torch Compile
- JIT compilation with `torch.compile`
- Mode: "reduce-overhead" for inference optimization
- Requires warmup but provides significant speedup

### 5. Optimized Generation Config
- Tuned generation parameters for speed
- Uses greedy decoding (num_beams=1)
- Optimized for code generation tasks

### 6. Batch Processing
- Tests multiple batch sizes (1, 2, 4)
- Measures throughput improvements
- Compares per-prompt vs total throughput

## Test Configuration

### Test Prompts
```python
[
    "# Write a function to calculate factorial\n",
    "# Create a class for a binary tree\n",
    "# Implement bubble sort\n",
    "# Check if a string is palindrome\n"
]
```

### Generation Parameters
- **Max New Tokens**: 100
- **Temperature**: 0.1
- **Top-p**: 0.95
- **Sampling**: True
- **Early Stopping**: Enabled

## Benchmark Metrics

- **Tokens per Second**: Primary performance metric
- **Memory Usage**: GPU memory consumption
- **Speedup**: Relative to baseline performance
- **Latency**: Time per generation request

## Hardware Requirements

- **GPU**: NVIDIA T4 (14GB) or equivalent
- **Memory**: Varies by optimization (2-6GB)
- **CUDA**: Compatible GPU required

## Files

- `Inference_Optimization_Notebook_.ipynb`: Main optimization notebook
- Loads models from Google Drive or local paths

## Model Loading

1. **Direct Loading**: Pre-merged fine-tuned model
2. **LoRA Loading**: Base model + adapter weights
3. **Fallback**: Error handling for different formats

## Expected Results

### Performance Improvements
- **LoRA Merge**: 10-20% speedup
- **8-bit Quantization**: 40-60% memory reduction
- **BetterTransformer**: 15-25% speedup
- **Torch Compile**: 30-50% speedup (after warmup)
- **Batch Processing**: 2-3x throughput for larger batches

### Memory Optimization
- Baseline: ~6GB
- 8-bit: ~2-3GB
- Optimized config: Minimal additional overhead

## Usage

1. Load fine-tuned model from Google Drive
2. Run baseline performance test
3. Apply optimizations sequentially
4. Compare results and select best configuration
5. Deploy optimized model for production

## Dependencies

```
transformers
peft
accelerate
bitsandbytes
torch
optimum
numpy
```

## Configuration Options

```python
RUN_8BIT_TEST = True        # Enable 8-bit quantization test
RUN_COMPILE_TEST = True     # Enable torch.compile test
RUN_BATCH_TEST = True       # Enable batch processing test
```

## Notes

- Supports both LoRA and merged models
- Compatible with Google Colab
- Includes error handling for different model formats 