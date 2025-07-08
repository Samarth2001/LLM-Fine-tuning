# CodeGen-2B Fine-tuning

Fine-tuning CodeGen-2B-mono on the CodeAlpaca dataset using QLoRA for Python code generation.

## Model Configuration

- **Base Model**: `Salesforce/codegen-2B-mono` (Python-focused)
- **Dataset**: `sahil2801/CodeAlpaca-20k`
- **Training Samples**: 1,500 samples
- **LoRA Rank**: 16
- **Sequence Length**: 768 tokens
- **Quantization**: 4-bit with NF4

## Training Parameters

```python
- LoRA Rank: 16
- Batch Size: 1
- Gradient Accumulation: 4 steps
- Learning Rate: 5e-5
- Epochs: 1
- Optimizer: paged_adamw_32bit
- Precision: FP16
```

## Dataset Format

The notebook uses a specialized formatting for CodeGen models:

```
# Question: {instruction}
# Input: {input}
# Solution:
{output}
```

## Hardware Requirements

- **GPU**: NVIDIA T4 (14GB) or equivalent
- **Memory Usage**: ~6GB during training
- **Training Time**: ~50 minutes for 1,500 samples

## Files

- `codegen_2B_1500samples.ipynb`: Main fine-tuning notebook
- Model outputs saved to `./codegen_finetuned_lora/`

## Results

- **Final Training Loss**: 1.0475
- **Training Speed**: 0.46 samples/second
- **Model Size**: 80MB adapter weights
- **Success**: Model generates functional Python code

## Usage

1. Install dependencies
2. Configure model and dataset parameters
3. Run training cells sequentially
4. Test the fine-tuned model
5. Save to Google Drive (optional)

## Dependencies

```
transformers==4.41.2
peft==0.11.1
accelerate==0.30.1
bitsandbytes
datasets==2.19.1
trl==0.8.6
torch
```

## Notes

- Optimized for Python code generation
- Uses CodeGen-specific tokenizer settings  
- Compatible with Google Colab 