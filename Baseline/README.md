# Baseline Fine-tuning Experiments

QLoRA fine-tuning experiments on Phi-2 and Mistral-7B models with standard configurations.

## Experiments

### 1. Phi-2 Fine-tuning (`Fine_Tune_Phi_2.ipynb`)
- **Model**: microsoft/phi-2 (2.7B parameters)
- **Dataset**: OpenAssistant Guanaco
- **Samples**: 1,000

### 2. Mistral-7B Fine-tuning (`mistral7b_300samples.ipynb`)
- **Model**: mistralai/Mistral-7B-v0.1 (7B parameters)  
- **Dataset**: OpenAssistant Guanaco  
- **Samples**: 300

## Key Results

| Model | Samples | LoRA Config | Memory Usage | Training Time | Final Loss |
|-------|---------|-------------|--------------|---------------|------------|
| Phi-2 | 1000 | r=16, α=32 | 6.2 GB | 15 min | 1.42 |
| Mistral-7B | 300 | r=8, α=16 | 11.3 GB | 20 min | 1.89 |

## Technical Configuration

### Hardware Requirements
- **GPU**: NVIDIA T4 (15GB VRAM)
- **Memory Usage**: 6-12 GB depending on model size
- **Training Time**: 15-30 minutes per experiment

### Software Stack
- **Framework**: Transformers 4.41.2, PEFT 0.11.1, TRL 0.8.6
- **Quantization**: bitsandbytes 4-bit QLoRA
- **Environment**: Google Colab compatible

## Memory Guidelines

Estimated GPU memory usage with 4-bit quantization:

```
Phi-2 (4-bit): 1.5 GB base + 4.7 GB training = 6.2 GB total
Mistral-7B (4-bit): 3.5 GB base + 7.8 GB training = 11.3 GB total
```

## Standard Configuration

- **Batch size**: 1
- **Sequence length**: 512
- **LoRA rank**: 8-16 (model dependent)
- **Learning rate**: 2e-4
- **Gradient accumulation**: 8-16 steps

## Usage

1. Open desired notebook in Google Colab
2. Install required dependencies
3. Run cells sequentially
4. Monitor GPU memory usage
5. Save trained models for comparison

## Notes

- Uses OpenAssistant Guanaco dataset
- Optimized for T4 GPU (15GB VRAM)
- Models saved for comparison with advanced techniques 