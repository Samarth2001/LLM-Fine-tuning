# LLM Fine-tuning Experiments

Systematic evaluation of parameter-efficient fine-tuning methods for 7B language models on consumer hardware.

## Overview

This repository contains experiments and benchmarks for fine-tuning large language models (7B parameters) using QLoRA on limited compute resources (Google Colab T4 GPU, 15GB VRAM). The focus is on practical implementations that balance memory efficiency with model performance.

## Key Results

| Model | Dataset | Samples | LoRA Config | Memory Usage | Training Time | Final Loss |
|-------|---------|---------|-------------|--------------|---------------|------------|
| microsoft/phi-2 | openassistant-guanaco | 1000 | r=16, α=32 | 6.2 GB | 15 min | 1.42 |
| mistralai/Mistral-7B-v0.1 | openassistant-guanaco | 300 | r=8, α=16 | 11.3 GB | 20 min | 1.89 |
| mistralai/Mistral-7B-v0.1 | openassistant-guanaco | 500 | r=16, α=32 | 13.1 GB | 30 min | 1.65 |
| meta-llama/Llama-2-7b-hf | openassistant-guanaco | 500 | r=8, α=16 | 12.8 GB | 28 min | 1.72 |

<!-- ## Repository Structure

```
├── 01-basic-finetuning/      # Baseline experiments and scaling tests
├── 02-model-comparison/      # Comparative analysis across 7B models
├── 03-dataset-experiments/   # Task-specific fine-tuning evaluations
├── 04-advanced-techniques/   # Multi-LoRA and DPO implementations
├── 05-optimization/          # Quantization and inference optimization
├── 06-production/           # Production-ready training pipelines
├── experiments-archive/      # Historical experiments and ablations
└── utils/                   # Shared utilities and setup scripts
``` -->

## Technical Stack

- **Framework**: Transformers 4.41.2, PEFT 0.11.1, TRL 0.8.6
- **Quantization**: bitsandbytes 4-bit QLoRA
- **Hardware**: NVIDIA T4 GPU (15GB VRAM)
- **Models**: Mistral-7B, Llama-2-7B, Gemma-7B, Phi-2
- **Datasets**: OpenAssistant, CodeAlpaca, GSM8K

 

## Memory Requirements

Estimated GPU memory usage for 7B models with 4-bit quantization:

```
Base model (4-bit): 3.5 GB
LoRA parameters (r=16): 0.5 GB
Training overhead: 7-10 GB
Total: 11-14 GB
```

## Configuration Guidelines

Recommended settings for T4 GPU (15GB VRAM):
- **Batch size**: 1
- **Sequence length**: 512
- **LoRA rank**: 8-16
- **Dataset size**: 300-1000 samples
- **Gradient accumulation**: 8-16 steps

 
## License

MIT License. See LICENSE file for details.


## References

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [OpenAssistant Conversations Dataset](https://huggingface.co/datasets/OpenAssistant/oasst1)