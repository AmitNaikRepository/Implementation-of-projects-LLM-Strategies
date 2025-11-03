# Reinforcement Learning from Human Feedback (RLHF) Pipeline

Production-grade implementation of RLHF for large language model alignment.

## Overview

This repository contains a complete RLHF pipeline for aligning language models with human preferences, featuring:
- Supervised Fine-Tuning (SFT)
- Reward Model Training
- Proximal Policy Optimization (PPO)
- Direct Preference Optimization (DPO)
- Comprehensive evaluation and monitoring

## Architecture

```
┌─────────────────┐
│ Base LLM Model  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SFT Training   │ ← High-quality demonstrations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Reward Model    │ ← Human preference pairs
│   Training      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PPO/DPO RL      │ ← Policy optimization
│   Training      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Aligned Model   │
└─────────────────┘
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from rlhf import RLHFTrainer, RLHFConfig

# Configure RLHF pipeline
config = RLHFConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    sft_dataset="anthropic/hh-rlhf",
    reward_dataset="anthropic/hh-rlhf",
    output_dir="./checkpoints",
    ppo_epochs=4,
    learning_rate=1.41e-5
)

# Initialize and run training
trainer = RLHFTrainer(config)
trainer.train_sft()
trainer.train_reward_model()
trainer.train_ppo()
```

## Project Structure

```
strategies/Reinforcement_learning_LLM/
├── src/
│   ├── training/
│   │   ├── sft_trainer.py           # Supervised fine-tuning
│   │   ├── reward_trainer.py        # Reward model training
│   │   ├── ppo_trainer.py           # PPO implementation
│   │   └── dpo_trainer.py           # Direct Preference Optimization
│   ├── models/
│   │   ├── reward_model.py          # Reward model architecture
│   │   └── policy_model.py          # Policy model wrapper
│   ├── data/
│   │   ├── dataset_builder.py       # Data preprocessing
│   │   └── preference_collector.py  # Human feedback collection
│   ├── evaluation/
│   │   ├── metrics.py               # Evaluation metrics
│   │   └── benchmark.py             # Model benchmarking
│   └── utils/
│       ├── config.py                # Configuration management
│       ├── logger.py                # Logging utilities
│       └── distributed.py           # Distributed training
├── configs/
│   ├── sft_config.yaml
│   ├── reward_config.yaml
│   └── ppo_config.yaml
├── scripts/
│   ├── train_sft.py
│   ├── train_reward.py
│   ├── train_ppo.py
│   └── evaluate.py
├── tests/
│   ├── test_training.py
│   └── test_models.py
├── requirements.txt
└── README.md
```

## Core Components

### 1. Supervised Fine-Tuning (SFT)

Fine-tune the base model on high-quality demonstration data:

```python
from src.training.sft_trainer import SFTTrainer

sft_trainer = SFTTrainer(
    model_name="meta-llama/Llama-2-7b-hf",
    dataset="anthropic/hh-rlhf",
    max_length=512,
    batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_epochs=3
)

sft_model = sft_trainer.train()
```

### 2. Reward Model Training

Train a reward model to score model outputs based on human preferences:

```python
from src.training.reward_trainer import RewardTrainer

reward_trainer = RewardTrainer(
    base_model=sft_model,
    preference_dataset="anthropic/hh-rlhf",
    batch_size=8,
    learning_rate=1e-5,
    num_epochs=1
)

reward_model = reward_trainer.train()
```

### 3. PPO Training

Optimize the policy using Proximal Policy Optimization:

```python
from src.training.ppo_trainer import PPOTrainer

ppo_trainer = PPOTrainer(
    policy_model=sft_model,
    reward_model=reward_model,
    kl_penalty=0.1,
    clip_range=0.2,
    ppo_epochs=4,
    batch_size=256,
    mini_batch_size=32
)

aligned_model = ppo_trainer.train()
```

### 4. Direct Preference Optimization (Alternative to PPO)

```python
from src.training.dpo_trainer import DPOTrainer

dpo_trainer = DPOTrainer(
    model=sft_model,
    preference_dataset="anthropic/hh-rlhf",
    beta=0.1,
    learning_rate=5e-7,
    num_epochs=1
)

aligned_model = dpo_trainer.train()
```

## Configuration

### SFT Configuration (`configs/sft_config.yaml`)

```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"
  max_length: 512
  use_flash_attention: true

dataset:
  name: "anthropic/hh-rlhf"
  split: "train"
  preprocessing:
    remove_duplicates: true
    filter_length: 2048

training:
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-5
  warmup_steps: 100
  num_epochs: 3
  fp16: true
  gradient_checkpointing: true

optimization:
  optimizer: "adamw"
  weight_decay: 0.01
  max_grad_norm: 1.0
```

### Reward Model Configuration (`configs/reward_config.yaml`)

```yaml
model:
  base_model: "path/to/sft_model"
  reward_head_hidden_size: 1024
  dropout: 0.1

dataset:
  name: "anthropic/hh-rlhf"
  comparison_type: "pairwise"

training:
  batch_size: 8
  learning_rate: 1.0e-5
  num_epochs: 1
  loss_type: "ranking_loss"
  margin: 0.5
```

### PPO Configuration (`configs/ppo_config.yaml`)

```yaml
algorithm:
  kl_penalty: 0.1
  clip_range: 0.2
  value_clip_range: 0.2
  ppo_epochs: 4
  gae_lambda: 0.95
  gamma: 1.0

training:
  batch_size: 256
  mini_batch_size: 32
  learning_rate: 1.41e-5
  max_grad_norm: 1.0
  normalize_advantages: true

generation:
  max_new_tokens: 128
  temperature: 1.0
  top_k: 50
  top_p: 0.95
```

## Training Scripts

### Full RLHF Pipeline

```python
# scripts/train_full_pipeline.py
import torch
from src.training import SFTTrainer, RewardTrainer, PPOTrainer
from src.utils import RLHFConfig, setup_logging, setup_distributed

def main():
    # Setup
    config = RLHFConfig.from_yaml("configs/rlhf_config.yaml")
    logger = setup_logging(config.output_dir)
    setup_distributed()

    # Stage 1: Supervised Fine-Tuning
    logger.info("Starting SFT training...")
    sft_trainer = SFTTrainer(config.sft)
    sft_model = sft_trainer.train()
    sft_model.save_pretrained(f"{config.output_dir}/sft_model")

    # Stage 2: Reward Model Training
    logger.info("Starting reward model training...")
    reward_trainer = RewardTrainer(config.reward, base_model=sft_model)
    reward_model = reward_trainer.train()
    reward_model.save_pretrained(f"{config.output_dir}/reward_model")

    # Stage 3: PPO Training
    logger.info("Starting PPO training...")
    ppo_trainer = PPOTrainer(
        config.ppo,
        policy_model=sft_model,
        reward_model=reward_model
    )
    aligned_model = ppo_trainer.train()
    aligned_model.save_pretrained(f"{config.output_dir}/aligned_model")

    logger.info("RLHF training complete!")

if __name__ == "__main__":
    main()
```

## Evaluation

### Metrics

```python
from src.evaluation.metrics import evaluate_model

metrics = evaluate_model(
    model=aligned_model,
    test_dataset=test_data,
    metrics=["reward_score", "kl_divergence", "perplexity", "toxicity"]
)

print(f"Average Reward: {metrics['reward_score']:.3f}")
print(f"KL Divergence: {metrics['kl_divergence']:.3f}")
print(f"Perplexity: {metrics['perplexity']:.3f}")
print(f"Toxicity Score: {metrics['toxicity']:.3f}")
```

### Benchmarking

```python
from src.evaluation.benchmark import run_benchmark

results = run_benchmark(
    model=aligned_model,
    benchmarks=["mt-bench", "alpaca-eval", "truthfulqa"],
    output_dir="./eval_results"
)
```

## Production Deployment

### Model Export

```python
# Export for production
from src.utils.export import export_model

export_model(
    model=aligned_model,
    format="onnx",  # or "tensorrt", "torchscript"
    quantization="int8",
    output_path="./production/model.onnx"
)
```

### API Server

```python
from src.serving.api import create_app

app = create_app(
    model_path="./production/aligned_model",
    max_batch_size=32,
    max_concurrent_requests=100
)

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
```

## Monitoring

### Training Metrics

- Reward score progression
- KL divergence from reference model
- Policy entropy
- Value function loss
- Gradient norms

### Production Metrics

- Inference latency (p50, p95, p99)
- Throughput (requests/second)
- Reward scores on production traffic
- User satisfaction ratings

## Best Practices

### 1. Data Quality
- Use high-quality, diverse demonstration data for SFT
- Ensure preference data represents target user preferences
- Regularly update preference datasets with new feedback

### 2. Training Stability
- Use gradient clipping to prevent exploding gradients
- Monitor KL divergence to prevent policy collapse
- Implement adaptive KL penalties
- Use value function clipping in PPO

### 3. Hyperparameter Tuning
- Start with smaller learning rates (1e-6 to 1e-5)
- Tune KL penalty coefficient (0.01 to 0.5)
- Adjust PPO clip range (0.1 to 0.3)
- Experiment with batch sizes based on GPU memory

### 4. Evaluation
- Test on diverse prompts and domains
- Monitor both automated metrics and human evaluations
- Check for potential biases and failure modes
- Conduct red-teaming exercises

## Performance Optimization

### Distributed Training

```bash
# Multi-GPU training with DeepSpeed
deepspeed --num_gpus=8 scripts/train_ppo.py \
  --deepspeed configs/deepspeed_config.json \
  --config configs/ppo_config.yaml
```

### Memory Optimization

- Gradient checkpointing
- Mixed precision training (FP16/BF16)
- Flash Attention 2
- Parameter-efficient fine-tuning (LoRA, QLoRA)

## Troubleshooting

### Common Issues

**Reward Hacking**
- Solution: Increase KL penalty, use ensemble reward models

**Policy Collapse**
- Solution: Reduce learning rate, adjust clip range, warm up KL coefficient

**Out of Memory**
- Solution: Reduce batch size, enable gradient checkpointing, use smaller models

**Slow Training**
- Solution: Use distributed training, optimize data loading, use compiled models

## Citation

```bibtex
@article{rlhf2024,
  title={Production-Grade RLHF Pipeline for LLM Alignment},
  author={Your Name},
  year={2024}
}
```

## References

1. Ouyang et al. "Training language models to follow instructions with human feedback" (2022)
2. Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
3. Rafailov et al. "Direct Preference Optimization" (2023)
4. Bai et al. "Training a Helpful and Harmless Assistant with RLHF" (2022)

## License

MIT License

## Contact

For questions and support, please open an issue on GitHub.
