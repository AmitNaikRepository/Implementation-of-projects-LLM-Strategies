# 💾 Memory Management for LLMs Strategy

> **Strategy #S of 10**: Train Massive Models on Consumer Hardware with Smart Memory Optimization

## 📋 Executive Summary

Memory is the #1 bottleneck in LLM training and inference. We show you how to **train 70B models on 24GB GPUs** and run inference that would normally require 80GB on just 16GB through advanced memory management techniques.

### The Impact

| Model | Naive Approach | Our Techniques | GPU Needed |
|-------|---------------|----------------|------------|
| **7B Training** | 28GB VRAM | 8GB VRAM | RTX 3090 ✅ |
| **13B Training** | 52GB VRAM | 16GB VRAM | RTX 4090 ✅ |
| **70B Training** | 280GB VRAM | 48GB VRAM | 2x A6000 ✅ |
| **7B Inference** | 14GB VRAM | 4GB VRAM | Any GPU ✅ |

---

## 🎯 The Problem

### Why Models Don't Fit in Memory

```
Training Llama-2-7B (Naive Approach):

Model Weights:        7B × 2 bytes (FP16)     = 14GB
Gradients:            7B × 2 bytes             = 14GB
Optimizer States:     7B × 8 bytes (Adam)     = 56GB
Activations:          ~4GB (batch size 4)     = 4GB
────────────────────────────────────────────────────
Total:                                         = 88GB

Your GPU: RTX 4090 = 24GB
Problem: Need 88GB, have 24GB ❌
Solution: Can't train... or can we? ✅
```

### Common Scenarios

```
❌ Scenario 1: "Can't fit model in memory"
   Error: CUDA out of memory
   Trying to: Train 13B model on RTX 4090
   Naive needs: 52GB, Have: 24GB

❌ Scenario 2: "Training is extremely slow"  
   Trying to: Train with batch size 1
   Problem: GPU underutilized, terrible throughput

❌ Scenario 3: "Inference too slow/expensive"
   Trying to: Run 70B model for production
   Naive needs: 8x A100 ($16/hour)
   
❌ Scenario 4: "Multi-GPU not helping"
   Have: 2x GPUs
   Using: Data parallel (doesn't reduce memory per GPU)
   Problem: Still running out of memory
```

---

## 💡 Memory Optimization Techniques

### The Memory Reduction Stack

```
┌────────────────────────────────────────────────────────┐
│         Memory Optimization Hierarchy                  │
├────────────────────────────────────────────────────────┤
│                                                        │
│ Level 1: Mixed Precision (FP16/BF16)                  │
│ ├─ Reduction: 50% (FP32 → FP16)                       │
│ ├─ Impact: Free speed boost                           │
│ └─ Difficulty: Easy (one flag)                        │
│                                                        │
│ Level 2: Gradient Accumulation                        │
│ ├─ Reduction: N/A (enables larger effective batch)    │
│ ├─ Impact: Allows batch size 1 with good training     │
│ └─ Difficulty: Easy                                   │
│                                                        │
│ Level 3: Gradient Checkpointing                       │
│ ├─ Reduction: 60-80% activations                      │
│ ├─ Impact: Trade compute for memory                   │
│ └─ Difficulty: Easy (one flag)                        │
│                                                        │
│ Level 4: LoRA/QLoRA                                   │
│ ├─ Reduction: 99% trainable params                    │
│ ├─ Impact: Massive memory savings                     │
│ └─ Difficulty: Easy                                   │
│                                                        │
│ Level 5: 8-bit Optimizers                             │
│ ├─ Reduction: 75% optimizer memory                    │
│ ├─ Impact: 56GB → 14GB for Adam                      │
│ └─ Difficulty: Easy (bitsandbytes)                    │
│                                                        │
│ Level 6: Quantization (4-bit/8-bit)                   │
│ ├─ Reduction: 75% model weights                       │
│ ├─ Impact: 14GB → 3.5GB for 7B model                 │
│ └─ Difficulty: Medium                                 │
│                                                        │
│ Level 7: DeepSpeed ZeRO                               │
│ ├─ Reduction: Split optimizer/gradients/params        │
│ ├─ Impact: Linear scaling across GPUs                 │
│ └─ Difficulty: Medium-Hard                            │
│                                                        │
│ Level 8: CPU/Disk Offloading                          │
│ ├─ Reduction: Unlimited (use RAM/SSD)                 │
│ ├─ Impact: Slower but enables huge models             │
│ └─ Difficulty: Medium                                 │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## 🔬 Implementation Guide

### Technique 1: Mixed Precision Training

```python
"""
Mixed Precision: FP16/BF16 instead of FP32
Instant 50% memory reduction + faster training
"""

from transformers import TrainingArguments

# Bad: FP32 (default, memory hungry)
bad_args = TrainingArguments(
    output_dir="./output",
    # No precision specified = FP32 = 2x memory
)

# Good: FP16 (half memory, faster)
good_args = TrainingArguments(
    output_dir="./output",
    fp16=True,  # Use FP16 mixed precision
    # Saves 50% memory instantly!
)

# Best: BF16 (if supported, more stable than FP16)
best_args = TrainingArguments(
    output_dir="./output",
    bf16=True,  # Use BF16 (better numerical stability)
    # Requires: Ampere GPUs (RTX 30xx/40xx, A100)
)

"""
Impact on 7B model:
- FP32: 28GB (weights + gradients)
- FP16: 14GB (weights + gradients)
- Savings: 14GB = Can now fit on 24GB GPU!
"""
```

### Technique 2: Gradient Accumulation

```python
"""
Gradient Accumulation: Simulate large batch with small batches
Key for memory-constrained training
"""

# Bad: Large batch (OOM on 24GB GPU)
bad_args = TrainingArguments(
    per_device_train_batch_size=16,  # OOM!
    gradient_accumulation_steps=1,
    # Effective batch: 16 (won't fit in memory)
)

# Good: Small batch + accumulation
good_args = TrainingArguments(
    per_device_train_batch_size=1,      # Tiny batch
    gradient_accumulation_steps=16,     # Accumulate 16 steps
    # Effective batch: 1 × 16 = 16 (same as above!)
    # Memory: Only needs 1 sample at a time
)

"""
How it works:
1. Forward pass on sample 1 → compute loss
2. Backward pass → accumulate gradients (don't update)
3. Forward pass on sample 2 → compute loss  
4. Backward pass → accumulate more gradients
5. ... repeat 16 times ...
6. Update weights with accumulated gradients

Result: Same training quality, 16x less memory!
"""
```

### Technique 3: Gradient Checkpointing

```python
"""
Gradient Checkpointing: Trade compute for memory
Recompute activations during backward pass instead of storing
"""

from transformers import TrainingArguments

# Bad: Store all activations (memory hungry)
bad_args = TrainingArguments(
    gradient_checkpointing=False,
    # Stores activations for all layers: 4-8GB
)

# Good: Checkpoint activations (save memory)
good_args = TrainingArguments(
    gradient_checkpointing=True,  # Save 60-80% activation memory
    # Only store checkpoints, recompute rest
)

"""
Memory saved on 7B model:
- Without: ~6GB activations
- With: ~1.5GB activations
- Savings: 4.5GB

Cost: 20-30% slower training (acceptable tradeoff)
"""
```

### Technique 4: QLoRA (4-bit Training)

```python
"""
QLoRA: Train in 4-bit with LoRA adapters
The killer technique - enables 70B on 24GB GPU!
"""

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # Load model in 4-bit
    bnb_4bit_quant_type="nf4",             # Use NF4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in BF16
    bnb_4bit_use_double_quant=True,        # Double quantization
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA configuration (only train small adapters)
lora_config = LoraConfig(
    r=16,                    # LoRA rank
    lora_alpha=32,           # Scaling
    target_modules=[         # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Print memory usage
model.print_trainable_parameters()
# Output: trainable params: 4.2M || all params: 6.7B || trainable%: 0.06%

"""
Memory breakdown (7B model):
- Full FP16: 14GB
- 4-bit quantized: 3.5GB (75% reduction!)
- LoRA adapters: 0.05GB
- Gradients (only adapters): 0.05GB
- Optimizer (only adapters): 0.2GB
────────────────────────────────
Total: ~4GB (vs 88GB naive!)

Can now train 7B on 8GB GPU! 🎉
Can train 13B on 16GB GPU!
Can train 70B on 48GB GPU!
"""
```

### Technique 5: 8-bit Optimizers

```python
"""
8-bit Optimizers: Reduce optimizer memory by 75%
Especially important for Adam (stores 2 moments per parameter)
"""

from transformers import TrainingArguments

# Bad: 32-bit Adam optimizer
bad_args = TrainingArguments(
    optim="adamw_torch",  # 32-bit optimizer
    # Adam state: 8 bytes per param
    # 7B model: 7B × 8 = 56GB!
)

# Good: 8-bit Adam optimizer
good_args = TrainingArguments(
    optim="adamw_8bit",  # 8-bit optimizer via bitsandbytes
    # Adam state: 2 bytes per param  
    # 7B model: 7B × 2 = 14GB
    # Savings: 42GB!
)

"""
Optimizer memory comparison (7B model):
- AdamW 32-bit: 56GB
- AdamW 8-bit: 14GB
- SGD (no momentum): 0GB (but worse convergence)

8-bit Adam gives same results, 75% less memory!
"""
```

### Technique 6: DeepSpeed ZeRO

```python
"""
DeepSpeed ZeRO: Split memory across GPUs
Essential for multi-GPU training of large models
"""

# DeepSpeed ZeRO Stage 1: Shard optimizer states
zero_stage1_config = {
    "train_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 1,  # Shard optimizer states only
        # Each GPU: Full model + gradients + 1/N optimizer
    }
}

# DeepSpeed ZeRO Stage 2: Shard optimizer + gradients
zero_stage2_config = {
    "train_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,  # Shard optimizer + gradients
        # Each GPU: Full model + 1/N gradients + 1/N optimizer
        # Best for most use cases
    }
}

# DeepSpeed ZeRO Stage 3: Shard everything
zero_stage3_config = {
    "train_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,  # Shard model + gradients + optimizer
        # Each GPU: 1/N model + 1/N gradients + 1/N optimizer
        # Maximum memory reduction
        "offload_param": {
            "device": "cpu"  # Optional: offload to CPU RAM
        }
    }
}

# Launch with DeepSpeed
# deepspeed --num_gpus=2 train.py --deepspeed zero_stage2_config.json

"""
Memory per GPU (7B model on 2 GPUs):

Without DeepSpeed:
├─ GPU 0: 88GB (OOM!)
└─ GPU 1: 88GB (OOM!)

ZeRO Stage 1 (optimizer sharding):
├─ GPU 0: 60GB (14GB model + 14GB grad + 32GB optimizer)
└─ GPU 1: 60GB (14GB model + 14GB grad + 32GB optimizer)

ZeRO Stage 2 (optimizer + gradient sharding):  
├─ GPU 0: 39GB (14GB model + 7GB grad + 18GB optimizer)
└─ GPU 1: 39GB (14GB model + 7GB grad + 18GB optimizer)

ZeRO Stage 3 (everything sharded):
├─ GPU 0: 23GB (7GB model + 7GB grad + 9GB optimizer)
└─ GPU 1: 23GB (7GB model + 7GB grad + 9GB optimizer)

✅ Now fits on 2x 24GB GPUs!
"""
```

### Technique 7: CPU Offloading

```python
"""
CPU Offloading: Use RAM when VRAM runs out
Slower but enables training massive models
"""

from accelerate import Accelerator

# Configure offloading
accelerator = Accelerator(
    mixed_precision="fp16",
    cpu=True,  # Enable CPU offloading
)

# Or with DeepSpeed
deepspeed_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",        # Offload optimizer to RAM
            "pin_memory": True      # Faster transfer
        },
        "offload_param": {
            "device": "cpu",        # Offload parameters to RAM
            "pin_memory": True
        }
    }
}

"""
Memory hierarchy:
├─ VRAM (24GB): Active computations
├─ RAM (64GB): Offloaded optimizer/params
└─ Disk (1TB): Emergency overflow (very slow)

Example: Training 70B model on 24GB GPU
├─ Active layer in VRAM: 3GB
├─ Rest of model in RAM: 35GB
├─ Optimizer in RAM: 70GB
└─ Total system requirement: 24GB VRAM + 128GB RAM

Slower? Yes (2-3x)
Possible? Yes! (vs impossible before)
"""
```

---

## 📊 Memory Reduction Examples

### Example 1: Train Llama-2-7B on RTX 3090 (24GB)

```python
"""
Complete configuration for 7B model on 24GB GPU
"""

from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model
import torch

# Step 1: Load model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Step 2: Add LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Step 3: Configure training with all optimizations
training_args = TrainingArguments(
    output_dir="./output",
    
    # Mixed precision
    bf16=True,
    
    # Small batch + accumulation
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # Effective batch: 16
    
    # Gradient checkpointing
    gradient_checkpointing=True,
    
    # 8-bit optimizer
    optim="adamw_8bit",
    
    # Other settings
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=500,
)

"""
Memory breakdown:
├─ Model (4-bit): 3.5GB
├─ LoRA adapters: 0.05GB
├─ Gradients (adapters only): 0.05GB
├─ Optimizer (8-bit, adapters): 0.2GB
├─ Activations (checkpointed): 1.5GB
├─ Batch (size 1): 0.5GB
└─ Buffer: 1GB
────────────────────────────
Total: ~7GB / 24GB available ✅

Can even increase batch size to 2-3!
"""
```

### Example 2: Train Llama-2-70B on 2x RTX A6000 (96GB total)

```python
"""
Train 70B model on 2x 48GB GPUs using DeepSpeed ZeRO-3
"""

# deepspeed_config.json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 8,
    "gradient_clipping": 1.0,
    
    "fp16": {
        "enabled": true
    },
    
    "zero_optimization": {
        "stage": 3,  # Shard everything
        
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    }
}

# Launch training
# deepspeed --num_gpus=2 train.py --deepspeed deepspeed_config.json

"""
Memory per GPU:
├─ Model params (sharded): 17.5GB (70B × 2bytes ÷ 2 GPUs ÷ 4)
├─ Gradients (sharded): 8.8GB
├─ Optimizer (offloaded): 0GB (in CPU RAM)
├─ Activations: 12GB
├─ Working memory: 8GB
└─ Total: ~46GB / 48GB available ✅

Offloaded to CPU RAM:
└─ Optimizer states: 140GB (needs 256GB RAM)

Training speed: 2-3x slower than full GPU, but possible!
"""
```

### Example 3: Inference Llama-2-70B on Single RTX 4090 (24GB)

```python
"""
Run 70B inference on 24GB GPU with aggressive quantization
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load in 4-bit with CPU offload
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",             # Auto distribute
    load_in_4bit=True,             # 4-bit quantization
    max_memory={0: "22GB", "cpu": "60GB"},  # Use 22GB VRAM + 60GB RAM
    offload_folder="offload",      # Disk offload if needed
    offload_state_dict=True,
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")

# Generate
prompt = "Explain quantum computing:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

"""
Memory distribution:
├─ VRAM (22GB): Active layers + KV cache
├─ RAM (60GB): Offloaded layers
└─ Disk (if needed): Emergency overflow

Performance:
├─ Speed: ~5-10 tokens/sec (vs 50 on 8xA100)
├─ Usable? Yes, for development/testing
└─ Production? Use smaller model or more GPUs
"""
```

---

## 🎯 Quick Reference Guide

### Memory Optimization Checklist

```
For Training:
✅ Enable mixed precision (FP16/BF16)
✅ Use gradient accumulation (batch size 1-2)
✅ Enable gradient checkpointing
✅ Use QLoRA for large models (>7B)
✅ Use 8-bit optimizer
✅ Multi-GPU? Use DeepSpeed ZeRO-2 or ZeRO-3
✅ Still OOM? Offload optimizer to CPU
✅ Still OOM? Offload parameters to CPU

For Inference:
✅ Use 4-bit or 8-bit quantization
✅ Smaller batch sizes
✅ Flash Attention (if supported)
✅ KV cache optimization
✅ Multi-GPU? Use model parallelism
✅ Still OOM? CPU offloading
```

### Troubleshooting OOM Errors

```
Error: "CUDA out of memory"

Try in order:
1. Enable mixed precision (fp16=True)
2. Reduce batch size to 1
3. Increase gradient accumulation
4. Enable gradient checkpointing
5. Use QLoRA instead of full finetuning
6. Use 8-bit optimizer
7. Add more GPUs with DeepSpeed
8. Enable CPU offloading
9. Reduce model size or sequence length
10. Last resort: Use smaller model
```

---

## 💰 Cost Impact

### Training Cost Comparison

```
Train Llama-2-7B for 1 epoch:

Without Optimization:
├─ Requires: 2x A100 80GB
├─ Cost: $4/hour × 10 hours = $40
└─ Why 2 GPUs: Model doesn't fit on 1

With Our Techniques:
├─ Requires: 1x RTX 4090 24GB
├─ Cost: $0.50/hour × 12 hours = $6
└─ Why works: QLoRA + optimizations

Savings: $40 → $6 (85% reduction!)
```

---

## 📞 Next Steps

**Ready to train models you thought were impossible?**

📧 Email: your-email@company.com  
💼 Schedule: [Calendar Link]

---

<div align="center">

**Strategy #S: Memory Management**  
*Train 70B on 24GB | 85% Cost Savings | No Limits*

💾 Smart Memory | 🚀 Big Models | 💰 Small GPUs

</div>
