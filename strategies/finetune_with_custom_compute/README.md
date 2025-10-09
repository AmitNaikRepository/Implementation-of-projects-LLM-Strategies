# 🖥️ Finetuning with Custom Compute Strategy

> **Strategy #W of 10**: Slash Training Costs by 70% Using Spot Instances, Consumer GPUs, and Smart Resource Management

## 📋 Executive Summary

Most companies overpay for LLM finetuning by using expensive cloud services or managed platforms. We show you how to train models using **consumer GPUs, spot instances, and custom compute** at a fraction of the cost.

### The Savings

| Approach | Cost for 7B Model | Time | Our Rating |
|----------|------------------|------|------------|
| **Managed Platform** (Replicate, Modal) | $150-300 | 8 hours | ❌ Expensive |
| **Cloud GPU** (AWS/GCP/Azure on-demand) | $80-120 | 8 hours | ⚠️ Costly |
| **RunPod Spot** | $15-25 | 8 hours | ✅ Good |
| **Consumer GPU** (RTX 4090 local) | $0 (owned) | 10 hours | ✅ Best |
| **Hybrid** (Spot + optimizations) | $8-12 | 6 hours | ⭐ Optimal |

---

## 🎯 The Problem

### Companies Are Overpaying for Training

```
Typical Scenario: Finetuning Llama-2-7B

Using Managed Platform (e.g., Replicate):
├─ Platform fee: $200
├─ Compute markup: 3-5x over bare metal
├─ Limited customization
├─ Vendor lock-in
└─ Total: $200-300 per training run

Your Reality:
├─ Need to iterate 10-20 times (experimenting)
├─ Cost: $2,000-6,000 per project
├─ Budget constraints kill innovation
└─ "AI is too expensive" mindset
```

### Why This Happens

```
❌ Problem 1: Cloud On-Demand Pricing
   AWS/GCP A100: $2-4/hour
   Training time: 40 hours
   Cost per run: $80-160

❌ Problem 2: Don't Know About Spot Instances  
   Same GPU on spot: $0.50-1.00/hour
   Most teams aren't aware or don't use them
   
❌ Problem 3: Over-Provisioned Resources
   Using A100 when RTX 4090 is sufficient
   Training on full precision when FP16 works
   Not using quantization or LoRA

❌ Problem 4: Poor Resource Utilization
   GPU sitting idle during data loading
   No mixed precision training
   Inefficient batch sizes
   Single GPU when multi-GPU is available
```

---

## 💡 What We Do Differently

### Our Custom Compute Strategy

```
┌────────────────────────────────────────────────────────┐
│           Smart Resource Selection                     │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Step 1: Match GPU to Task                            │
│  ├─ 7B model? → RTX 4090 (24GB) ✅                    │
│  ├─ 13B model? → A6000 (48GB) ✅                      │
│  └─ 70B model? → 2x A100 (80GB) ✅                    │
│                                                        │
│  Step 2: Use Spot Instances                           │
│  ├─ 70-80% cheaper than on-demand                     │
│  ├─ Checkpointing handles interruptions               │
│  └─ Auto-resume on new instance                       │
│                                                        │
│  Step 3: Optimize Training                            │
│  ├─ LoRA instead of full finetuning                   │
│  ├─ Mixed precision (FP16/BF16)                       │
│  ├─ Gradient accumulation                             │
│  └─ Efficient data loading                            │
│                                                        │
│  Step 4: Multi-GPU When Needed                        │
│  ├─ Cheaper dual RTX 4090 vs single A100              │
│  ├─ Parallel data loading                             │
│  └─ Faster training time                              │
│                                                        │
└────────────────────────────────────────────────────────┘

Result: Same quality, 70-90% lower cost
```

---

## 🔬 How We Do It

### 1. GPU Selection Matrix

```python
"""
Choose the right GPU for your model
"""

gpu_recommendations = {
    "llama-7b": {
        "minimum": "RTX 3090 (24GB)",
        "recommended": "RTX 4090 (24GB)",
        "cost_per_hour": "$0.50",
        "training_time": "8-10 hours",
        "total_cost": "$4-5",
        "use_lora": True  # Essential for 24GB
    },
    
    "llama-13b": {
        "minimum": "RTX A6000 (48GB)", 
        "recommended": "A6000 (48GB)",
        "cost_per_hour": "$0.80",
        "training_time": "12-16 hours",
        "total_cost": "$10-13",
        "use_lora": True
    },
    
    "llama-70b": {
        "minimum": "2x A100 (80GB)",
        "recommended": "4x A100 (80GB) or 8x RTX 4090",
        "cost_per_hour": "$2.50-4.00",
        "training_time": "24-40 hours",
        "total_cost": "$60-160",
        "use_lora": True,
        "use_deepspeed": True
    }
}

def recommend_gpu(model_name, budget):
    """Get GPU recommendation based on model and budget"""
    config = gpu_recommendations.get(model_name, {})
    
    if budget == "minimal":
        return config["minimum"]
    elif budget == "optimal":
        return config["recommended"]
```

### 2. Spot Instance Strategy

```python
"""
Use spot instances with automatic checkpointing
"""

class SpotInstanceTrainer:
    def __init__(self, model_name, checkpoint_dir="./checkpoints"):
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = 500  # Save every 500 steps
        
    def setup_training(self):
        """
        Configure training for spot instance interruptions
        """
        training_args = TrainingArguments(
            output_dir=self.checkpoint_dir,
            
            # Aggressive checkpointing for spot instances
            save_strategy="steps",
            save_steps=self.checkpoint_interval,
            save_total_limit=3,  # Keep last 3 checkpoints
            
            # Auto-resume from checkpoint
            resume_from_checkpoint=True,
            
            # Push to cloud storage (S3/GCS) regularly
            push_to_hub=True,  # Or use S3
            hub_strategy="checkpoint",
            
            # Enable all optimizations
            fp16=True,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
        )
        
        return training_args
    
    def train_with_spot(self):
        """
        Train with automatic recovery from spot interruptions
        """
        print("🎯 Training on spot instance with auto-recovery...")
        
        try:
            # Normal training
            trainer.train(resume_from_checkpoint=True)
            
        except Exception as e:
            if "spot instance interrupted" in str(e):
                print("⚠️ Spot interrupted! Checkpoint saved.")
                print("💾 Resume training on new instance with:")
                print(f"   trainer.train(resume_from_checkpoint='{self.checkpoint_dir}')")
            else:
                raise e

# Usage
trainer_config = SpotInstanceTrainer("llama-2-7b")
args = trainer_config.setup_training()

# Training will auto-save and can resume after interruption
```

### 3. LoRA for Memory Efficiency

```python
"""
Use LoRA to train on consumer GPUs
"""

from peft import LoraConfig, get_peft_model, TaskType

def setup_lora_training(base_model, target_modules=None):
    """
    Configure LoRA for efficient training
    """
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # LoRA rank (higher = more parameters, but more memory)
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.05,
        target_modules=target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"  # FFN
        ],
        bias="none",
    )
    
    # Apply LoRA to model
    model = get_peft_model(base_model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"💡 Trainable params: {trainable_params:,}")
    print(f"📊 Total params: {total_params:,}")
    print(f"🎯 Percentage: {100 * trainable_params / total_params:.2f}%")
    
    return model

# Example: Llama-2-7B with LoRA
# Full finetuning: 7B parameters (28GB memory)
# LoRA finetuning: 4.2M parameters (8GB memory)
# Reduction: 99.94% fewer parameters!
```

### 4. Optimization Stack

```python
"""
Complete optimization for custom compute
"""

from transformers import TrainingArguments

def get_optimized_training_args(
    model_size="7b",
    gpu_memory="24gb",
    use_spot=True
):
    """
    Generate optimized training configuration
    """
    
    # Base configuration
    config = {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,  # Effective batch size: 16
        "num_train_epochs": 3,
        "learning_rate": 2e-4,
        
        # Memory optimizations
        "fp16": True,  # Use mixed precision
        "gradient_checkpointing": True,  # Trade compute for memory
        "optim": "paged_adamw_8bit",  # 8-bit optimizer
        
        # Speed optimizations
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
        "ddp_find_unused_parameters": False,
        
        # Logging
        "logging_steps": 10,
        "evaluation_strategy": "steps",
        "eval_steps": 100,
    }
    
    # Adjust for GPU memory
    if gpu_memory == "24gb":
        config["per_device_train_batch_size"] = 1
        config["gradient_accumulation_steps"] = 16
    elif gpu_memory == "48gb":
        config["per_device_train_batch_size"] = 2
        config["gradient_accumulation_steps"] = 8
    
    # Spot instance settings
    if use_spot:
        config.update({
            "save_strategy": "steps",
            "save_steps": 500,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
        })
    
    return TrainingArguments(**config)

# Usage
training_args = get_optimized_training_args(
    model_size="7b",
    gpu_memory="24gb", 
    use_spot=True
)
```

### 5. Multi-GPU Cost Optimization

```python
"""
Use multiple cheap GPUs instead of one expensive GPU
"""

# Option 1: Single A100 (80GB)
single_a100 = {
    "gpu": "1x A100 80GB",
    "cost_per_hour": "$2.50",
    "training_time": "10 hours",
    "total_cost": "$25"
}

# Option 2: Dual RTX 4090 (48GB total)
dual_4090 = {
    "gpu": "2x RTX 4090 24GB",
    "cost_per_hour": "$1.00",  # $0.50 each
    "training_time": "7 hours",  # Faster with parallelism
    "total_cost": "$7"
}

# Savings: $25 → $7 (72% cheaper!)

# Implementation with DeepSpeed
deepspeed_config = {
    "train_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,  # Partition optimizer states
        "offload_optimizer": {"device": "cpu"},  # Offload to CPU
    }
}

# Launch training with multiple GPUs
# deepspeed --num_gpus=2 train.py --deepspeed deepspeed_config.json
```

---

## 🏗️ Platform Comparison

### Where to Train

```
┌─────────────────────────────────────────────────────────┐
│ Platform Comparison (7B Model Training)                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 🏆 RunPod (Spot)                                        │
│ ├─ Cost: $8-15                                          │
│ ├─ Setup: 5 minutes                                     │
│ ├─ Flexibility: High                                    │
│ └─ Best for: Most teams                                 │
│                                                         │
│ ✅ Vast.ai (Spot)                                       │
│ ├─ Cost: $5-12                                          │
│ ├─ Setup: 10 minutes                                    │
│ ├─ Flexibility: Very high                               │
│ └─ Best for: Lowest cost priority                       │
│                                                         │
│ ✅ Lambda Labs                                          │
│ ├─ Cost: $15-25                                         │
│ ├─ Setup: Instant                                       │
│ ├─ Flexibility: Medium                                  │
│ └─ Best for: Reliability priority                       │
│                                                         │
│ ⚠️ AWS/GCP/Azure (On-demand)                            │
│ ├─ Cost: $80-120                                        │
│ ├─ Setup: 30 minutes                                    │
│ ├─ Flexibility: Very high                               │
│ └─ Best for: Enterprise with credits                    │
│                                                         │
│ ❌ Managed Platforms (Replicate, Modal)                 │
│ ├─ Cost: $150-300                                       │
│ ├─ Setup: Instant                                       │
│ ├─ Flexibility: Low                                     │
│ └─ Best for: Quick demos only                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Local vs Cloud Decision

```python
"""
When to train locally vs cloud
"""

def should_use_local(gpu_owned, model_size, iterations):
    """
    Decide between local and cloud training
    """
    scenarios = {
        "have_rtx_4090_train_7b_once": {
            "decision": "LOCAL",
            "reason": "Free, one-time job",
            "cost": "$0"
        },
        
        "have_rtx_4090_train_7b_10x": {
            "decision": "LOCAL", 
            "reason": "Free vs $80 cloud",
            "cost": "$0 vs $80"
        },
        
        "no_gpu_train_7b_once": {
            "decision": "CLOUD_SPOT",
            "reason": "Cheap one-time ($10 vs $2000 GPU)",
            "cost": "$10"
        },
        
        "no_gpu_train_70b": {
            "decision": "CLOUD_SPOT",
            "reason": "Can't afford 8x A100 locally",
            "cost": "$60-100"
        },
        
        "have_rtx_3090_train_13b": {
            "decision": "CLOUD_SPOT",
            "reason": "Need 48GB, only have 24GB",
            "cost": "$15-20"
        }
    }
    
    return scenarios

# Key insight: Use local if you have the hardware, 
# use cloud spot for everything else
```

---

## 💰 Real Cost Breakdown

### Case Study: Training Llama-2-7B for Customer Support

```
Project: Finetune Llama-2-7B on 10K customer support conversations
Iterations: 5 (experimenting with hyperparameters)
Total training runs: 5

Option 1: Managed Platform (Replicate)
├─ Cost per run: $250
├─ Total: 5 × $250 = $1,250
└─ Time: 40 hours total

Option 2: AWS On-Demand (A100)
├─ Cost per run: $100 (50 hours × $2/hr)
├─ Total: 5 × $100 = $500
└─ Time: 50 hours total

Option 3: RunPod Spot (RTX 4090) ⭐
├─ Cost per run: $10 (20 hours × $0.50/hr)
├─ Total: 5 × $10 = $50
└─ Time: 50 hours total (slightly slower)

Option 4: Local RTX 4090 (if owned) 🏆
├─ Cost per run: $0 (free)
├─ Total: $0
└─ Time: 60 hours total (no parallelism)

Savings Comparison:
├─ Replicate → RunPod: $1,250 → $50 (96% savings!)
├─ AWS → RunPod: $500 → $50 (90% savings!)
├─ RunPod → Local: $50 → $0 (100% savings!)
```

### Annual Savings

```
Scenario: ML team training 2 models per month

Managed Platform Annual Cost:
├─ $250 per model × 24 models/year
└─ Total: $6,000/year

Custom Compute Annual Cost:
├─ $10 per model × 24 models/year
└─ Total: $240/year

Annual Savings: $5,760 (96% reduction)

Break-even Point for Buying GPU:
├─ RTX 4090: $1,600
├─ Pays for itself in: 3.3 months
└─ After that: Everything is free!
```

---

## 🚀 Quick Start Guide

### Step 1: Choose Your Platform

```bash
# RunPod (Recommended)
# 1. Go to runpod.io
# 2. Select "GPU Pods" → "Spot"
# 3. Choose RTX 4090 or A6000
# 4. Select PyTorch template
# 5. SSH into instance

# Vast.ai (Cheapest)
# 1. Go to vast.ai
# 2. Search for RTX 4090 or A6000
# 3. Sort by price
# 4. Rent and SSH in
```

### Step 2: Setup Environment

```bash
# Quick setup script
pip install torch transformers accelerate peft bitsandbytes

# Verify GPU
nvidia-smi

# Clone training repo
git clone <your-training-repo>
cd training
```

### Step 3: Configure Training

```python
# config.yaml
model_name: "meta-llama/Llama-2-7b-hf"
dataset: "your-dataset"

# LoRA settings (memory efficient)
use_lora: true
lora_r: 16
lora_alpha: 32

# Training settings
batch_size: 1
gradient_accumulation: 16
epochs: 3
learning_rate: 2e-4

# Optimizations
fp16: true
gradient_checkpointing: true
optimizer: "paged_adamw_8bit"

# Spot instance settings
save_steps: 500
checkpoint_dir: "./checkpoints"
```

### Step 4: Start Training

```bash
# Single GPU training
python train.py --config config.yaml

# Multi-GPU training (if available)
accelerate launch --num_processes=2 train.py --config config.yaml

# Monitor progress
watch -n 1 nvidia-smi
```

### Step 5: Handle Spot Interruptions

```bash
# If spot instance interrupted, simply resume:
python train.py --config config.yaml --resume_from_checkpoint ./checkpoints/checkpoint-1000

# Auto-resume script (recommended)
while true; do
    python train.py --config config.yaml --resume_from_checkpoint auto || break
    sleep 5
done
```

---

## 📊 Optimization Checklist

### Before Training

```
✅ Chose right GPU for model size
✅ Using spot instances (not on-demand)
✅ LoRA enabled for 7B-13B models
✅ Mixed precision (FP16) enabled
✅ Gradient checkpointing enabled
✅ Batch size optimized for GPU memory
✅ Gradient accumulation configured
✅ Checkpoint saving every 500 steps
✅ Data preprocessing complete
✅ Test run on small dataset first
```

### During Training

```
✅ Monitor GPU utilization (should be >90%)
✅ Check memory usage (should be near max)
✅ Watch loss curve (should decrease)
✅ Verify checkpoints saving correctly
✅ Track training speed (tokens/sec)
```

### Common Issues & Fixes

```
Issue: Out of Memory (OOM)
Fix:
├─ Reduce batch size to 1
├─ Enable gradient checkpointing
├─ Use LoRA (if not already)
└─ Try quantized optimizer

Issue: Slow Training
Fix:
├─ Increase batch size (if memory allows)
├─ Use multiple GPUs
├─ Check dataloader workers (4-8)
└─ Profile with torch.profiler

Issue: Spot Instance Interrupted
Fix:
├─ Auto-resume from checkpoint
├─ Use cheaper GPU if available
└─ Run during off-peak hours
```

---

## 🎯 Best Practices

### Cost Optimization

```
1. Always use spot instances
   - 70-80% cheaper than on-demand
   - Interruptions are rare
   - Checkpointing handles them

2. Right-size your GPU
   - Don't use A100 for 7B models
   - RTX 4090 is sufficient
   
3. Use LoRA for everything
   - 99% fewer parameters
   - 4x less memory
   - Minimal accuracy loss

4. Buy GPU if training regularly
   - RTX 4090: $1,600
   - Pays for itself in months
   - Then everything is free
```

### Performance Optimization

```
1. Enable all memory optimizations
   ✅ FP16/BF16 mixed precision
   ✅ Gradient checkpointing
   ✅ 8-bit optimizer
   ✅ Gradient accumulation

2. Optimize data loading
   ✅ Preprocess data once
   ✅ Use fast dataloader
   ✅ Pin memory
   ✅ Multiple workers

3. Monitor and adjust
   ✅ Watch GPU utilization
   ✅ Increase batch size if possible
   ✅ Profile slow sections
```

---

## 💼 Service Packages

### Package 1: Training Setup & Optimization
**Duration:** 1 week | **Price:** $3,000 - $5,000

```
✅ GPU selection consultation
✅ Training pipeline setup
✅ Optimization implementation
✅ Spot instance configuration
✅ Cost analysis & recommendations

Deliverable: Optimized training setup + documentation
Savings: 70-90% on training costs
```

### Package 2: Managed Training Service
**Duration:** Ongoing | **Price:** $500/month + compute costs

```
✅ We handle all training
✅ Spot instance management
✅ Automatic checkpointing
✅ Progress monitoring
✅ Model delivery

Deliverable: Trained models on demand
Your cost: Only actual compute (at our discounted rates)
```

---

## 📞 Get Started

**Ready to cut your training costs by 70-90%?**

📧 Email: your-email@company.com  
💼 Schedule: [Calendar Link]

**Free Consultation:**
- Analyze your training needs
- Calculate potential savings  
- Recommend optimal setup

---

<div align="center">

**Strategy #W: Custom Compute Training**  
*70-90% Cost Reduction | Same Quality | Full Control*

💰 Train Smart | ⚡ Save Big | 🚀 Stay Flexible

</div>
