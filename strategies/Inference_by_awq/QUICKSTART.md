# 🚀 Quick Start Guide - AWQ Inference Strategy

This guide will get you up and running with AWQ inference in under 10 minutes.

---

## ⚡ Prerequisites

- RunPod account (or GPU with CUDA support)
- Python 3.8+
- CUDA 11.8+
- 24GB+ GPU RAM (for 7B models)

---

## 📦 Step 1: Environment Setup

### Option A: Automated Setup (Recommended)

```bash
# Clone repository
git clone <your-repo-url>
cd Inference_by_awq

# Run automated setup
chmod +x setup.sh
./setup.sh
```

### Option B: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 🔧 Step 2: Quantize Your Model

### Quantize Llama-2-7B

```bash
python3 quantize_model.py \
    --model meta-llama/Llama-2-7b-hf \
    --output ./models/llama-2-7b-awq \
    --w-bit 4 \
    --group-size 128
```

**Expected Output:**
```
🚀 AWQ Model Quantization
================================================================================
📦 Model: meta-llama/Llama-2-7b-hf
💾 Output: ./models/llama-2-7b-awq

[1/4] Loading model and tokenizer...
✅ Model loaded in 45.23s

[2/4] Quantizing model (this may take a while)...
✅ Quantization completed in 312.45s

[3/4] Saving quantized model...
✅ Model saved in 12.34s

[4/4] Verification...
📊 Model Size Summary:
   Estimated Original (FP16): ~14.00GB
   Quantized (4-bit AWQ): 3.80GB
   Reduction: ~72.9%
   Space Saved: ~10.20GB

✅ Quantization complete!
```

### Quantize Other Models

```bash
# Mistral-7B
python3 quantize_model.py \
    --model mistralai/Mistral-7B-v0.1 \
    --output ./models/mistral-7b-awq

# CodeLlama-13B
python3 quantize_model.py \
    --model codellama/CodeLlama-13b-hf \
    --output ./models/codellama-13b-awq

# Custom model
python3 quantize_model.py \
    --model /path/to/your/model \
    --output ./models/your-model-awq
```

---

## 🎯 Step 3: Run Inference

### Interactive Mode

```bash
python3 inference_server.py --model ./models/llama-2-7b-awq
```

**Example Session:**
```
💬 Interactive Mode - Enter prompts (or 'quit' to exit)
================================================================================

📝 Prompt: Explain machine learning in simple terms

🤖 Generating...

================================================================================
🤖 Generated Text:
================================================================================
Machine learning is like teaching a computer to learn from examples, just like 
how we learn from experience. Instead of programming specific rules, we show 
the computer many examples and it figures out patterns on its own...

================================================================================
⚡ Metrics:
   Tokens: 87
   Time: 0.31s
   Speed: 280.6 tok/s
   Memory: 4.12GB
================================================================================
```

### Single Query Mode

```bash
# Quick generation
python3 -c "
from inference_server import AWQInferenceServer
server = AWQInferenceServer('./models/llama-2-7b-awq')
result = server.generate('Write a Python hello world:')
print(result['generated_text'])
"
```

---

## 📊 Step 4: Run Benchmarks

### Quick Benchmark (5 iterations)

```bash
./quick_test.sh
```

### Comprehensive Benchmark (20 iterations)

```bash
./run_benchmark.sh
```

**Expected Results:**
```
🔥 Running Benchmark (20 iterations)
================================================================================

📊 Benchmark Results:
================================================================================
   Average Speed: 285.3 tokens/second
   Min Speed: 268.1 tokens/second
   Max Speed: 302.7 tokens/second
   Average Time: 0.35s
   Average Memory: 4.15GB
================================================================================
```

---

## 🌐 Step 5: Deploy REST API

### Start API Server

```bash
python3 api_server.py \
    --model ./models/llama-2-7b-awq \
    --host 0.0.0.0 \
    --port 8000
```

### Test API with cURL

```bash
# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing:",
    "max_new_tokens": 256,
    "temperature": 0.7
  }'

# Health check
curl http://localhost:8000/health

# Get metrics
curl http://localhost:8000/metrics
```

### Test API with Python

```python
import requests

url = "http://localhost:8000/generate"
payload = {
    "prompt": "Write a Python function to sort a list:",
    "max_new_tokens": 200,
    "temperature": 0.7
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Generated: {result['generated_text']}")
print(f"Speed: {result['tokens_per_second']} tok/s")
```

### API Documentation

Access interactive API docs at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🔥 Common Use Cases

### 1. Chatbot

```python
from inference_server import AWQInferenceServer

server = AWQInferenceServer('./models/llama-2-7b-awq')

conversation_history = ""

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    prompt = f"{conversation_history}\nUser: {user_input}\nAssistant:"
    result = server.generate(prompt, max_new_tokens=200)
    
    response = result['generated_text'].split("Assistant:")[-1].strip()
    print(f"Bot: {response}\n")
    
    conversation_history += f"\nUser: {user_input}\nAssistant: {response}"
```

### 2. Code Generation

```python
from inference_server import AWQInferenceServer

server = AWQInferenceServer('./models/codellama-13b-awq')

prompt = """
Write a Python function that:
1. Takes a list of numbers
2. Filters out negative numbers
3. Returns the sum of squares of remaining numbers

Include docstring and type hints.
"""

result = server.generate(prompt, max_new_tokens=300, temperature=0.2)
print(result['generated_text'])
```

### 3. Batch Processing

```python
from inference_server import AWQInferenceServer
import concurrent.futures

server = AWQInferenceServer('./models/llama-2-7b-awq')

prompts = [
    "Summarize: Article text here...",
    "Translate to French: Hello world",
    "Explain: What is recursion?"
]

def process_prompt(prompt):
    return server.generate(prompt, max_new_tokens=150)

# Process in parallel (if using multiple GPUs)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_prompt, prompts))

for i, result in enumerate(results):
    print(f"\nPrompt {i+1}: {result['generated_text'][:100]}...")
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```bash
# Reduce max_new_tokens
python3 inference_server.py --model ./models/llama-2-7b-awq --max-tokens 256

# Or clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache()"
```

### Issue: Slow Generation

**Check:**
1. Verify GPU is being used: `nvidia-smi`
2. Check if layer fusion is enabled (default: True)
3. Reduce batch size if using batching

**Optimize:**
```python
# Enable all optimizations
server = AWQInferenceServer(
    model_path='./models/llama-2-7b-awq',
    fuse_layers=True,  # Faster inference
    device='cuda:0'
)
```

### Issue: Model Not Found

**Check:**
1. Model path is correct
2. Model was quantized successfully
3. Files exist: `ls -la ./models/llama-2-7b-awq/`

### Issue: Import Errors

**Solution:**
```bash
# Reinstall dependencies
pip uninstall autoawq -y
pip install autoawq --no-cache-dir

# Verify
python3 -c "from awq import AutoAWQForCausalLM; print('OK')"
```

---

## 📈 Performance Tuning

### Optimize for Speed

```python
# Use smaller max_new_tokens
result = server.generate(prompt, max_new_tokens=100)

# Reduce sampling complexity
result = server.generate(
    prompt,
    temperature=1.0,  # Faster
    top_k=0,  # Disable top-k
    do_sample=False  # Greedy decoding
)
```

### Optimize for Quality

```python
# Use higher sampling parameters
result = server.generate(
    prompt,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1
)
```

### Batch Inference (Advanced)

```python
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('./models/llama-2-7b-awq')
model = server.model

# Batch multiple prompts
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to('cuda')

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

for i, output in enumerate(outputs):
    print(f"Output {i}: {tokenizer.decode(output, skip_special_tokens=True)}")
```

---

## 💰 Cost Estimation

### RunPod Pricing Examples

**Scenario 1: Development/Testing**
- GPU: RTX 4090 (24GB)
- Usage: 40 hours/month
- Cost: $0.50/hour × 40 = **$20/month**

**Scenario 2: Small Production**
- GPU: RTX 4090 (24GB)
- Usage: 24/7 operation
- Cost: $0.50/hour × 730 = **$365/month**
- Requests: ~500,000/month
- Cost per 1000 requests: **$0.73**

**Scenario 3: Large Production**
- GPU: 2x RTX A6000 (48GB each)
- Usage: 24/7 operation
- Cost: $0.80/hour × 2 × 730 = **$1,168/month**
- Requests: ~2,000,000/month
- Cost per 1000 requests: **$0.58**

**Compare to FP16 (No Quantization):**
- GPU needed: A100 40GB
- Cost: $2.00/hour × 730 = **$1,460/month**
- Same throughput as 1x RTX 4090
- **Savings with AWQ: 75%**

---

## 📚 Next Steps

1. **Read Full Documentation**: Check `README.md` for detailed information
2. **Explore API**: Visit http://localhost:8000/docs for interactive API documentation
3. **Run Benchmarks**: Test different configurations for your use case
4. **Deploy to Production**: Setup load balancing and monitoring
5. **Optimize Further**: Experiment with different quantization settings

---

## 🆘 Need Help?

- **Documentation**: See main `README.md`
- **Issues**: Check GitHub issues or create a new one
- **RunPod Support**: https://docs.runpod.io
- **AWQ Documentation**: https://github.com/casper-hansen/AutoAWQ

---

## ✅ Success Checklist

- [ ] Environment setup complete
- [ ] Model quantized successfully
- [ ] Benchmark results > 250 tokens/sec
- [ ] API server running
- [ ] Health check passes
- [ ] Test generation successful
- [ ] Memory usage < 5GB for 7B model
- [ ] Ready for production deployment

---

**🎉 Congratulations!** You're now running efficient LLM inference with AWQ quantization.

**Estimated Time to Complete**: 10-15 minutes (excluding model download/quantization time)
