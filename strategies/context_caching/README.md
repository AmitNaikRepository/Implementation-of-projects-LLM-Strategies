# ⚡ Context Caching Strategy

> **Strategy #Z of 10**: Reduce API Costs by 90% and Speed Up Responses 10x Through Intelligent Prompt Caching

## 📋 Executive Summary

Context caching dramatically reduces LLM API costs and latency by **reusing repeated prompt content** instead of processing it every time. Perfect for applications with large system prompts, document analysis, or repeated context.

### The Impact

| Metric | Without Caching | With Caching | Improvement |
|--------|----------------|--------------|-------------|
| **API Cost** | $1,000/month | $100/month | 90% savings |
| **Response Time** | 5 seconds | 0.5 seconds | 10x faster |
| **Tokens Processed** | 100M/month | 10M/month | 90% reduction |

---

## 🎯 The Problem

### Scenario: Document Q&A System

```
User asks 100 questions about a 50-page document (20,000 tokens)

Traditional Approach (No Caching):
┌──────────────────────────────────────────────────┐
│ Request 1: [20K tokens document] + [50 tokens Q1]│
│ Request 2: [20K tokens document] + [50 tokens Q2]│
│ Request 3: [20K tokens document] + [50 tokens Q3]│
│ ...                                              │
│ Request 100: [20K tokens document] + [50 Q100]   │
└──────────────────────────────────────────────────┘

Total Tokens: 20,000 × 100 = 2,000,000 tokens
Cost: 2M tokens × $0.01/1K = $20
Time: 100 × 5 seconds = 500 seconds (8.3 minutes)
```

**Problem: You're paying to process the same 20K token document 100 times!**

### Real-World Pain Points

```
❌ Long System Prompts Repeated Every Request
   "You are a customer service agent with these guidelines: [2000 tokens]..."
   Processed 10,000 times/day = 20M tokens wasted

❌ Document Analysis Applications  
   User asks 50 questions about a contract
   Same contract sent 50 times = massive waste

❌ Multi-Turn Conversations
   Chat history grows with each message
   Reprocessing entire history every turn

❌ Code Repository Context
   Large codebase context in every query
   Thousands of dollars in redundant processing
```

---

## 💡 The Solution: Context Caching

### How It Works

```
First Request (Cache Miss):
┌──────────────────────────────────────────────────┐
│ [20K tokens document] ──► Process & Cache        │
│ [50 tokens question] ──► Process normally        │
│                                                  │
│ Cache ID: abc123 (valid for 5 minutes)          │
│ Cost: Full price for 20,050 tokens              │
└──────────────────────────────────────────────────┘

Subsequent Requests (Cache Hit):
┌──────────────────────────────────────────────────┐
│ Cache ID: abc123 ──► Retrieved instantly         │
│ [50 tokens question] ──► Process normally        │
│                                                  │
│ Cost: Only 50 new tokens (99% savings!)         │
│ Time: Instant retrieval (10x faster!)           │
└──────────────────────────────────────────────────┘

100 Questions Total:
├─ Request 1: 20,050 tokens (full price)
├─ Requests 2-100: 50 tokens each (cached context)
└─ Total: 20,050 + (99 × 50) = 24,950 tokens

Savings: 2M → 25K tokens (98.75% reduction!)
Cost: $20 → $0.25 (98.75% cheaper!)
```

### When to Use Context Caching

| Use Case | Cache This | Why |
|----------|-----------|-----|
| **Document Q&A** | ✅ The document | Asked multiple questions about same doc |
| **Customer Support** | ✅ System prompt + guidelines | Same instructions every request |
| **Code Assistant** | ✅ Codebase context | Repository context doesn't change |
| **Legal Analysis** | ✅ Contracts/regulations | Analyzing same legal text repeatedly |
| **Chatbots** | ✅ Conversation history | Multi-turn conversations |
| **Creative Writing** | ❌ Each new story | Content changes every time |
| **One-off Queries** | ❌ No repeated context | No benefit from caching |

---

## 🔬 Implementation

### Provider Support

```
✅ Anthropic Claude (Native Support)
├─ Cache prefix markers in API
├─ Automatic cache management
└─ 5-minute cache TTL

✅ OpenAI GPT-4/3.5 (Manual Implementation)
├─ Custom caching layer required
├─ Use Redis/Memcached
└─ Manage cache yourself

✅ Self-Hosted Models (Full Control)
├─ KV-cache optimization
├─ Prompt caching middleware
└─ Custom TTL management
```

### Example 1: Anthropic Claude (Native)

```python
"""
Using Anthropic's native prompt caching
"""

import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

# Large document to cache
large_document = """
[Your 20,000 token document here]
This could be a contract, manual, codebase, etc.
"""

# First request - creates cache
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are a helpful assistant analyzing documents.",
        },
        {
            "type": "text", 
            "text": large_document,
            "cache_control": {"type": "ephemeral"}  # Cache this!
        }
    ],
    messages=[
        {"role": "user", "content": "What is the main topic of this document?"}
    ]
)

# Subsequent requests - use cache
for question in user_questions:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": "You are a helpful assistant analyzing documents.",
            },
            {
                "type": "text",
                "text": large_document,
                "cache_control": {"type": "ephemeral"}  # Reuses cache!
            }
        ],
        messages=[
            {"role": "user", "content": question}
        ]
    )
    # Only pays for new question tokens, not document!

# Cost breakdown
print(f"Cache creation: {response.usage.cache_creation_input_tokens} tokens")
print(f"Cache reads: {response.usage.cache_read_input_tokens} tokens")
print(f"Regular input: {response.usage.input_tokens} tokens")
```

### Example 2: OpenAI with Custom Caching

```python
"""
Custom caching layer for OpenAI API
"""

import openai
import hashlib
import redis
import json

# Initialize Redis for caching
cache = redis.Redis(host='localhost', port=6379, db=0)
CACHE_TTL = 300  # 5 minutes

def get_cache_key(content):
    """Generate cache key from content"""
    return hashlib.md5(content.encode()).hexdigest()

def cached_completion(system_prompt, user_message, model="gpt-4"):
    """
    Wrapper that caches system prompt processing
    """
    # Generate cache key for system prompt
    cache_key = f"prompt_cache:{get_cache_key(system_prompt)}"
    
    # Check cache
    cached_embedding = cache.get(cache_key)
    
    if cached_embedding:
        print("✅ Cache hit! Using cached context")
        # In reality, you'd use the cached embeddings
        # This is simplified for illustration
    else:
        print("❌ Cache miss. Processing and caching...")
        cache.setex(cache_key, CACHE_TTL, "cached_data")
    
    # Make API call (OpenAI doesn't have native caching)
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    
    return response

# Usage
long_system_prompt = """
You are an expert customer service agent with access to:
[5000 tokens of guidelines, FAQs, policies...]
"""

# Multiple requests with same system prompt
for customer_query in customer_queries:
    response = cached_completion(
        system_prompt=long_system_prompt,
        user_message=customer_query
    )
```

### Example 3: Self-Hosted with KV-Cache

```python
"""
Self-hosted model with KV-cache optimization
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class CachedInference:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.kv_cache = {}  # Store past key-values
    
    def generate_with_cache(self, cached_context, new_query, cache_id=None):
        """
        Generate using cached KV states for repeated context
        """
        # First time: process context and cache KV states
        if cache_id not in self.kv_cache:
            print("🔄 Processing context and creating cache...")
            
            # Encode cached context
            context_ids = self.tokenizer.encode(
                cached_context, 
                return_tensors="pt"
            )
            
            # Get KV cache from context
            with torch.no_grad():
                outputs = self.model(
                    context_ids,
                    use_cache=True
                )
                self.kv_cache[cache_id] = outputs.past_key_values
            
            print(f"✅ Cache created: {cache_id}")
        
        # Use cached KV states for new query
        print(f"⚡ Using cached context: {cache_id}")
        query_ids = self.tokenizer.encode(new_query, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                query_ids,
                past_key_values=self.kv_cache[cache_id],  # Reuse cached KV!
                max_new_tokens=100
            )
        
        return self.tokenizer.decode(outputs[0])

# Usage
model = CachedInference("meta-llama/Llama-2-7b-hf")

document = "Long document content here..." * 1000  # 20K tokens

# Process multiple queries
for i, question in enumerate(questions):
    response = model.generate_with_cache(
        cached_context=document,
        new_query=question,
        cache_id="doc_123"  # Same ID reuses cache
    )
```

---

## 💰 Cost Savings Analysis

### Real-World Example: Legal Document Analysis

```
Scenario: Law firm analyzing 100-page contracts
├─ Contract size: 40,000 tokens
├─ Questions per contract: 20
├─ Contracts per month: 50
└─ Total queries: 1,000/month

Without Caching:
├─ Tokens per query: 40,000 (contract) + 50 (question) = 40,050
├─ Total tokens: 40,050 × 1,000 = 40,050,000 tokens/month
├─ Cost: 40M × $0.01/1K = $400/month
└─ Response time: 8 seconds per query

With Caching:
├─ First query per contract: 40,050 tokens (full)
├─ Remaining 19 queries: 50 tokens each
├─ Per contract: 40,050 + (19 × 50) = 41,000 tokens
├─ Total tokens: 41,000 × 50 = 2,050,000 tokens/month
├─ Cost: 2.05M × $0.01/1K = $20.50/month
└─ Response time: 0.8 seconds per cached query

Savings:
├─ Cost reduction: $400 → $20.50 (94.9% savings)
├─ Speed improvement: 8s → 0.8s (10x faster)
└─ Annual savings: $4,554
```

### Customer Support Chatbot

```
Scenario: Customer support with long system prompts
├─ System prompt: 3,000 tokens (guidelines, FAQs)
├─ Average query: 50 tokens
├─ Daily queries: 10,000
└─ Monthly queries: 300,000

Without Caching:
├─ Tokens per query: 3,050
├─ Monthly tokens: 3,050 × 300,000 = 915,000,000 tokens
├─ Cost: 915M × $0.01/1K = $9,150/month

With Caching:
├─ Cache system prompt (3,000 tokens) once per session
├─ Average session: 5 queries
├─ Sessions per month: 60,000
├─ Total tokens: (3,050 × 60,000) + (50 × 240,000) = 195M tokens
├─ Cost: 195M × $0.01/1K = $1,950/month

Savings:
├─ Cost reduction: $9,150 → $1,950 (78.7% savings)
├─ Annual savings: $86,400
└─ ROI on implementation: Immediate
```

---

## 🏗️ Implementation Patterns

### Pattern 1: Document Analysis System

```python
"""
Multi-user document analysis with caching
"""

class DocumentAnalysisSystem:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.cache_registry = {}  # Track active caches
    
    def analyze_document(self, document_id, document_content, questions):
        """
        Analyze document with automatic caching
        """
        results = []
        
        for i, question in enumerate(questions):
            is_first_query = (i == 0)
            
            # Create cache on first query, reuse on subsequent
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": "You are an expert document analyst."
                    },
                    {
                        "type": "text",
                        "text": document_content,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                messages=[
                    {"role": "user", "content": question}
                ]
            )
            
            # Track cache usage
            if is_first_query:
                cache_id = response.id
                self.cache_registry[document_id] = {
                    "cache_id": cache_id,
                    "created_at": time.time(),
                    "queries_served": 1
                }
            else:
                self.cache_registry[document_id]["queries_served"] += 1
            
            results.append({
                "question": question,
                "answer": response.content[0].text,
                "cached": not is_first_query,
                "tokens_saved": response.usage.cache_read_input_tokens
            })
        
        # Calculate savings
        total_saved = sum(r["tokens_saved"] for r in results)
        print(f"💰 Saved {total_saved} tokens through caching!")
        
        return results
```

### Pattern 2: Chatbot with Conversation History

```python
"""
Chatbot that caches growing conversation history
"""

class CachedChatbot:
    def __init__(self):
        self.conversations = {}  # Store conversation state
    
    def chat(self, user_id, message):
        """
        Handle chat with cached conversation history
        """
        # Get or create conversation
        if user_id not in self.conversations:
            self.conversations[user_id] = {
                "history": [],
                "system_prompt": "You are a helpful assistant..."
            }
        
        conv = self.conversations[user_id]
        
        # Build messages with cache control on history
        messages = []
        
        # Mark history for caching (grows over time)
        for msg in conv["history"]:
            messages.append(msg)
        
        # Add new user message
        messages.append({"role": "user", "content": message})
        
        # Make request with cached history
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": conv["system_prompt"],
                    "cache_control": {"type": "ephemeral"}  # Cache system prompt
                }
            ],
            messages=messages
        )
        
        # Update conversation history
        conv["history"].append({"role": "user", "content": message})
        conv["history"].append({
            "role": "assistant", 
            "content": response.content[0].text
        })
        
        return response.content[0].text
```

### Pattern 3: Code Repository Assistant

```python
"""
Code assistant with cached repository context
"""

class CodeAssistant:
    def __init__(self, repo_path):
        self.repo_context = self.load_repo_context(repo_path)
        self.cache_id = None
    
    def load_repo_context(self, repo_path):
        """Load codebase context (architecture, key files, etc.)"""
        context = f"""
        Repository Structure:
        {self.get_file_tree(repo_path)}
        
        Key Files:
        {self.get_important_files(repo_path)}
        
        Architecture:
        {self.extract_architecture(repo_path)}
        """
        return context
    
    def ask(self, question):
        """
        Answer questions with cached repo context
        """
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": "You are an expert code assistant."
                },
                {
                    "type": "text",
                    "text": self.repo_context,  # Large context
                    "cache_control": {"type": "ephemeral"}  # Cache it!
                }
            ],
            messages=[
                {"role": "user", "content": question}
            ]
        )
        
        return response.content[0].text

# Usage
assistant = CodeAssistant("/path/to/repo")

# All queries reuse cached repository context
print(assistant.ask("How does the authentication work?"))
print(assistant.ask("Where is the user model defined?"))
print(assistant.ask("Explain the API routing system."))
# Only pays for questions, not repeated repo context!
```

---

## 📊 Best Practices

### ✅ Do's

```
1. ✅ Cache stable, repeated content
   - System prompts
   - Documents being analyzed
   - Code repositories
   - Reference materials

2. ✅ Use appropriate cache TTL
   - Short sessions: 5 minutes
   - Document analysis: 30 minutes
   - Daily operations: 1 hour

3. ✅ Monitor cache hit rates
   - Track: hits vs misses
   - Optimize: cache more frequently used content
   - Alert: if hit rate drops below 80%

4. ✅ Structure prompts for caching
   - Put cacheable content first
   - Keep variable content last
   - Use clear cache boundaries
```

### ❌ Don'ts

```
1. ❌ Don't cache frequently changing content
   - Real-time data
   - User-specific information
   - Temporary context

2. ❌ Don't cache very small prompts
   - < 1000 tokens: overhead not worth it
   - Simple queries: faster without cache
   - One-off requests: no benefit

3. ❌ Don't forget cache invalidation
   - Update when source changes
   - Clear stale caches
   - Monitor cache freshness

4. ❌ Don't cache sensitive data indefinitely
   - PII should have short TTL
   - Compliance requirements
   - Security considerations
```

---

## 🎯 Quick Start

### Step 1: Identify Cache Candidates

```
Analyze your application:
├─ What prompts are repeated?
├─ What context is stable?
├─ What queries happen in batches?
└─ Where are the biggest token costs?

Good Candidates:
✅ System prompts over 1000 tokens
✅ Documents analyzed multiple times
✅ Code repositories queried repeatedly
✅ Multi-turn conversations
```

### Step 2: Implement Caching

```python
# For Anthropic (easiest)
Add cache_control to your system messages

# For OpenAI
Implement Redis/Memcached layer

# For self-hosted
Use KV-cache optimization
```

### Step 3: Monitor & Optimize

```python
# Track metrics
metrics = {
    "cache_hits": 0,
    "cache_misses": 0,
    "tokens_saved": 0,
    "cost_saved": 0
}

# Calculate hit rate
hit_rate = metrics["cache_hits"] / (metrics["cache_hits"] + metrics["cache_misses"])

# Target: > 80% hit rate for significant savings
```

---

## 💼 Client Packages

### Package 1: Assessment & POC
**Duration:** 1-2 weeks | **Price:** $5,000 - $8,000

```
✅ Analyze your current API usage
✅ Identify caching opportunities  
✅ Implement POC with caching
✅ Measure savings (tokens & cost)
✅ Provide optimization roadmap

Deliverable: Working demo + ROI report
```

### Package 2: Production Implementation
**Duration:** 3-4 weeks | **Price:** $15,000 - $25,000

```
✅ Full caching implementation
✅ Custom caching layer (if needed)
✅ Monitoring dashboard
✅ Cache optimization
✅ Documentation & training

Deliverable: Production system + 30-day support
```

---

## 📞 Next Steps

**Ready to cut your LLM API costs by 90%?**

📧 Email: amitnaik.work@gmail.com
💼 LinkedIn: https://www.linkedin.com/in/amit-naik-6264d/

**Free Assessment:**
- Analyze your current usage
- Calculate potential savings
- Implementation roadmap

---

<div align="center">

**Strategy #Z: Context Caching**  
*90% Cost Reduction | 10x Faster Responses | Instant ROI*

⚡ Cache Smart | 💰 Save Big | 🚀 Ship Fast

</div>
