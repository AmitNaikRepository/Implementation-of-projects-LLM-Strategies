# test_groq.py
import sys

try:
    from openai import OpenAI
except ImportError:
    print("❌ OpenAI library not installed")
    print("Run: pip install openai")
    exit(1)

# Check if API key was passed as argument
if len(sys.argv) < 2:
    print("❌ No API key provided!")
    print()
    print("Usage:")
    print("  python test_groq.py YOUR_API_KEY")
    print()
    print("Example:")
    print("  python test_groq.py gsk_abc123xyz...")
    print()
    print("Get your free API key from:")
    print("👉 https://console.groq.com/keys")
    exit(1)

# Get API key from command line argument
GROQ_API_KEY = sys.argv[1]

print("🔍 Testing Groq API connection...")
print(f"🔑 Using key: {GROQ_API_KEY[:8]}...{GROQ_API_KEY[-4:]}")
print()

try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )
    
    print("📡 Sending test request...")
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": "Say 'Hello! Groq is working!' in exactly 5 words."}
        ],
        max_tokens=50,
        temperature=0.5
    )
    
    message = response.choices[0].message.content
    tokens_used = response.usage.total_tokens
    
    print("✅ SUCCESS! Groq API is working!")
    print()
    print("📝 Response:")
    print(f"   {message}")
    print()
    print("📊 Usage:")
    print(f"   Input tokens:  {response.usage.prompt_tokens}")
    print(f"   Output tokens: {response.usage.completion_tokens}")
    print(f"   Total tokens:  {tokens_used}")
    print()
    print("💰 Cost: FREE! (Groq free tier)")
    print()
    print("🎉 Your Groq API key is valid and working!")
    
except Exception as e:
    print("❌ ERROR!")
    print(f"   {str(e)}")
    print()
    
    if "401" in str(e) or "authentication" in str(e).lower():
        print("🔧 Troubleshooting:")
        print("   1. Check your API key is correct")
        print("   2. Make sure you copied the full key from Groq console")
        print("   3. Get a new key at: https://console.groq.com/keys")
    elif "rate" in str(e).lower() or "limit" in str(e).lower():
        print("⏳ Rate limit reached - try again in a few seconds")
    else:
        print("🔧 Troubleshooting:")
        print("   1. Check your internet connection")
        print("   2. Verify Groq service is up")
        print("   3. Try again in a few seconds")