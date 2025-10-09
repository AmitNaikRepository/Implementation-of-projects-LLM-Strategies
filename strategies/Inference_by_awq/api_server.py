"""
AWQ Inference REST API Server

Production-ready REST API for serving AWQ quantized models
Includes rate limiting, monitoring, and health checks

Author: Your Team
Date: 2025
"""

import argparse
import logging
import time
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from inference_server import AWQInferenceServer


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global model server instance
model_server: Optional[AWQInferenceServer] = None


# Pydantic models for API
class GenerationRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Input text prompt", min_length=1)
    max_new_tokens: int = Field(512, description="Maximum tokens to generate", ge=1, le=2048)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: float = Field(0.9, description="Nucleus sampling probability", ge=0.0, le=1.0)
    top_k: int = Field(50, description="Top-k sampling parameter", ge=0)
    repetition_penalty: float = Field(1.0, description="Repetition penalty", ge=1.0, le=2.0)
    do_sample: bool = Field(True, description="Whether to use sampling")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain quantum computing in simple terms:",
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50
            }
        }


class GenerationResponse(BaseModel):
    """Response model for text generation"""
    generated_text: str
    prompt: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    inference_time_seconds: float
    tokens_per_second: float
    memory_used_gb: float
    request_id: int
    timestamp: float


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str] = None
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Response model for metrics"""
    total_requests: int
    total_tokens_generated: int
    avg_tokens_per_request: float
    gpu_memory_allocated_gb: Optional[float] = None
    gpu_memory_reserved_gb: Optional[float] = None
    model_path: str
    device: str


# Initialize FastAPI app
app = FastAPI(
    title="AWQ Inference API",
    description="Production-ready LLM inference with AWQ quantization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiter to app
app.state.limiter = limiter

# Track server start time
server_start_time = time.time()


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors"""
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": "Too many requests. Please try again later."
        }
    )


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model_server
    
    logger.info("=" * 80)
    logger.info("🚀 Starting AWQ Inference API Server")
    logger.info("=" * 80)
    
    try:
        # Get model path from environment or command line
        import os
        model_path = os.getenv("MODEL_PATH", "./models/llama-2-7b-awq")
        device = os.getenv("DEVICE", "cuda:0")
        
        logger.info(f"📦 Loading model from: {model_path}")
        model_server = AWQInferenceServer(model_path=model_path, device=device)
        logger.info("✅ API ready to serve requests!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down API server...")
    
    if model_server:
        stats = model_server.get_stats()
        logger.info(f"📊 Final stats: {stats['total_requests']} requests served")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AWQ Inference API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint
    
    Returns server health status and GPU information
    """
    uptime = time.time() - server_start_time
    
    health_data = {
        "status": "healthy" if model_server is not None else "unhealthy",
        "model_loaded": model_server is not None,
        "gpu_available": torch.cuda.is_available(),
        "uptime_seconds": uptime
    }
    
    if torch.cuda.is_available():
        health_data["gpu_name"] = torch.cuda.get_device_name(0)
    
    return HealthResponse(**health_data)


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Get server metrics
    
    Returns detailed metrics about requests and resource usage
    """
    if model_server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    stats = model_server.get_stats()
    return MetricsResponse(**stats)


@app.post("/generate", response_model=GenerationResponse, tags=["Inference"])
@limiter.limit("100/minute")  # 100 requests per minute per IP
async def generate_text(request: Request, gen_request: GenerationRequest):
    """
    Generate text from prompt
    
    This endpoint accepts a text prompt and generation parameters,
    then returns generated text with performance metrics.
    
    **Rate Limit**: 100 requests per minute per IP address
    """
    if model_server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"📝 New request: {gen_request.prompt[:50]}...")
        
        result = model_server.generate(
            prompt=gen_request.prompt,
            max_new_tokens=gen_request.max_new_tokens,
            temperature=gen_request.temperature,
            top_p=gen_request.top_p,
            top_k=gen_request.top_k,
            repetition_penalty=gen_request.repetition_penalty,
            do_sample=gen_request.do_sample
        )
        
        # Add timestamp
        result["timestamp"] = time.time()
        
        return GenerationResponse(**result)
        
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU out of memory")
        raise HTTPException(
            status_code=507,
            detail="GPU memory exceeded. Try reducing max_new_tokens or try again later."
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/stream", tags=["Inference"])
async def generate_text_stream(request: Request, gen_request: GenerationRequest):
    """
    Stream generated text (token by token)
    
    This endpoint streams the generated tokens as they are produced.
    Useful for real-time applications like chatbots.
    
    **Note**: Streaming implementation requires additional setup.
    This is a placeholder for future enhancement.
    """
    raise HTTPException(
        status_code=501,
        detail="Streaming not yet implemented. Use /generate endpoint instead."
    )


@app.post("/benchmark", tags=["Testing"])
@limiter.limit("10/hour")  # Limited to prevent abuse
async def run_benchmark(request: Request, iterations: int = 10):
    """
    Run performance benchmark
    
    Runs multiple inference iterations and returns performance statistics.
    
    **Rate Limit**: 10 requests per hour per IP address
    """
    if model_server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if iterations < 1 or iterations > 100:
        raise HTTPException(
            status_code=400,
            detail="Iterations must be between 1 and 100"
        )
    
    try:
        logger.info(f"🔥 Running benchmark with {iterations} iterations")
        results = model_server.benchmark(num_iterations=iterations)
        return results
        
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Main entry point for the API server"""
    parser = argparse.ArgumentParser(
        description="AWQ Inference REST API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with default settings
  python3 api_server.py --model ./models/llama-2-7b-awq
  
  # Start on custom port
  python3 api_server.py --model ./models/llama-2-7b-awq --port 8080
  
  # Start with custom host and workers
  python3 api_server.py --model ./models/llama-2-7b-awq --host 0.0.0.0 --workers 2
  
  # Test the API
  curl -X POST http://localhost:8000/generate \\
       -H "Content-Type: application/json" \\
       -d '{"prompt": "Hello, how are you?", "max_new_tokens": 100}'
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to quantized AWQ model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (default: cuda:0)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    # Set environment variables for startup event
    import os
    os.environ["MODEL_PATH"] = args.model
    os.environ["DEVICE"] = args.device
    
    # Check CUDA availability
    if "cuda" in args.device and not torch.cuda.is_available():
        logger.error("❌ CUDA device specified but CUDA is not available")
        return
    
    # Start server
    logger.info(f"🌐 Starting server on {args.host}:{args.port}")
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
