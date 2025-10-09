"""
AWQ Inference Server

Production-ready inference server for AWQ quantized models
Provides high-performance text generation with monitoring and metrics

Author: Your Team
Date: 2025
"""

import argparse
import time
from typing import Dict, Optional
import logging

import torch
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AWQInferenceServer:
    """
    Production-ready AWQ inference server with performance monitoring
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        fuse_layers: bool = True,
        max_batch_size: int = 4
    ):
        """
        Initialize the inference server
        
        Args:
            model_path: Path to quantized AWQ model
            device: Device to load model on (default: "cuda:0")
            fuse_layers: Enable layer fusion for speed (default: True)
            max_batch_size: Maximum batch size (default: 4)
        """
        self.model_path = model_path
        self.device = device
        self.max_batch_size = max_batch_size
        self.total_requests = 0
        self.total_tokens_generated = 0
        
        logger.info("=" * 80)
        logger.info("🚀 Initializing AWQ Inference Server")
        logger.info("=" * 80)
        
        # Load model
        logger.info(f"📦 Loading model from: {model_path}")
        start_time = time.time()
        
        try:
            self.model = AutoAWQForCausalLM.from_quantized(
                model_path,
                fuse_layers=fuse_layers,
                device_map=device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
        
        # Print system info
        self._print_system_info()
        
        logger.info("=" * 80)
        logger.info("✅ Server ready to accept requests!")
        logger.info("=" * 80)
    
    def _print_system_info(self):
        """Print system and model information"""
        logger.info("\n📊 System Information:")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Model: {self.model_path}")
        logger.info(f"   Max Batch Size: {self.max_batch_size}")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            gpu_name = torch.cuda.get_device_name(0)
            
            logger.info(f"   GPU: {gpu_name}")
            logger.info(f"   GPU Memory Allocated: {memory_allocated:.2f}GB")
            logger.info(f"   GPU Memory Reserved: {memory_reserved:.2f}GB")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True
    ) -> Dict:
        """
        Generate text from prompt with performance metrics
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            do_sample: Whether to use sampling
            
        Returns:
            Dictionary with generated text and metrics
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            input_length = len(inputs.input_ids[0])
            
            # Measure inference time
            start_time = time.time()
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Calculate metrics
            output_length = len(outputs[0])
            tokens_generated = output_length - input_length
            tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0
            
            # Update server statistics
            self.total_requests += 1
            self.total_tokens_generated += tokens_generated
            
            # Memory info
            memory_used = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
            
            result = {
                "generated_text": generated_text,
                "prompt": prompt,
                "input_tokens": input_length,
                "output_tokens": tokens_generated,
                "total_tokens": output_length,
                "inference_time_seconds": round(inference_time, 3),
                "tokens_per_second": round(tokens_per_second, 1),
                "memory_used_gb": round(memory_used, 2),
                "request_id": self.total_requests
            }
            
            logger.info(
                f"Request #{self.total_requests}: "
                f"{tokens_generated} tokens in {inference_time:.2f}s "
                f"({tokens_per_second:.1f} tok/s)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
    
    def benchmark(
        self,
        num_iterations: int = 10,
        prompt: Optional[str] = None,
        max_new_tokens: int = 100
    ) -> Dict:
        """
        Run performance benchmarks
        
        Args:
            num_iterations: Number of benchmark iterations
            prompt: Test prompt (uses default if None)
            max_new_tokens: Tokens to generate per iteration
            
        Returns:
            Dictionary with benchmark results
        """
        if prompt is None:
            prompt = "Explain the concept of artificial intelligence in simple terms:"
        
        logger.info("=" * 80)
        logger.info(f"🔥 Running Benchmark ({num_iterations} iterations)")
        logger.info("=" * 80)
        logger.info(f"Prompt: {prompt[:50]}...")
        logger.info(f"Max tokens: {max_new_tokens}")
        logger.info("")
        
        times = []
        tokens_per_sec = []
        memory_usage = []
        
        # Warmup
        logger.info("🔄 Warming up...")
        self.generate(prompt, max_new_tokens=50)
        
        # Run benchmark
        logger.info("📊 Running benchmark iterations...")
        for i in range(num_iterations):
            result = self.generate(prompt, max_new_tokens=max_new_tokens)
            
            times.append(result["inference_time_seconds"])
            tokens_per_sec.append(result["tokens_per_second"])
            memory_usage.append(result["memory_used_gb"])
            
            logger.info(
                f"   Iteration {i+1}/{num_iterations}: "
                f"{result['tokens_per_second']} tok/s, "
                f"{result['inference_time_seconds']}s"
            )
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        avg_tokens_per_sec = sum(tokens_per_sec) / len(tokens_per_sec)
        avg_memory = sum(memory_usage) / len(memory_usage)
        
        min_time = min(times)
        max_time = max(times)
        min_tokens_per_sec = min(tokens_per_sec)
        max_tokens_per_sec = max(tokens_per_sec)
        
        results = {
            "iterations": num_iterations,
            "avg_inference_time_seconds": round(avg_time, 3),
            "avg_tokens_per_second": round(avg_tokens_per_sec, 1),
            "min_tokens_per_second": round(min_tokens_per_sec, 1),
            "max_tokens_per_second": round(max_tokens_per_sec, 1),
            "avg_memory_gb": round(avg_memory, 2),
            "fastest_time_seconds": round(min_time, 3),
            "slowest_time_seconds": round(max_time, 3)
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 Benchmark Results:")
        logger.info("=" * 80)
        logger.info(f"   Average Speed: {results['avg_tokens_per_second']} tokens/second")
        logger.info(f"   Min Speed: {results['min_tokens_per_second']} tokens/second")
        logger.info(f"   Max Speed: {results['max_tokens_per_second']} tokens/second")
        logger.info(f"   Average Time: {results['avg_inference_time_seconds']}s")
        logger.info(f"   Average Memory: {results['avg_memory_gb']}GB")
        logger.info("=" * 80)
        
        return results
    
    def get_stats(self) -> Dict:
        """
        Get server statistics
        
        Returns:
            Dictionary with server statistics
        """
        stats = {
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "avg_tokens_per_request": (
                self.total_tokens_generated / self.total_requests 
                if self.total_requests > 0 else 0
            ),
            "model_path": self.model_path,
            "device": self.device
        }
        
        if torch.cuda.is_available():
            stats["gpu_memory_allocated_gb"] = round(
                torch.cuda.memory_allocated(0) / 1024**3, 2
            )
            stats["gpu_memory_reserved_gb"] = round(
                torch.cuda.memory_reserved(0) / 1024**3, 2
            )
            stats["gpu_name"] = torch.cuda.get_device_name(0)
        
        return stats


def main():
    """Main entry point for the inference server"""
    parser = argparse.ArgumentParser(
        description="AWQ Inference Server - Production-ready LLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run interactive inference
  python inference_server.py --model ./models/llama-2-7b-awq
  
  # Run benchmark
  python inference_server.py --model ./models/llama-2-7b-awq --benchmark --iterations 20
  
  # Custom device
  python inference_server.py --model ./models/llama-2-7b-awq --device cuda:1
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
        "--benchmark",
        action="store_true",
        help="Run benchmark mode"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if "cuda" in args.device and not torch.cuda.is_available():
        logger.error("❌ CUDA device specified but CUDA is not available")
        return
    
    # Initialize server
    try:
        server = AWQInferenceServer(
            model_path=args.model,
            device=args.device
        )
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        return
    
    # Run benchmark if requested
    if args.benchmark:
        server.benchmark(num_iterations=args.iterations)
        return
    
    # Interactive mode
    logger.info("\n💬 Interactive Mode - Enter prompts (or 'quit' to exit)")
    logger.info("=" * 80)
    
    while True:
        try:
            prompt = input("\n📝 Prompt: ")
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                logger.info("Exiting...")
                break
            
            if not prompt.strip():
                continue
            
            logger.info("\n🤖 Generating...")
            result = server.generate(prompt, max_new_tokens=args.max_tokens)
            
            print("\n" + "=" * 80)
            print("🤖 Generated Text:")
            print("=" * 80)
            print(result["generated_text"])
            print("\n" + "=" * 80)
            print("⚡ Metrics:")
            print(f"   Tokens: {result['output_tokens']}")
            print(f"   Time: {result['inference_time_seconds']}s")
            print(f"   Speed: {result['tokens_per_second']} tok/s")
            print(f"   Memory: {result['memory_used_gb']}GB")
            print("=" * 80)
            
        except KeyboardInterrupt:
            logger.info("\n\nInterrupted by user")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
    
    # Print final stats
    stats = server.get_stats()
    logger.info("\n" + "=" * 80)
    logger.info("📊 Final Statistics:")
    logger.info("=" * 80)
    logger.info(f"   Total Requests: {stats['total_requests']}")
    logger.info(f"   Total Tokens Generated: {stats['total_tokens_generated']}")
    logger.info(f"   Avg Tokens/Request: {stats['avg_tokens_per_request']:.1f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
