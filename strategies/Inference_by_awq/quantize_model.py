"""
AWQ Model Quantization Script

This script quantizes large language models using AWQ (Activation-aware Weight Quantization)
for efficient inference deployment.

Author: Your Team
Date: 2025
"""

import argparse
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM


class AWQQuantizer:
    """
    Handles the quantization of language models using AWQ technique
    """
    
    def __init__(
        self,
        model_path: str,
        output_path: str,
        w_bit: int = 4,
        q_group_size: int = 128,
        zero_point: bool = True
    ):
        """
        Initialize the quantizer
        
        Args:
            model_path: Path to the original model (HuggingFace model ID or local path)
            output_path: Directory to save quantized model
            w_bit: Quantization bit width (default: 4)
            q_group_size: Quantization group size (default: 128)
            zero_point: Whether to use zero point quantization (default: True)
        """
        self.model_path = model_path
        self.output_path = output_path
        self.quant_config = {
            "zero_point": zero_point,
            "q_group_size": q_group_size,
            "w_bit": w_bit,
            "version": "GEMM"
        }
        
    def quantize(self, calib_data: str = "pileval", n_samples: int = 512):
        """
        Perform AWQ quantization on the model
        
        Args:
            calib_data: Calibration dataset name (default: "pileval")
            n_samples: Number of calibration samples (default: 512)
        """
        print("=" * 80)
        print("🚀 AWQ Model Quantization")
        print("=" * 80)
        print(f"\n📦 Model: {self.model_path}")
        print(f"💾 Output: {self.output_path}")
        print(f"⚙️  Config: {self.quant_config}")
        print(f"📊 Calibration: {calib_data} ({n_samples} samples)")
        print("\n" + "=" * 80)
        
        # Step 1: Load model and tokenizer
        print("\n[1/4] Loading model and tokenizer...")
        start_time = time.time()
        
        try:
            model = AutoAWQForCausalLM.from_pretrained(
                self.model_path,
                safetensors=True,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.2f}s")
            
            # Print model info
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                print(f"   GPU Memory: {memory_allocated:.2f}GB")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Step 2: Quantize the model
        print("\n[2/4] Quantizing model (this may take a while)...")
        quant_start = time.time()
        
        try:
            model.quantize(
                tokenizer,
                quant_config=self.quant_config,
                calib_data=calib_data,
                n_samples=n_samples
            )
            
            quant_time = time.time() - quant_start
            print(f"✅ Quantization completed in {quant_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Error during quantization: {e}")
            raise
        
        # Step 3: Save quantized model
        print("\n[3/4] Saving quantized model...")
        save_start = time.time()
        
        try:
            # Create output directory
            os.makedirs(self.output_path, exist_ok=True)
            
            # Save model and tokenizer
            model.save_quantized(self.output_path)
            tokenizer.save_pretrained(self.output_path)
            
            save_time = time.time() - save_start
            print(f"✅ Model saved in {save_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            raise
        
        # Step 4: Verify and report
        print("\n[4/4] Verification...")
        self._print_summary()
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total time: {total_time:.2f}s")
        print("\n" + "=" * 80)
        print("✅ Quantization complete!")
        print("=" * 80)
    
    def _print_summary(self):
        """Print model size comparison and savings"""
        try:
            # Calculate quantized model size
            total_size = 0
            for root, dirs, files in os.walk(self.output_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
            
            quantized_size_gb = total_size / (1024**3)
            
            # Estimate original size (rough approximation)
            # FP16 models are roughly 2 bytes per parameter
            # AWQ 4-bit models are roughly 0.5 bytes per parameter
            estimated_original_gb = quantized_size_gb * 4
            reduction_percent = ((estimated_original_gb - quantized_size_gb) / 
                                estimated_original_gb * 100)
            
            print("\n📊 Model Size Summary:")
            print(f"   Estimated Original (FP16): ~{estimated_original_gb:.2f}GB")
            print(f"   Quantized (4-bit AWQ): {quantized_size_gb:.2f}GB")
            print(f"   Reduction: ~{reduction_percent:.1f}%")
            print(f"   Space Saved: ~{estimated_original_gb - quantized_size_gb:.2f}GB")
            
        except Exception as e:
            print(f"⚠️  Could not calculate size summary: {e}")


def main():
    """Main entry point for the quantization script"""
    parser = argparse.ArgumentParser(
        description="Quantize LLMs using AWQ for efficient inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize Llama-2-7B
  python quantize_model.py --model meta-llama/Llama-2-7b-hf --output ./models/llama-2-7b-awq
  
  # Quantize with custom settings
  python quantize_model.py --model mistralai/Mistral-7B-v0.1 \\
                          --output ./models/mistral-7b-awq \\
                          --w-bit 4 \\
                          --group-size 128 \\
                          --samples 1024
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for quantized model"
    )
    parser.add_argument(
        "--w-bit",
        type=int,
        default=4,
        choices=[2, 3, 4, 8],
        help="Quantization bit width (default: 4)"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Quantization group size (default: 128)"
    )
    parser.add_argument(
        "--zero-point",
        action="store_true",
        default=True,
        help="Use zero point quantization (default: True)"
    )
    parser.add_argument(
        "--calib-data",
        type=str,
        default="pileval",
        help="Calibration dataset name (default: pileval)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=512,
        help="Number of calibration samples (default: 512)"
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA is not available. Quantization will be slow.")
        print("   Please ensure you're running on a GPU instance.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Initialize quantizer
    quantizer = AWQQuantizer(
        model_path=args.model,
        output_path=args.output,
        w_bit=args.w_bit,
        q_group_size=args.group_size,
        zero_point=args.zero_point
    )
    
    # Run quantization
    try:
        quantizer.quantize(
            calib_data=args.calib_data,
            n_samples=args.samples
        )
    except KeyboardInterrupt:
        print("\n\n❌ Quantization interrupted by user")
        return
    except Exception as e:
        print(f"\n\n❌ Quantization failed: {e}")
        raise


if __name__ == "__main__":
    main()
