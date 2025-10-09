#!/bin/bash

# AWQ Inference Strategy - Setup Script
# Automated setup for RunPod or local GPU environments

set -e  # Exit on error

echo "=================================="
echo "🚀 AWQ Inference Strategy Setup"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on GPU instance
check_gpu() {
    echo "🔍 Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        echo -e "${GREEN}✅ GPU detected${NC}"
    else
        echo -e "${YELLOW}⚠️  No GPU detected. This setup works best with CUDA-capable GPUs.${NC}"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    echo ""
}

# Update system packages
update_system() {
    echo "📦 Updating system packages..."
    sudo apt update -qq
    sudo apt upgrade -y -qq
    echo -e "${GREEN}✅ System updated${NC}"
    echo ""
}

# Install system dependencies
install_system_deps() {
    echo "🔧 Installing system dependencies..."
    sudo apt install -y -qq \
        git \
        wget \
        curl \
        htop \
        vim \
        tmux \
        build-essential
    echo -e "${GREEN}✅ System dependencies installed${NC}"
    echo ""
}

# Setup Python environment
setup_python() {
    echo "🐍 Setting up Python environment..."
    
    # Upgrade pip
    pip install --upgrade pip -q
    
    # Install PyTorch with CUDA support
    echo "   Installing PyTorch with CUDA..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
    
    echo -e "${GREEN}✅ PyTorch installed${NC}"
    echo ""
}

# Install AWQ dependencies
install_awq_deps() {
    echo "⚙️  Installing AWQ dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt -q
        echo -e "${GREEN}✅ All dependencies installed${NC}"
    else
        echo -e "${RED}❌ requirements.txt not found${NC}"
        exit 1
    fi
    echo ""
}

# Verify installation
verify_installation() {
    echo "🔍 Verifying installation..."
    
    python3 << EOF
import torch
import transformers
from awq import AutoAWQForCausalLM

print("Python imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
EOF
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Installation verified${NC}"
    else
        echo -e "${RED}❌ Verification failed${NC}"
        exit 1
    fi
    echo ""
}

# Create directory structure
setup_directories() {
    echo "📁 Creating directory structure..."
    
    mkdir -p models
    mkdir -p outputs
    mkdir -p logs
    mkdir -p benchmarks
    
    echo -e "${GREEN}✅ Directories created${NC}"
    echo ""
}

# Download example model (optional)
download_example_model() {
    echo ""
    read -p "📥 Download example AWQ model (llama-2-7b-awq)? This is ~4GB. (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Downloading model from HuggingFace..."
        python3 << EOF
from huggingface_hub import snapshot_download
import os

model_id = "TheBloke/Llama-2-7B-AWQ"
local_dir = "./models/llama-2-7b-awq"

print(f"Downloading {model_id} to {local_dir}...")
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print("Download complete!")
EOF
        echo -e "${GREEN}✅ Model downloaded${NC}"
    else
        echo "Skipping model download"
    fi
    echo ""
}

# Create example scripts
create_example_scripts() {
    echo "📝 Creating example scripts..."
    
    # Create quick test script
    cat > quick_test.sh << 'EOFTEST'
#!/bin/bash
# Quick test script for AWQ inference

echo "🧪 Running quick test..."
python3 inference_server.py \
    --model ./models/llama-2-7b-awq \
    --benchmark \
    --iterations 5

echo ""
echo "✅ Test complete!"
EOFTEST
    
    chmod +x quick_test.sh
    
    # Create benchmark script
    cat > run_benchmark.sh << 'EOFBENCH'
#!/bin/bash
# Comprehensive benchmark script

echo "🔥 Running comprehensive benchmark..."
python3 inference_server.py \
    --model ./models/llama-2-7b-awq \
    --benchmark \
    --iterations 20 \
    | tee benchmarks/benchmark_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Benchmark complete! Results saved to benchmarks/"
EOFBENCH
    
    chmod +x run_benchmark.sh
    
    echo -e "${GREEN}✅ Example scripts created${NC}"
    echo ""
}

# Print usage instructions
print_usage() {
    echo "=================================="
    echo "✅ Setup Complete!"
    echo "=================================="
    echo ""
    echo "📚 Quick Start Guide:"
    echo ""
    echo "1. Quantize a model:"
    echo "   python3 quantize_model.py --model meta-llama/Llama-2-7b-hf --output ./models/llama-2-7b-awq"
    echo ""
    echo "2. Run inference server:"
    echo "   python3 inference_server.py --model ./models/llama-2-7b-awq"
    echo ""
    echo "3. Run benchmark:"
    echo "   ./run_benchmark.sh"
    echo ""
    echo "4. Start API server:"
    echo "   python3 api_server.py --model ./models/llama-2-7b-awq --port 8000"
    echo ""
    echo "📖 Documentation: README.md"
    echo "🐛 Logs: logs/"
    echo "📊 Benchmarks: benchmarks/"
    echo ""
    echo "=================================="
}

# Main execution
main() {
    echo "Starting setup process..."
    echo ""
    
    check_gpu
    
    # Ask for full or minimal setup
    echo "Setup options:"
    echo "1. Full setup (recommended for RunPod)"
    echo "2. Minimal setup (dependencies only)"
    echo ""
    read -p "Choose option (1 or 2): " -n 1 -r
    echo ""
    
    if [[ $REPLY == "1" ]]; then
        update_system
        install_system_deps
    fi
    
    setup_python
    install_awq_deps
    verify_installation
    setup_directories
    create_example_scripts
    
    if [[ $REPLY == "1" ]]; then
        download_example_model
    fi
    
    print_usage
    
    echo -e "${GREEN}🎉 All done! Ready to run AWQ inference.${NC}"
}

# Run main function
main
