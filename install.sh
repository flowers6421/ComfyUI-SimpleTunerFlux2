#!/bin/bash
#
# Installation script for ComfyUI-SimpleTunerFlux2
#
# This script:
# 1. Initializes and updates the SimpleTuner git submodule
# 2. Installs SimpleTuner's dependencies
# 3. Installs this node's dependencies
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "ComfyUI-SimpleTunerFlux2 Installation"
echo "=========================================="

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed. Please install git first."
    exit 1
fi

# Initialize and update submodule
echo ""
echo "[1/3] Initializing SimpleTuner submodule..."
if [ -f ".gitmodules" ]; then
    git submodule update --init --recursive
    echo "SimpleTuner submodule initialized successfully."
else
    echo "Warning: .gitmodules not found. Checking if SimpleTuner exists..."
    if [ ! -d "SimpleTuner" ]; then
        echo "Cloning SimpleTuner..."
        git clone https://github.com/bghira/SimpleTuner.git SimpleTuner
    else
        echo "SimpleTuner directory already exists."
    fi
fi

# Verify SimpleTuner is present
if [ ! -d "SimpleTuner/simpletuner" ]; then
    echo "Error: SimpleTuner submodule not properly initialized."
    echo "Please run: git submodule update --init --recursive"
    exit 1
fi

echo ""
echo "[2/3] Installing SimpleTuner dependencies..."
if [ -f "SimpleTuner/requirements/pytorch-cuda.txt" ]; then
    pip install -r SimpleTuner/requirements/pytorch-cuda.txt --quiet 2>/dev/null || true
fi
if [ -f "SimpleTuner/requirements.txt" ]; then
    pip install -r SimpleTuner/requirements.txt --quiet || {
        echo "Warning: Some SimpleTuner dependencies may have failed to install."
        echo "This is often okay if the core dependencies (torch, diffusers, peft) are already installed."
    }
fi

echo ""
echo "[3/3] Installing ComfyUI-SimpleTunerFlux2 dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet || {
        echo "Warning: Some dependencies may have failed to install."
    }
fi

echo ""
echo "[4/4] HuggingFace Token (optional)..."
echo "Some models require authentication with HuggingFace."
read -p "Do you want to configure HuggingFace token? (y/N): " configure_hf
if [[ "$configure_hf" =~ ^[Yy]$ ]]; then
    read -p "Enter your HuggingFace token: " hf_token
    if [ -n "$hf_token" ]; then
        pip install -q huggingface_hub 2>/dev/null || true
        python -c "from huggingface_hub import login; login(token='$hf_token')" 2>/dev/null && \
            echo "HuggingFace token configured successfully." || \
            echo "Warning: Failed to configure HuggingFace token."
    fi
else
    echo "Skipping HuggingFace token configuration."
    echo "You can configure it later with: huggingface-cli login"
fi

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "The SimpleTuner submodule is now available at:"
echo "  $SCRIPT_DIR/SimpleTuner"
echo ""
echo "To use the nodes in ComfyUI, restart ComfyUI."
echo ""
echo "If you have trained LoRAs, place them in one of these locations:"
echo "  1. SimpleTuner/output/<project>/checkpoint-<step>/pytorch_lora_weights.safetensors"
echo "  2. ComfyUI/models/loras/"
echo "  3. Or specify the full path in the LoRA loader node"
echo ""

