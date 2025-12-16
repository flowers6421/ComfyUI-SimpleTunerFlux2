#!/usr/bin/env python3
"""
Installation script for ComfyUI-SimpleTunerFlux2

This script:
1. Initializes and updates the SimpleTuner git submodule
2. Installs SimpleTuner's dependencies  
3. Installs this node's dependencies

Works on Windows, macOS, and Linux.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, cwd=None, ignore_errors=False):
    """Run a command and return success status."""
    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 and not ignore_errors:
            print(f"  Warning: {result.stderr.strip()}")
            return False
        return True
    except Exception as e:
        if not ignore_errors:
            print(f"  Error: {e}")
        return False


def main():
    print("=" * 50)
    print("ComfyUI-SimpleTunerFlux2 Installation")
    print("=" * 50)
    
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    
    # Step 1: Initialize submodule
    print("\n[1/3] Initializing SimpleTuner submodule...")
    
    gitmodules = script_dir / ".gitmodules"
    simpletuner_dir = script_dir / "SimpleTuner"
    
    if gitmodules.exists():
        run_command(["git", "submodule", "update", "--init", "--recursive"])
    elif not simpletuner_dir.exists():
        print("  Cloning SimpleTuner...")
        run_command([
            "git", "clone", 
            "https://github.com/bghira/SimpleTuner.git", 
            "SimpleTuner"
        ])
    else:
        print("  SimpleTuner directory already exists.")
    
    # Verify SimpleTuner
    simpletuner_module = simpletuner_dir / "simpletuner"
    if not simpletuner_module.exists():
        print("Error: SimpleTuner not properly initialized.")
        print("Please run: git submodule update --init --recursive")
        sys.exit(1)
    
    print("  SimpleTuner submodule ready.")
    
    # Step 2: Install SimpleTuner dependencies
    print("\n[2/3] Installing SimpleTuner dependencies...")
    
    pytorch_req = simpletuner_dir / "requirements" / "pytorch-cuda.txt"
    if pytorch_req.exists():
        run_command(
            [sys.executable, "-m", "pip", "install", "-r", str(pytorch_req), "-q"],
            ignore_errors=True
        )
    
    st_requirements = simpletuner_dir / "requirements.txt"
    if st_requirements.exists():
        run_command(
            [sys.executable, "-m", "pip", "install", "-r", str(st_requirements), "-q"],
            ignore_errors=True
        )
    
    # Step 3: Install this node's dependencies
    print("\n[3/3] Installing ComfyUI-SimpleTunerFlux2 dependencies...")
    
    requirements = script_dir / "requirements.txt"
    if requirements.exists():
        run_command(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements), "-q"],
            ignore_errors=True
        )
    
    # Step 4: Optional HuggingFace token
    print("\n[4/4] HuggingFace Token (optional)...")
    print("Some models require authentication with HuggingFace.")
    try:
        configure_hf = input("Do you want to configure HuggingFace token? (y/N): ").strip().lower()
        if configure_hf == 'y':
            hf_token = input("Enter your HuggingFace token: ").strip()
            if hf_token:
                try:
                    from huggingface_hub import login
                    login(token=hf_token)
                    print("HuggingFace token configured successfully.")
                except Exception as e:
                    print(f"Warning: Failed to configure HuggingFace token: {e}")
        else:
            print("Skipping HuggingFace token configuration.")
            print("You can configure it later with: huggingface-cli login")
    except EOFError:
        print("Skipping HuggingFace token (non-interactive mode).")

    print("\n" + "=" * 50)
    print("Installation complete!")
    print("=" * 50)
    print(f"\nSimpleTuner submodule is at:\n  {simpletuner_dir}")
    print("\nTo use the nodes, restart ComfyUI.")
    print("\nLoRA locations:")
    print("  1. SimpleTuner/output/<project>/checkpoint-<step>/")
    print("  2. ComfyUI/models/loras/")
    print("  3. Or specify full path in the LoRA loader node")


if __name__ == "__main__":
    main()

