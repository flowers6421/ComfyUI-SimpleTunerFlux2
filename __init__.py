"""
ComfyUI-SimpleTunerFlux2: Custom nodes for using SimpleTuner-trained Flux 2 LoRA models in ComfyUI.

This package provides nodes to load and use LoRA models trained with SimpleTuner's custom
Flux 2 architecture, which uses fused `to_qkv_mlp_proj` layers that are incompatible with
standard ComfyUI Flux nodes.

SimpleTuner is included as a git submodule. Run `install.sh` to initialize it.
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where this package is installed
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_simpletuner_path() -> str:
    """
    Get the path to SimpleTuner installation.
    Priority:
    1. Environment variable SIMPLETUNER_PATH
    2. Local submodule at ./SimpleTuner
    3. Fallback to external path at ../../Flux2-train/SimpleTuner
    """
    # Check environment variable first
    env_path = os.environ.get("SIMPLETUNER_PATH")
    if env_path and os.path.exists(env_path):
        return os.path.abspath(env_path)

    # Check local submodule (preferred)
    local_submodule = os.path.join(PACKAGE_DIR, "SimpleTuner")
    if os.path.exists(local_submodule) and os.path.isdir(local_submodule):
        # Verify it has the expected structure
        simpletuner_module = os.path.join(local_submodule, "simpletuner")
        if os.path.exists(simpletuner_module):
            return local_submodule

    # Fallback to external path
    external_path = os.path.join(PACKAGE_DIR, "..", "..", "..", "Flux2-train", "SimpleTuner")
    external_path = os.path.abspath(external_path)
    if os.path.exists(external_path):
        return external_path

    return local_submodule  # Return local path even if not found, for error messages

# Add SimpleTuner to the Python path
SIMPLETUNER_PATH = get_simpletuner_path()

if os.path.exists(SIMPLETUNER_PATH) and SIMPLETUNER_PATH not in sys.path:
    sys.path.insert(0, SIMPLETUNER_PATH)
    logger.info(f"Added SimpleTuner to Python path: {SIMPLETUNER_PATH}")
elif not os.path.exists(SIMPLETUNER_PATH):
    logger.warning(
        f"SimpleTuner not found at: {SIMPLETUNER_PATH}. "
        "Please run 'install.sh' to initialize the submodule, or "
        "set SIMPLETUNER_PATH environment variable to the correct location."
    )

# Import nodes
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    logger.info(f"ComfyUI-SimpleTunerFlux2 loaded successfully with {len(NODE_CLASS_MAPPINGS)} nodes")
except ImportError as e:
    logger.error(f"Failed to import nodes: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Metadata
__version__ = "1.0.0"
WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]

