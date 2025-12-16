"""
ComfyUI nodes for SimpleTuner Flux 2 pipeline with LoRA support.

These nodes wrap SimpleTuner's custom Flux 2 implementation to enable
using SimpleTuner-trained LoRA models that have the fused to_qkv_mlp_proj
architecture incompatible with standard ComfyUI Flux nodes.
"""

import os
import sys
import logging
import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image

import folder_paths

logger = logging.getLogger(__name__)

# Cached pipeline instance to avoid reloading
_cached_pipeline = None
_cached_pipeline_id = None

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


def ensure_simpletuner_imported():
    """Ensure SimpleTuner modules are importable."""
    simpletuner_path = get_simpletuner_path()
    if simpletuner_path not in sys.path:
        sys.path.insert(0, simpletuner_path)


def get_lora_files() -> List[str]:
    """Get list of available LoRA files from SimpleTuner output directory."""
    lora_paths = []
    
    # Check SimpleTuner output directory
    simpletuner_path = get_simpletuner_path()
    output_dir = os.path.join(simpletuner_path, "output")
    
    if os.path.exists(output_dir):
        for project in os.listdir(output_dir):
            project_path = os.path.join(output_dir, project)
            if os.path.isdir(project_path):
                # Look for checkpoints
                for item in os.listdir(project_path):
                    if item.startswith("checkpoint-"):
                        checkpoint_path = os.path.join(project_path, item)
                        lora_file = os.path.join(checkpoint_path, "pytorch_lora_weights.safetensors")
                        if os.path.exists(lora_file):
                            # Return relative path from output dir
                            rel_path = os.path.join(project, item, "pytorch_lora_weights.safetensors")
                            lora_paths.append(rel_path)
    
    # Also check ComfyUI loras folder
    try:
        comfy_loras = folder_paths.get_filename_list("loras")
        lora_paths.extend(comfy_loras)
    except Exception:
        pass
    
    return sorted(set(lora_paths)) if lora_paths else ["none"]


def get_diffusion_models() -> List[str]:
    """
    Get list of available diffusion models from multiple possible locations.
    Returns model directory names that contain a valid model_index.json.
    """
    models = ["none"]  # Default option for manual entry

    # Get ComfyUI base path
    comfy_base = os.path.dirname(os.path.dirname(PACKAGE_DIR))

    # Directories to scan for models (name, path)
    model_dirs = [
        # ComfyUI model directories
        ("diffusion_models", os.path.join(comfy_base, "models", "diffusion_models")),
        ("checkpoints", os.path.join(comfy_base, "models", "checkpoints")),
        ("unet", os.path.join(comfy_base, "models", "unet")),
        # Root level directories
        ("/checkpoints", "/checkpoints"),
        ("/diffusion_models", "/diffusion_models"),
        # Home directory
        ("~/checkpoints", os.path.expanduser("~/checkpoints")),
        ("~/ComfyUI/models/checkpoints", os.path.expanduser("~/ComfyUI/models/checkpoints")),
        ("~/ComfyUI/models/diffusion_models", os.path.expanduser("~/ComfyUI/models/diffusion_models")),
    ]

    for dir_name, model_dir in model_dirs:
        if not model_dir or not os.path.exists(model_dir):
            continue

        try:
            for item in os.listdir(model_dir):
                item_path = os.path.join(model_dir, item)
                if os.path.isdir(item_path):
                    # Check if it's a valid diffusers model (has model_index.json)
                    if os.path.exists(os.path.join(item_path, "model_index.json")):
                        # Prefix with folder name to distinguish source
                        models.append(f"{dir_name}/{item}")
                    # Also check HuggingFace cache structure (models--org--name)
                    elif item.startswith("models--"):
                        snapshots_dir = os.path.join(item_path, "snapshots")
                        if os.path.exists(snapshots_dir):
                            for snapshot in os.listdir(snapshots_dir):
                                snapshot_path = os.path.join(snapshots_dir, snapshot)
                                if os.path.exists(os.path.join(snapshot_path, "model_index.json")):
                                    # Convert back to repo format: models--org--name -> org/name
                                    repo_id = item.replace("models--", "").replace("--", "/")
                                    models.append(f"{dir_name}/{repo_id}")
                                    break
        except Exception as e:
            logger.warning(f"Error scanning {dir_name} directory: {e}")

    return sorted(set(models))


class SimpleTunerFlux2PipelineLoader:
    """
    Load the Flux 2 base model using SimpleTuner's custom pipeline implementation.
    This handles the fused to_qkv_mlp_proj architecture specific to SimpleTuner.
    """

    CATEGORY = "SimpleTuner/Flux2"
    FUNCTION = "load_pipeline"
    RETURN_TYPES = ("ST_FLUX2_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        available_models = get_diffusion_models()
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to model or HuggingFace ID (e.g., /path/to/model or black-forest-labs/FLUX.2-dev)",
                }),
                "torch_dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "device": (["cuda", "cpu", "auto"], {"default": "auto"}),
            },
            "optional": {
                "model_dropdown": (available_models, {"default": available_models[0]}),
                "use_safetensors": ("BOOLEAN", {"default": True}),
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            }
        }

    def load_pipeline(
        self,
        model_path: str,
        torch_dtype: str,
        device: str,
        model_dropdown: str = "none",
        use_safetensors: bool = True,
        hf_token: str = "",
    ):
        global _cached_pipeline, _cached_pipeline_id

        # Use model_path if provided, otherwise use dropdown selection
        model_id = model_path.strip() if model_path.strip() else model_dropdown

        # Skip if "none" selected and no path provided
        if model_id == "none":
            raise ValueError(
                "No model selected. Either select a local model from the dropdown "
                "or enter a HuggingFace model ID in the 'model_id_override' field."
            )

        # Check cache
        cache_key = f"{model_id}_{torch_dtype}_{device}"
        if _cached_pipeline is not None and _cached_pipeline_id == cache_key:
            logger.info("Using cached Flux2 pipeline")
            return (_cached_pipeline,)

        ensure_simpletuner_imported()

        # Import SimpleTuner modules
        try:
            from simpletuner.helpers.models.flux2.pipeline import Flux2Pipeline
        except ImportError as e:
            raise ImportError(
                f"Failed to import SimpleTuner Flux2Pipeline. "
                f"Ensure SimpleTuner is installed at {get_simpletuner_path()}: {e}"
            )

        # Parse dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set up HuggingFace token if provided
        token = hf_token.strip() if hf_token else None
        if token:
            logger.info("Using provided HuggingFace token for authentication")

        # Set cache directory to ComfyUI/models/diffusion_models/
        # PACKAGE_DIR is already absolute (from os.path.abspath in definition)
        # Go up: custom_nodes/ComfyUI-SimpleTunerFlux2 -> custom_nodes -> ComfyUI
        comfy_base = os.path.dirname(os.path.dirname(PACKAGE_DIR))
        cache_dir = os.path.abspath(os.path.join(comfy_base, "models", "diffusion_models"))
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Model cache directory: {cache_dir}")

        # Check for local model first (local-first loading strategy)
        local_path = self._find_local_model(model_id, cache_dir)

        if local_path:
            logger.info(f"Found local model at: {local_path}")
            logger.info(f"Loading Flux2 pipeline from local path with dtype={torch_dtype}, device={device}")

            # Load from local path without network access
            # Try preferred format first, then fall back to the other
            pipeline = self._load_with_fallback(
                Flux2Pipeline,
                local_path,
                dtype=dtype,
                use_safetensors_preferred=use_safetensors,
                local_files_only=True,
                token=None,
                cache_dir=None,
            )
        else:
            logger.info(f"Loading Flux2 pipeline from {model_id} with dtype={torch_dtype}, device={device}")

            # Load the pipeline from HuggingFace
            # Try preferred format first, then fall back to the other
            pipeline = self._load_with_fallback(
                Flux2Pipeline,
                model_id,
                dtype=dtype,
                use_safetensors_preferred=use_safetensors,
                local_files_only=False,
                token=token,
                cache_dir=cache_dir,
            )

        pipeline = pipeline.to(device)

        # Log transformer architecture info for debugging
        transformer = getattr(pipeline, 'transformer', None)
        if transformer is not None:
            transformer_class = transformer.__class__.__name__
            logger.info(f"Loaded transformer: {transformer_class}")

            # Check for fused architecture (to_qkv_mlp_proj in single blocks)
            has_fused = False
            if hasattr(transformer, 'single_transformer_blocks') and len(transformer.single_transformer_blocks) > 0:
                first_block = transformer.single_transformer_blocks[0]
                if hasattr(first_block, 'attn') and hasattr(first_block.attn, 'to_qkv_mlp_proj'):
                    has_fused = True
                    logger.info("Transformer has fused to_qkv_mlp_proj architecture (SimpleTuner-compatible)")

            if not has_fused:
                logger.warning(
                    "Transformer does NOT have fused to_qkv_mlp_proj architecture. "
                    "SimpleTuner-trained LoRAs may not load correctly."
                )

        # Cache the pipeline
        _cached_pipeline = pipeline
        _cached_pipeline_id = cache_key

        logger.info("Flux2 pipeline loaded successfully")
        return (pipeline,)

    def _load_with_fallback(
        self,
        pipeline_class,
        model_path: str,
        dtype: torch.dtype,
        use_safetensors_preferred: bool,
        local_files_only: bool,
        token: Optional[str],
        cache_dir: Optional[str],
    ):
        """
        Load pipeline with automatic fallback between safetensors and bin formats.

        The use_safetensors parameter is treated as a preference, not a strict requirement.
        If the preferred format fails, it will automatically try the other format.
        """
        # Build common kwargs
        kwargs = {
            "torch_dtype": dtype,
            "local_files_only": local_files_only,
        }
        if token:
            kwargs["token"] = token
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        # Try preferred format first
        formats_to_try = [use_safetensors_preferred, not use_safetensors_preferred]
        last_error = None

        for use_safetensors in formats_to_try:
            format_name = "safetensors" if use_safetensors else "bin"
            try:
                logger.info(f"Attempting to load with use_safetensors={use_safetensors} ({format_name} format)")
                pipeline = pipeline_class.from_pretrained(
                    model_path,
                    use_safetensors=use_safetensors,
                    **kwargs
                )
                logger.info(f"Successfully loaded model with {format_name} format")
                return pipeline
            except Exception as e:
                error_msg = str(e).lower()
                # Check if this is a file format error (missing .bin or .safetensors)
                if "could not find" in error_msg or "does not appear to have" in error_msg or "no file named" in error_msg:
                    logger.warning(f"Format {format_name} not available, will try alternative: {e}")
                    last_error = e
                    continue
                else:
                    # Other error - re-raise immediately
                    logger.error(f"Failed to load pipeline: {e}")
                    raise

        # If we get here, both formats failed
        if last_error:
            logger.error(f"Failed to load model with both safetensors and bin formats. Last error: {last_error}")
            raise last_error
        else:
            raise RuntimeError(f"Failed to load model from {model_path}")

    def _find_local_model(self, model_id: str, cache_dir: str) -> Optional[str]:
        """
        Check if model exists locally in various possible locations.
        Returns the absolute local path if found, None otherwise.

        Searches in order:
        1. Absolute path (if model_id is absolute)
        2. Relative to current directory
        3. ComfyUI/models/diffusion_models/
        4. ComfyUI/models/checkpoints/
        5. /checkpoints/ (root level)
        6. /diffusion_models/ (root level)
        7. HuggingFace cache structures in all above locations
        """
        # Ensure cache_dir is absolute
        cache_dir = os.path.abspath(cache_dir)

        # Get ComfyUI base for checking multiple folders
        comfy_base = os.path.dirname(os.path.dirname(PACKAGE_DIR))

        # Possible local paths to check
        paths_to_check = []

        # 0. Check if model_id is already an absolute path that exists
        if os.path.isabs(model_id):
            if os.path.exists(os.path.join(model_id, "model_index.json")):
                logger.info(f"Found valid local model at absolute path: {model_id}")
                return model_id
            paths_to_check.append(model_id)

        # 1. Check if model_id has a folder prefix (e.g., "diffusion_models/FLUX.2-dev" or "checkpoints/FLUX.2-dev")
        if model_id.startswith("diffusion_models/") or model_id.startswith("checkpoints/"):
            # Direct path from dropdown selection
            direct_path = os.path.join(comfy_base, "models", model_id.replace("/", os.sep))
            if os.path.exists(os.path.join(direct_path, "model_index.json")):
                logger.info(f"Found valid local model at: {direct_path}")
                return os.path.abspath(direct_path)

        # All directories to search for models
        search_dirs = [
            # ComfyUI model directories
            cache_dir,  # ComfyUI/models/diffusion_models/
            os.path.join(comfy_base, "models", "checkpoints"),  # ComfyUI/models/checkpoints/
            os.path.join(comfy_base, "models", "unet"),  # ComfyUI/models/unet/
            # Root level directories
            "/checkpoints",
            "/diffusion_models",
            "/models/checkpoints",
            "/models/diffusion_models",
            # Home directory
            os.path.expanduser("~/checkpoints"),
            os.path.expanduser("~/models/checkpoints"),
            os.path.expanduser("~/ComfyUI/models/checkpoints"),
            os.path.expanduser("~/ComfyUI/models/diffusion_models"),
        ]

        # Extract model name (last part of path)
        model_name = model_id.split("/")[-1] if "/" in model_id else model_id

        for search_dir in search_dirs:
            if not search_dir or not os.path.exists(search_dir):
                continue

            # 1. Direct path (e.g., /checkpoints/FLUX.2-dev/)
            paths_to_check.append(os.path.join(search_dir, model_name))

            # 2. Full model_id path (e.g., /checkpoints/black-forest-labs/FLUX.2-dev/)
            if "/" in model_id and not model_id.startswith("diffusion_models/") and not model_id.startswith("checkpoints/"):
                paths_to_check.append(os.path.join(search_dir, model_id.replace("/", os.sep)))

            # 3. HuggingFace cache structure (models--org--name/snapshots/...)
            hf_cache_name = f"models--{model_id.replace('/', '--')}"
            hf_cache_path = os.path.join(search_dir, hf_cache_name)
            if os.path.exists(hf_cache_path):
                # Find the latest snapshot
                snapshots_dir = os.path.join(hf_cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    try:
                        snapshots = os.listdir(snapshots_dir)
                        if snapshots:
                            # Use the first (or latest) snapshot
                            paths_to_check.append(os.path.join(snapshots_dir, snapshots[0]))
                    except Exception:
                        pass

        # Check each path for model_index.json (indicates a valid diffusers model)
        for path in paths_to_check:
            try:
                # Ensure absolute path
                abs_path = os.path.abspath(path)
                model_index = os.path.join(abs_path, "model_index.json")
                if os.path.exists(model_index):
                    logger.info(f"Found valid local model at: {abs_path}")
                    return abs_path
            except Exception:
                continue

        logger.warning(f"No local model found for: {model_id}")
        return None


class SimpleTunerFlux2LoRALoader:
    """
    Load SimpleTuner-trained LoRA weights into the Flux 2 pipeline.
    Supports the fused to_qkv_mlp_proj architecture used by SimpleTuner.
    """

    CATEGORY = "SimpleTuner/Flux2"
    FUNCTION = "load_lora"
    RETURN_TYPES = ("ST_FLUX2_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        lora_files = get_lora_files()
        return {
            "required": {
                "pipeline": ("ST_FLUX2_PIPELINE",),
                "lora_name": (lora_files, {"default": lora_files[0] if lora_files else "none"}),
                "lora_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                }),
            },
            "optional": {
                "lora_path_override": ("STRING", {"default": ""}),
                "adapter_name": ("STRING", {"default": "default"}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, pipeline, lora_name, lora_scale, lora_path_override="", adapter_name="default"):
        # Always reload if lora changes
        return lora_path_override if lora_path_override else lora_name

    def load_lora(
        self,
        pipeline,
        lora_name: str,
        lora_scale: float = 1.0,
        lora_path_override: str = "",
        adapter_name: str = "default",
    ):
        # Use override path if provided, otherwise use dropdown selection
        lora_path = lora_path_override if lora_path_override else lora_name

        # Handle skip values - "none", "default", empty string should all skip loading
        skip_values = {"none", "default", ""}
        if not lora_path or lora_path.lower() in skip_values:
            logger.info(f"Skipping LoRA loading (lora_path='{lora_path}')")
            return (pipeline,)

        # Resolve the LoRA path
        resolved_path = self._resolve_lora_path(lora_path)

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"LoRA file not found: {resolved_path}")

        logger.info(f"Loading LoRA weights from {resolved_path} with scale={lora_scale}")

        try:
            # Load the LoRA weights using SimpleTuner's method
            pipeline.load_lora_weights(
                resolved_path,
                adapter_name=adapter_name,
            )
        except ValueError as e:
            error_msg = str(e)
            if "Target modules" in error_msg and "not found" in error_msg:
                # Provide more helpful error message for architecture mismatch
                logger.error(
                    f"LoRA architecture mismatch: {error_msg}\n"
                    f"This LoRA may have been trained with a different model architecture.\n"
                    f"SimpleTuner Flux2 uses fused 'to_qkv_mlp_proj' layers in single-stream blocks.\n"
                    f"Make sure the base model is loaded via SimpleTuner's Flux2Pipeline."
                )
            raise

        # Set the LoRA scale
        if hasattr(pipeline, 'set_adapters'):
            pipeline.set_adapters([adapter_name], adapter_weights=[lora_scale])

        logger.info(f"LoRA loaded successfully: {adapter_name}")
        return (pipeline,)

    def _resolve_lora_path(self, lora_path: str) -> str:
        """Resolve LoRA path - could be absolute, relative to SimpleTuner output, or in ComfyUI loras."""
        # If absolute path exists, use it
        if os.path.isabs(lora_path) and os.path.exists(lora_path):
            return lora_path

        # Check SimpleTuner output directory
        simpletuner_path = get_simpletuner_path()
        st_path = os.path.join(simpletuner_path, "output", lora_path)
        if os.path.exists(st_path):
            return st_path

        # Check if it's a checkpoint directory name
        if not lora_path.endswith(".safetensors"):
            st_path_with_weights = os.path.join(st_path, "pytorch_lora_weights.safetensors")
            if os.path.exists(st_path_with_weights):
                return st_path_with_weights

        # Check ComfyUI loras folder
        try:
            comfy_path = folder_paths.get_full_path("loras", lora_path)
            if comfy_path and os.path.exists(comfy_path):
                return comfy_path
        except Exception:
            pass

        # Return as-is and let the caller handle the error
        return lora_path


class SimpleTunerFlux2Sampler:
    """
    Generate images using SimpleTuner's Flux 2 pipeline.
    Outputs images compatible with ComfyUI's image handling.
    """

    CATEGORY = "SimpleTuner/Flux2"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("ST_FLUX2_PIPELINE",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                }),
                "num_inference_steps": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.5,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
            },
            "optional": {
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                }),
                "max_sequence_length": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 1024,
                    "step": 64,
                }),
            }
        }

    def generate(
        self,
        pipeline,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        num_images: int = 1,
        max_sequence_length: int = 512,
    ):
        logger.info(f"Generating image with prompt: {prompt[:50]}...")

        # Set up generator for reproducibility
        device = next(pipeline.transformer.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)

        # Generate images
        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                num_images_per_prompt=num_images,
                max_sequence_length=max_sequence_length,
                return_dict=True,
            )

        images = result.images

        # Convert PIL images to ComfyUI tensor format (B, H, W, C) normalized to [0, 1]
        image_tensors = []
        for img in images:
            if isinstance(img, Image.Image):
                img_array = np.array(img).astype(np.float32) / 255.0
                image_tensors.append(torch.from_numpy(img_array))

        if image_tensors:
            # Stack into batch tensor
            batch_tensor = torch.stack(image_tensors, dim=0)
        else:
            # Return empty tensor if no images
            batch_tensor = torch.zeros((1, height, width, 3))

        logger.info(f"Generated {len(images)} image(s)")
        return (batch_tensor,)


class SimpleTunerFlux2LoRAUnloader:
    """
    Unload/remove LoRA adapters from the pipeline.
    Useful for switching between different LoRAs without reloading the base model.
    """

    CATEGORY = "SimpleTuner/Flux2"
    FUNCTION = "unload_lora"
    RETURN_TYPES = ("ST_FLUX2_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("ST_FLUX2_PIPELINE",),
            },
            "optional": {
                "adapter_name": ("STRING", {"default": ""}),
            }
        }

    def unload_lora(
        self,
        pipeline,
        adapter_name: str = "",
    ):
        try:
            if adapter_name:
                # Unload specific adapter
                if hasattr(pipeline, 'delete_adapters'):
                    pipeline.delete_adapters([adapter_name])
                    logger.info(f"Unloaded LoRA adapter: {adapter_name}")
            else:
                # Unload all adapters
                if hasattr(pipeline, 'unload_lora_weights'):
                    pipeline.unload_lora_weights()
                    logger.info("Unloaded all LoRA adapters")
        except Exception as e:
            logger.warning(f"Error unloading LoRA: {e}")

        return (pipeline,)


class SimpleTunerFlux2LoRASelector:
    """
    Browse and select LoRA files from SimpleTuner output directory.
    Provides a dropdown of available trained LoRAs.
    """

    CATEGORY = "SimpleTuner/Flux2"
    FUNCTION = "select_lora"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_path",)

    @classmethod
    def INPUT_TYPES(cls):
        lora_files = get_lora_files()
        return {
            "required": {
                "lora_name": (lora_files, {"default": lora_files[0] if lora_files else "none"}),
            },
        }

    def select_lora(self, lora_name: str):
        if lora_name == "none":
            return ("",)

        # Resolve to full path
        simpletuner_path = get_simpletuner_path()
        full_path = os.path.join(simpletuner_path, "output", lora_name)

        if os.path.exists(full_path):
            return (full_path,)

        # Try ComfyUI loras folder
        try:
            comfy_path = folder_paths.get_full_path("loras", lora_name)
            if comfy_path:
                return (comfy_path,)
        except Exception:
            pass

        return (lora_name,)


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "SimpleTunerFlux2PipelineLoader": SimpleTunerFlux2PipelineLoader,
    "SimpleTunerFlux2LoRALoader": SimpleTunerFlux2LoRALoader,
    "SimpleTunerFlux2Sampler": SimpleTunerFlux2Sampler,
    "SimpleTunerFlux2LoRAUnloader": SimpleTunerFlux2LoRAUnloader,
    "SimpleTunerFlux2LoRASelector": SimpleTunerFlux2LoRASelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleTunerFlux2PipelineLoader": "Load Flux 2 Pipeline (SimpleTuner)",
    "SimpleTunerFlux2LoRALoader": "Load LoRA (SimpleTuner Flux 2)",
    "SimpleTunerFlux2Sampler": "Sample (SimpleTuner Flux 2)",
    "SimpleTunerFlux2LoRAUnloader": "Unload LoRA (SimpleTuner Flux 2)",
    "SimpleTunerFlux2LoRASelector": "Select LoRA (SimpleTuner Flux 2)",
}

