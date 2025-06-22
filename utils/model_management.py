# jitstandalone/utils/model_management.py
# A simplified, standalone version of ComfyUI's model_management.py

import torch
import logging

OOM_EXCEPTION = torch.cuda.OutOfMemoryError

def get_free_memory(device="cuda"):
    """Returns free VRAM in bytes."""
    if torch.cuda.is_available():
        return torch.cuda.mem_get_info(device)[0]
    else:
        return 0  # No GPU

def soft_empty_cache(force=False):
    """A wrapper for torch.cuda.empty_cache()."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def xformers_enabled_vae():
    """Checks if xformers is installed and returns True."""
    try:
        import xformers
        return True
    except ImportError:
        return False

def pytorch_attention_enabled_vae():
    """Returns True, assuming modern PyTorch is used."""
    return True

def is_device_mps(device):
    return device.type == 'mps'

def is_intel_xpu():
    return hasattr(torch, 'xpu') and torch.xpu.is_available()

def is_directml_enabled():
    return hasattr(torch, 'version') and hasattr(torch.version, 'directml') and torch.version.directml is not None

def cast_to(weight, dtype, device, non_blocking=False, copy=False, stream=None):
    """A simplified cast_to function."""
    if weight is None:
        return None
    if copy:
        return weight.to(dtype=dtype, device=device, non_blocking=non_blocking).clone()
    else:
        return weight.to(dtype=dtype, device=device, non_blocking=non_blocking)

def get_offload_stream(device):
    return None

def device_supports_non_blocking(device):
    return device.type == 'cuda'

def sync_stream(device, stream):
    if stream is not None:
        stream.synchronize()