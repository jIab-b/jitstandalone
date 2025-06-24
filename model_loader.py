"""
jitstandalone/model_loader.py

Entry point for JIT model loading pipeline.
Loads only metadata from .safetensors using safetensor_metadata,
creates dummy blueprints, and initializes InferenceScheduler for each model.
"""
import json
from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel
import torch
from utils import ops 
from utils.autoencoder import AutoencoderKL
from utils.flux import Flux

from safetensor_loader import SafetensorLoader, extract_safetensor_metadata
from scheduler import T5Scheduler, CLIPScheduler, VAEScheduler, FluxScheduler
from mem_allocator import CUDAMemoryAllocator


from utils.mem_util import print_memory_usage





# This function is obsolete in the new static scheduling architecture.
# def _calculate_max_module_size_for_plan(...)


def _load_vae_model_config():
    """
    Returns the hardcoded VAE configuration.
    """
    return {
        "embed_dim": 16,
        "ddconfig": {
            "double_z": True, "z_channels": 16, "resolution": 1024, "in_channels": 3,
            "out_ch": 3, "ch": 128, "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2,
            "attn_resolutions": [], "dropout": 0.0
        }
    }


def _load_flux_model_config():
    """
    Returns the hardcoded FLUX model configuration.
    """
    return {
        "in_channels": 16, "out_channels": 16, "hidden_size": 2048,
        "depth": 28, "num_heads": 32, "patch_size": 2,
        "context_in_dim": 4096, "vec_in_dim": 512, "mlp_ratio": 4.0,
        "qkv_bias": True, "depth_single_blocks": 28, "axes_dim": [64],
        "theta": 10000, "guidance_embed": True,
    }

def _load_t5_config(path: str):
    """
    Correctly loads a model's config.json file.
    """
    with open(path, 'r') as f:
        return json.load(f)

def _load_clip_config(path: str):
    """
    Infers the CLIPTextConfig parameters by inspecting tensor shapes
    in the safetensor header.
    """
    header = extract_safetensor_metadata(path)
    
    embedding_info = header['text_model.embeddings.token_embedding.weight']
    vocab_size, hidden_size = embedding_info['shape']

    mlp_info = header['text_model.encoder.layers.0.mlp.fc1.weight']
    intermediate_size = mlp_info['shape'][0]

    layer_keys = [k for k in header if k.startswith('text_model.encoder.layers.')]
    layer_indices = {int(k.split('.')[3]) for k in layer_keys}
    num_hidden_layers = max(layer_indices) + 1 if layer_indices else 0

    return {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": 12,
        "projection_dim": 768
    }



def _load_t5_weight_map(path: str):
    """
    Loads the T5 weight map from its specific JSON index file.
    """
    with open(path, 'r') as f:
        return json.load(f)


def load_pipeline(device: str = "cuda"):
    """
    Loads model blueprints and initializes schedulers based on a pre-computed static schedule.
    Returns:
        dict: mapping model names to InferenceScheduler instances.
    """
    # Point to original jitloader directory for model weights
    clip_path = "../../ComfyUI/jitloader/clip/model.safetensors"
    t5_path = "../../ComfyUI/jitloader/t5/config.json"
    vae_path = "../../ComfyUI/jitloader/vae/ae.safetensors"
    flux_path = "../../ComfyUI/jitloader/transformer/flux1-dev.safetensors"

    # --- Create Model Blueprints on 'meta' device ---
    # clip_config = CLIPTextConfig(**_load_clip_config(clip_path))
    # clip_blueprint = CLIPTextModel(clip_config).to("meta")

    # print('loaded clip')
    # t5_config_dict = _load_t5_config(t5_path)
    # t5_config = T5Config.from_dict(t5_config_dict)
    # t5_config.use_cache = False
    # t5_blueprint = T5EncoderModel(t5_config).to("meta")

    # print('loaded t5')
    # vae_config = _load_vae_model_config()
    # vae_blueprint = AutoencoderKL(**vae_config).to("meta")

    # print('loaded vae')
    # flux_config = _load_flux_model_config()

    # flux_blueprint = Flux(**flux_config, operations=ops.disable_weight_init).to("meta")



# --- Create Model Blueprints on 'meta' device ---
    with torch.device("meta"):
        clip_config = CLIPTextConfig(**_load_clip_config(clip_path))
        clip_blueprint = CLIPTextModel(clip_config)

        t5_config_dict = _load_t5_config(t5_path)
        t5_config = T5Config.from_dict(t5_config_dict)
        t5_config.use_cache = False
        t5_blueprint = T5EncoderModel(t5_config)

        vae_config = _load_vae_model_config()
        vae_blueprint = AutoencoderKL(**vae_config)

        flux_config = _load_flux_model_config()
        # The ops.disable_weight_init is a good custom safety, but the 'meta'
        # device context is the primary mechanism that prevents allocation.
        flux_blueprint = Flux(**flux_config, operations=ops.disable_weight_init)


    print('loaded all models')
    # --- Initialize Schedulers based on the static schedule ---
    schedules_dir = "schedules"
    
    print("Initializing pipeline from static schedules...")
    
    # 1. Read a schedule to get the budget and initialize the allocator
    # (Assuming all schedules share the same budget metadata)
    with open(f"{schedules_dir}/t5.json", 'r') as f:
        schedule_meta = json.load(f)['metadata']
    gpu_budget_bytes = schedule_meta['gpu_budget_gb'] * 1024**3
    
    cuda_allocator = CUDAMemoryAllocator(gpu_budget_bytes, device=torch.device(device))
    
    # 2. CRITICAL: Register our allocator as the global default for PyTorch
    torch.cuda.memory.change_allocator(cuda_allocator.malloc, cuda_allocator.free)
    
    print(f"Custom CUDA memory allocator registered with a budget of {gpu_budget_bytes/1e9:.2f} GB.")

    # 3. Initialize all schedulers with their specific schedule files
    t5_model_dir = "../../ComfyUI/jitloader/t5"
    t5_scheduler = T5Scheduler(t5_blueprint, f"{schedules_dir}/t5.json", t5_model_dir, device)
    print("T5 scheduler initialized.")

    clip_model_dir = "../../ComfyUI/jitloader/clip"
    clip_scheduler = CLIPScheduler(clip_blueprint, f"{schedules_dir}/clip.json", clip_model_dir, device)
    print("CLIP scheduler initialized.")
    
    vae_model_dir = "../../ComfyUI/jitloader/vae"
    vae_scheduler = VAEScheduler(vae_blueprint, f"{schedules_dir}/vae.json", vae_model_dir, device)
    print("VAE scheduler initialized.")
    
    flux_model_dir = "../../ComfyUI/jitloader/transformer"
    flux_scheduler = FluxScheduler(flux_blueprint, f"{schedules_dir}/flux.json", flux_model_dir, device)
    print("FLUX scheduler initialized.")


    # The allocator is now global, so we don't need to return it separately.
    return {
        "clip": clip_scheduler,
        "t5": t5_scheduler,
        "vae": vae_scheduler,
        "flux": flux_scheduler,
    }


if __name__ == "__main__":
    schedulers = load_pipeline()
    print("Schedulers initialized:", list(schedulers.keys()))
    # The custom allocator is now active. All subsequent CUDA operations are budgeted.