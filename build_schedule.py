import argparse
import json
import os
import torch
from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel
from safetensors.torch import save_file

from utils import ops
from utils.autoencoder import AutoencoderKL
from utils.flux import Flux
from safetensor_loader import SafetensorLoader, extract_safetensor_metadata, SAFETENSORS_DTYPE_MAP

def _load_vae_model_config():
    """Returns the hardcoded VAE configuration."""
    return {
        "embed_dim": 16,
        "ddconfig": {
            "double_z": True, "z_channels": 16, "resolution": 1024, "in_channels": 3,
            "out_ch": 3, "ch": 128, "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2,
            "attn_resolutions": [], "dropout": 0.0
        }
    }

def _load_flux_model_config():
    """Returns the hardcoded FLUX model configuration."""
    return {
        "in_channels": 16, "out_channels": 16, "hidden_size": 2048,
        "depth": 28, "num_heads": 32, "patch_size": 2,
        "context_in_dim": 4096, "vec_in_dim": 512, "mlp_ratio": 4.0,
        "qkv_bias": True, "depth_single_blocks": 28, "axes_dim": [64],
        "theta": 10000, "guidance_embed": True,
    }

def _load_t5_config(path: str):
    """Correctly loads a model's config.json file."""
    with open(path, 'r') as f:
        return json.load(f)

def _load_clip_config(path: str):
    """Infers the CLIPTextConfig parameters by inspecting tensor shapes."""
    header = extract_safetensor_metadata(path)
    embedding_info = header['text_model.embeddings.token_embedding.weight']
    vocab_size, hidden_size = embedding_info['shape']
    mlp_info = header['text_model.encoder.layers.0.mlp.fc1.weight']
    intermediate_size = mlp_info['shape'][0]
    layer_keys = [k for k in header if k.startswith('text_model.encoder.layers.')]
    layer_indices = {int(k.split('.')[3]) for k in layer_keys}
    num_hidden_layers = max(layer_indices) + 1 if layer_indices else 0
    return {
        "vocab_size": vocab_size, "hidden_size": hidden_size,
        "intermediate_size": intermediate_size, "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": 12, "projection_dim": 768
    }

def build_schedule(args):
    """
    Main function to build the execution schedule.
    """
    print("1. Loading model blueprints on 'meta' device...")
    
    # Reuse blueprint loading logic from model_loader.py
    with torch.device("meta"):
        clip_config = CLIPTextConfig(**_load_clip_config(args.clip_path))
        clip_blueprint = CLIPTextModel(clip_config)

        t5_config_dict = _load_t5_config(args.t5_path)
        t5_config = T5Config.from_dict(t5_config_dict)
        t5_config.use_cache = False
        t5_blueprint = T5EncoderModel(t5_config)

        vae_config = _load_vae_model_config()
        vae_blueprint = AutoencoderKL(**vae_config)

        flux_config = _load_flux_model_config()
        flux_blueprint = Flux(**flux_config, operations=ops.disable_weight_init)

    print("   ...blueprints loaded successfully.")

    # --- T5 Model Processing ---
    t5_model_dir = os.path.dirname(args.t5_path)
    t5_weight_map_path = os.path.join(t5_model_dir, 'model.safetensors.index.json')
    with open(t5_weight_map_path, 'r') as f:
        t5_weight_map = json.load(f)
    
    t5_loader = SafetensorLoader(t5_model_dir, model_config=t5_weight_map)
    
    is_quantized = False
    if args.quantize:
        print("\n2. Quantizing T5 model to INT8...")
        quantized_t5_dir = os.path.join(os.path.dirname(args.output_path), "t5_quantized")
        os.makedirs(quantized_t5_dir, exist_ok=True)
        quantize_and_save_model(t5_loader, quantized_t5_dir)
        
        # Create a new loader for the quantized weights
        quantized_weight_map = {k: f"model_{i}.safetensors" for i, (k, v) in enumerate(t5_weight_map.items())} # Dummy map for now
        # This part needs to be smarter if we shard the quantized output
        t5_loader = SafetensorLoader(quantized_t5_dir)
        is_quantized = True
        print("   ...quantization complete.")

    print("\n3. Analyzing T5 model graph and calculating layer sizes...")
    t5_plan = _get_t5_execution_plan(t5_blueprint, t5_loader, is_quantized)
    print(f"   ...T5 plan created with {len(t5_plan)} layers.")

    print("\n4. Partitioning T5 plan into executable segments...")
    gpu_budget_bytes = args.gpu_budget * 1024**3
    t5_segments = _partition_plan_into_segments(t5_plan, gpu_budget_bytes)
    print(f"   ...T5 plan partitioned into {len(t5_segments)} segments.")

    print("\n5. Generating and saving schedule...")

    schedule = {
        "metadata": {
            "gpu_budget_gb": args.gpu_budget,
            "cpu_budget_gb": args.cpu_budget,
            "t5_quantized": is_quantized,
        },
        "t5_segments": t5_segments,
    }

    with open(args.output_path, 'w') as f:
        json.dump(schedule, f, indent=4)
    
    print(f"\nGenerated schedule for T5 model at: {args.output_path}")


def quantize_and_save_model(loader: SafetensorLoader, output_dir: str):
    """
    Loads all tensors from a loader, quantizes them to INT8, and saves them
    to a new set of safetensor files in the output directory.
    """
    # This is a simplified implementation that puts all tensors in one file.
    # A production system might shard this.
    quantized_weights = {}
    metadata = {}

    all_tensor_names = loader.weight_map.keys() if loader.weight_map else loader.headers['__single__'].keys()

    for name in all_tensor_names:
        if name == '__metadata__':
            continue
        
        info = loader.get_tensor_info(name)
        dtype = SAFETENSORS_DTYPE_MAP[info['dtype']]
        shape = info['shape']
        
        # Load original tensor into a temporary buffer
        original_tensor = torch.empty(shape, dtype=dtype)
        loader.load_tensor_into(name, original_tensor)
        
        # Perform symmetric INT8 quantization
        scale = original_tensor.abs().max() / 127.0
        quantized_tensor = (original_tensor / scale).round().to(torch.int8)
        
        # Store quantized tensor and its scale
        quantized_weights[f"{name}.weight"] = quantized_tensor
        quantized_weights[f"{name}.scale"] = scale.to(torch.float16)

    output_path = os.path.join(output_dir, "model.safetensors")
    save_file(quantized_weights, output_path)


def _get_t5_execution_plan(blueprint: T5EncoderModel, loader: SafetensorLoader, quantized: bool) -> list:
    """
    Generates an execution plan for the T5 model, calculating layer sizes
    based on whether the weights are quantized or not.
    """
    plan = []
    layer_order = ['shared'] + [f'encoder.block.{i}' for i in range(len(blueprint.encoder.block))] + ['encoder.final_layer_norm']

    for layer_name in layer_order:
        submodule = blueprint.get_submodule(layer_name)
        layer_size = 0
        
        for param_name, param in submodule.named_parameters(recurse=True):
            full_name = f"{layer_name}.{param_name}"
            if quantized:
                # For quantized weights, size is int8 data + fp16 scale
                info = loader.get_tensor_info(f"{full_name}.weight")
                layer_size += torch.prod(torch.tensor(info['shape'])) * 1  # INT8
                layer_size += 2 # FP16 scale
            else:
                info = loader.get_tensor_info(full_name)
                layer_size += torch.tensor([], dtype=SAFETENSORS_DTYPE_MAP[info['dtype']]).element_size() * torch.prod(torch.tensor(info['shape']))
        
        plan.append({'name': layer_name, 'size': int(layer_size)})
            
    return plan


def _partition_plan_into_segments(plan: list, budget_bytes: int) -> list:
    """
    Partitions a flat execution plan into segments that fit within a memory budget.
    This is a greedy implementation based on weight sizes only.
    """
    segments = []
    current_segment = []
    current_segment_size = 0

    # A simple safety margin (e.g., 10%) to leave room for activations
    safety_margin = 0.10
    effective_budget = budget_bytes * (1 - safety_margin)

    for layer in plan:
        layer_size = layer['size']
        
        if layer_size > effective_budget:
            print(f"Warning: Layer '{layer['name']}' with size {layer_size/1e6:.2f}MB exceeds the effective budget of {effective_budget/1e6:.2f}MB. It will be placed in its own segment.")
            if current_segment:
                segments.append(current_segment)
            segments.append([layer])
            current_segment = []
            current_segment_size = 0
            continue

        if current_segment_size + layer_size > effective_budget:
            segments.append(current_segment)
            current_segment = [layer]
            current_segment_size = layer_size
        else:
            current_segment.append(layer)
            current_segment_size += layer_size
    
    if current_segment:
        segments.append(current_segment)
        
    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a static execution schedule for the FLUX pipeline.")
    
    # --- Model Paths ---
    parser.add_argument("--clip_path", type=str, default="../../ComfyUI/jitloader/clip/model.safetensors", help="Path to CLIP model safetensors.")
    parser.add_argument("--t5_path", type=str, default="../../ComfyUI/jitloader/t5/config.json", help="Path to T5 model config.json.")
    parser.add_argument("--vae_path", type=str, default="../../ComfyUI/jitloader/vae/ae.safetensors", help="Path to VAE model safetensors.")
    parser.add_argument("--flux_path", type=str, default="../../ComfyUI/jitloader/transformer/flux1-dev.safetensors", help="Path to FLUX model safetensors.")

    # --- Budget and Output ---
    parser.add_argument("--gpu_budget", type=float, default=6.0, help="GPU VRAM budget in GB.")
    parser.add_argument("--cpu_budget", type=float, default=12.0, help="CPU RAM budget in GB.")
    parser.add_argument("--output_path", type=str, default="schedule.json", help="Path to save the generated schedule file.")
    parser.add_argument("--quantize", action='store_true', help="Enable INT8 weight quantization for encoders.")

    args = parser.parse_args()
    build_schedule(args)