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

def _build_t5_schedule(args, blueprint):
    """Builds the execution schedule for the T5 model."""
    print("\n-- Building T5 Schedule --")
    model_dir = os.path.dirname(args.t5_path)
    weight_map_path = os.path.join(model_dir, 'model.safetensors.index.json')
    with open(weight_map_path, 'r') as f:
        weight_map = json.load(f)
    loader = SafetensorLoader(model_dir, model_config=weight_map)
    
    quantized = args.quantize
    if quantized:
        print("   Quantizing T5 model...")
        quantized_dir = "quantized_models/t5"
        os.makedirs(quantized_dir, exist_ok=True)
        quantize_and_save_model(loader, quantized_dir)
        loader = SafetensorLoader(quantized_dir)
    
    plan = _get_t5_execution_plan(blueprint, loader, quantized, args.batch_size, args.sequence_length)
    segments = _partition_plan_into_segments(plan, args.gpu_budget * 1024**3)
    print(f"   ...T5 plan created with {len(segments)} segments.")
    return segments

def _build_clip_schedule(args, blueprint):
    """Builds the execution schedule for the CLIP model."""
    print("\n-- Building CLIP Schedule --")
    loader = SafetensorLoader(args.clip_path)
    quantized = args.quantize
    if quantized:
        print("   Quantizing CLIP model...")
        quantized_dir = "quantized_models/clip"
        os.makedirs(quantized_dir, exist_ok=True)
        quantize_and_save_model(loader, quantized_dir)
        loader = SafetensorLoader(quantized_dir)
        
    plan = _get_clip_execution_plan(blueprint, loader, quantized, args.batch_size, args.sequence_length)
    segments = _partition_plan_into_segments(plan, args.gpu_budget * 1024**3)
    print(f"   ...CLIP plan created with {len(segments)} segments.")
    return segments

def _build_vae_schedule(args, blueprint, config):
    """Builds the execution schedule for the VAE model."""
    print("\n-- Building VAE Schedule --")
    loader = SafetensorLoader(args.vae_path)
    plan = _get_vae_execution_plan(blueprint, loader, config, args.batch_size, args.image_size)
    segments = _partition_plan_into_segments(plan, args.gpu_budget * 1024**3)
    print(f"   ...VAE plan created with {len(segments)} segments.")
    return segments

def _build_flux_schedule(args, blueprint):
    """Builds the execution schedule for the FLUX model."""
    print("\n-- Building FLUX Schedule --")
    loader = SafetensorLoader(args.flux_path)
    plan = _get_flux_execution_plan(blueprint, loader, args.batch_size, args.image_size, args.sequence_length)
    segments = _partition_plan_into_segments(plan, args.gpu_budget * 1024**3)
    print(f"   ...FLUX plan created with {len(segments)} segments.")
    return segments

def build_schedule(args):
    """
    Main function to build the execution schedule.
    """
    print("1. Loading model blueprints on 'meta' device...")
    
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

    print("\n2. Building execution schedules for all models...")
    t5_segments = _build_t5_schedule(args, t5_blueprint)
    clip_segments = _build_clip_schedule(args, clip_blueprint)
    vae_segments = _build_vae_schedule(args, vae_blueprint, vae_config)
    flux_segments = _build_flux_schedule(args, flux_blueprint)

    print("\n3. Generating and saving schedules...")
    os.makedirs(args.schedules_dir, exist_ok=True)

    metadata = {
        "gpu_budget_gb": args.gpu_budget,
        "cpu_budget_gb": args.cpu_budget,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "image_size": args.image_size,
        "quantized": args.quantize,
    }

    model_schedules = {
        "t5": t5_segments,
        "clip": clip_segments,
        "vae": vae_segments,
        "flux": flux_segments,
    }

    for name, segments in model_schedules.items():
        schedule_path = os.path.join(args.schedules_dir, f"{name}.json")
        schedule_data = {"metadata": metadata, "segments": segments}
        with open(schedule_path, 'w') as f:
            json.dump(schedule_data, f, indent=4)
        print(f"   - Saved {name.upper()} schedule to {schedule_path}")

    print("\nAll schedules generated successfully.")


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


# This block is already present in the file, but included for diff context
def _get_plan_for_module(layer_name: str, submodule: torch.nn.Module, loader: SafetensorLoader, quantized: bool):
    """
    Helper to calculate weight size for a generic module.
    This is made robust to handle cases where a parameter exists in the blueprint
    but not in the safetensor file (e.g., heterogeneous blocks in FLUX).
    """
    weight_size = 0
    for param_name, param in submodule.named_parameters(recurse=True):
        full_name = f"{layer_name}.{param_name}"
        try:
            if quantized:
                # This assumes a simple quantization scheme for size calculation
                info = loader.get_tensor_info(f"{full_name}.weight")
                weight_size += torch.prod(torch.tensor(info['shape'])) * 1  # INT8
                weight_size += 2 # FP16 scale
            else:
                info = loader.get_tensor_info(full_name)
                weight_size += torch.tensor([], dtype=SAFETENSORS_DTYPE_MAP[info['dtype']]).element_size() * torch.prod(torch.tensor(info['shape']))
        except KeyError:
            # This parameter exists in the blueprint but not the safetensor file.
            # This can happen with models like FLUX where blocks are heterogeneous.
            # We can safely ignore it as it has no weights to load.
            pass
    return int(weight_size)


def _get_t5_execution_plan(blueprint: T5EncoderModel, loader: SafetensorLoader, quantized: bool, batch_size: int, seq_len: int) -> list:
    """
    Generates an execution plan for the T5 model, calculating both weight and
    peak activation sizes for each layer.
    """
    plan = []
    config = blueprint.config
    # Assuming bf16 for activations if weights are quantized, otherwise model's native dtype
    activation_dtype_size = 2 if quantized else torch.tensor([], dtype=config.torch_dtype).element_size()

    layer_order = ['shared'] + [f'encoder.block.{i}' for i in range(len(blueprint.encoder.block))] + ['encoder.final_layer_norm']

    for layer_name in layer_order:
        submodule = blueprint.get_submodule(layer_name)
        weight_size = _get_plan_for_module(layer_name, submodule, loader, quantized)
        
        # Estimate peak activation size
        activation_size = 0
        if 'block' in layer_name:
            activation_size = batch_size * seq_len * config.d_ff * activation_dtype_size
        else:
            activation_size = batch_size * seq_len * config.d_model * activation_dtype_size

        plan.append({
            'name': layer_name,
            'weight_size': weight_size,
            'activation_size': int(activation_size)
        })
            
    return plan


def _get_clip_execution_plan(blueprint: CLIPTextModel, loader: SafetensorLoader, quantized: bool, batch_size: int, seq_len: int) -> list:
    """Generates an execution plan for the CLIP model."""
    plan = []
    config = blueprint.config
    activation_dtype_size = 2 if quantized else torch.tensor([], dtype=torch.float16).element_size() # CLIP is usually fp16/32

    layer_order = ['text_model.embeddings'] + [f'text_model.encoder.layers.{i}' for i in range(len(blueprint.text_model.encoder.layers))] + ['text_model.final_layer_norm']

    for layer_name in layer_order:
        submodule = blueprint.get_submodule(layer_name)
        weight_size = _get_plan_for_module(layer_name, submodule, loader, quantized)
        activation_size = batch_size * seq_len * config.hidden_size * activation_dtype_size
        
        plan.append({
            'name': layer_name,
            'weight_size': weight_size,
            'activation_size': int(activation_size)
        })
    return plan


def _get_vae_execution_plan(blueprint: AutoencoderKL, loader: SafetensorLoader, config: dict, batch_size: int, image_size: int) -> list:
    """Generates an execution plan for the VAE decoder."""
    plan = []
    activation_dtype_size = torch.tensor([], dtype=torch.float16).element_size()
    
    # Simplified plan for VAE decoder
    decoder = blueprint.decoder
    # Corrected: Removed 'post_quant_conv' as it does not exist in the target VAE model.
    layer_order = ['decoder.conv_in']
    
    # Dynamically build layer order based on the blueprint's structure
    if hasattr(decoder, 'up'):
        num_resolutions = len(decoder.up)
        num_res_blocks = len(decoder.up[0].block) if num_resolutions > 0 and hasattr(decoder.up[0], 'block') else 2 # Fallback
        layer_order.extend([f'decoder.up.{i}.block.{j}' for i in reversed(range(num_resolutions)) for j in range(num_res_blocks + 1)])

    layer_order.extend(['decoder.norm_out', 'decoder.conv_out'])

    # Activation size estimation is complex, using a placeholder for now
    latent_dim = (image_size // 8) ** 2
    activation_size = batch_size * latent_dim * config['embed_dim'] * activation_dtype_size

    for layer_name in layer_order:
        try:
            submodule = blueprint.get_submodule(layer_name)
            weight_size = _get_plan_for_module(layer_name, submodule, loader, False)
            plan.append({
                'name': layer_name,
                'weight_size': weight_size,
                'activation_size': int(activation_size) # Placeholder
            })
        except (AttributeError, IndexError):
            # More robustly skip layers that don't exist in the blueprint
            # print(f"Warning: Could not find submodule for layer '{layer_name}'. Skipping.")
            pass
    return plan


def _get_flux_execution_plan(blueprint: Flux, loader: SafetensorLoader, batch_size: int, image_size: int, seq_len: int) -> list:
    """Generates an execution plan for the FLUX model."""
    plan = []
    config = blueprint.params
    activation_dtype_size = torch.tensor([], dtype=torch.float16).element_size()

    layer_order = ['img_in', 'time_in', 'guidance_in', 'vector_in', 'txt_in', 'pe_embedder']
    layer_order.extend([f'double_blocks.{i}' for i in range(len(blueprint.double_blocks))])
    layer_order.extend([f'single_blocks.{i}' for i in range(len(blueprint.single_blocks))])
    layer_order.append('final_layer')

    # Placeholder activation sizes
    latent_dim = (image_size // config.patch_size) ** 2
    activation_size = batch_size * latent_dim * config.hidden_size * activation_dtype_size
    text_activation_size = batch_size * seq_len * config.context_in_dim * activation_dtype_size

    for layer_name in layer_order:
        submodule = blueprint.get_submodule(layer_name)
        weight_size = _get_plan_for_module(layer_name, submodule, loader, False)
        
        current_activation = activation_size
        if 'txt' in layer_name or 'double_blocks' in layer_name:
            current_activation = max(activation_size, text_activation_size)

        plan.append({
            'name': layer_name,
            'weight_size': weight_size,
            'activation_size': int(current_activation) # Placeholder
        })
    return plan


def _partition_plan_into_segments(plan: list, budget_bytes: int) -> list:
    """
    Partitions a flat execution plan into segments that fit within a memory budget,
    accounting for both weights and peak activation memory.
    """
    segments = []
    if not plan:
        return []

    current_segment = []
    current_weight_size = 0
    max_activation_in_segment = 0

    for layer in plan:
        # Check if adding the next layer would violate the budget
        new_weight_total = current_weight_size + layer['weight_size']
        new_max_activation = max(max_activation_in_segment, layer['activation_size'])

        if new_weight_total + new_max_activation > budget_bytes:
            # The current segment is full. Finalize it.
            if not current_segment:
                # This layer by itself is too big. Error out.
                raise MemoryError(
                    f"Layer '{layer['name']}' is too large to fit in the budget. "
                    f"Weights: {layer['weight_size']/1e6:.2f}MB, Activations: {layer['activation_size']/1e6:.2f}MB, "
                    f"Budget: {budget_bytes/1e6:.2f}MB"
                )
            segments.append(current_segment)
            
            # Start a new segment with the current layer
            current_segment = [layer]
            current_weight_size = layer['weight_size']
            max_activation_in_segment = layer['activation_size']
        else:
            # Add the layer to the current segment
            current_segment.append(layer)
            current_weight_size = new_weight_total
            max_activation_in_segment = new_max_activation
    
    # Add the last segment if it's not empty
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
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for activation calculation.")
    parser.add_argument("--sequence_length", type=int, default=77, help="Sequence length for text encoders.")
    parser.add_argument("--image_size", type=int, default=1024, help="Image size for VAE/FLUX activation calculation.")
    parser.add_argument("--schedules_dir", type=str, default="schedules", help="Directory to save the generated schedule files.")
    parser.add_argument("--quantize", action='store_true', help="Enable INT8 weight quantization for T5 and CLIP.")

    args = parser.parse_args()
    build_schedule(args)