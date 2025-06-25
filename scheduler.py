import torch
import json
from torch.func import functional_call

import custom_t5_cpp
from safetensor_loader import SafetensorLoader, SAFETENSORS_DTYPE_MAP

class BaseScheduler:
    """
    Base class for runtime engines that execute a pre-computed schedule.
    """
    def __init__(self, blueprint: torch.nn.Module, schedule_path: str, model_dir: str, device: str = "cuda"):
        self.device = device
        self.blueprint = blueprint.to("meta")
        self.model_dir = model_dir

        with open(schedule_path, 'r') as f:
            self.schedule = json.load(f)

        self.is_quantized = self.schedule['metadata'].get('quantized', False)
        
        # The SafetensorLoader is now simplified and only needs the directory
        self.loader = SafetensorLoader(self.model_dir)
        
        self.stream_copy = torch.cuda.Stream()
        self.stream_comp = torch.cuda.Stream()

class T5Scheduler(BaseScheduler):
    """
    Runtime engine for the T5 text encoder.
    """
    def __init__(self, blueprint: torch.nn.Module, schedule_path: str, model_dir: str, device: str = "cuda"):
        # T5 has a sharded model structure, so we need to load its weight map
        t5_weight_map_path = f"{model_dir}/model.safetensors.index.json"
        with open(t5_weight_map_path, 'r') as f:
            model_config = json.load(f)
        
        # Initialize the loader with the specific weight map for T5
        self.loader = SafetensorLoader(model_dir, model_config=model_config)
        super().__init__(blueprint, schedule_path, model_dir, device)


    def run_encoder_inference(self, input_ids):
        hidden_states = input_ids.to(self.device)
        position_bias = None
        extended_attention_mask = None
        
        segments = self.schedule['segments']
        
        for seg in segments:
            # This simplified loop executes segments synchronously.
            # A fully optimized version would overlap the H2D copy of segment N+1
            # with the computation of segment N.

            # --- 1. Load and Transfer Segment Weights ---
            with torch.cuda.stream(self.stream_copy):
                # Create a single buffer on CPU and GPU for the entire segment's weights
                segment_weight_size = sum(layer['weight_size'] for layer in seg)
                cpu_weight_buffer = torch.empty(segment_weight_size, dtype=torch.uint8, device='cpu').pin_memory()
                gpu_weight_buffer = torch.empty(segment_weight_size, dtype=torch.uint8, device=self.device)
                
                # Load all weights for the segment into the CPU buffer
                current_offset = 0
                for layer_info in seg:
                    for tensor_info in layer_info['tensors']:
                        full_name = tensor_info['name']
                        dtype = SAFETENSORS_DTYPE_MAP[tensor_info['dtype']]
                        shape = tensor_info['shape']
                        size = torch.tensor([], dtype=dtype).element_size() * torch.prod(torch.tensor(shape))
                        
                        tensor_slice = cpu_weight_buffer.narrow(0, current_offset, int(size)).view(dtype).reshape(shape)
                        self.loader.load_tensor_into(full_name, tensor_slice)
                        current_offset += int(size)
                
                # Asynchronously copy the entire weight block to the GPU
                gpu_weight_buffer.copy_(cpu_weight_buffer, non_blocking=True)

            # --- 2. Compute Segment ---
            self.stream_comp.wait_stream(self.stream_copy)
            with torch.cuda.stream(self.stream_comp):
                current_offset = 0
                for layer_info in seg:
                    layer_name = layer_info['name']
                    submodule = self.blueprint.get_submodule(layer_name)
                    
                    # Create handles by slicing the single GPU buffer for the segment
                    gpu_handles = {}
                    # Reconstruct the state_dict for functional_call from the buffer
                    param_idx = 0
                    for param_name, param in submodule.named_parameters(recurse=True):
                        tensor_info = layer_info['tensors'][param_idx]
                        dtype = SAFETENSORS_DTYPE_MAP[tensor_info['dtype']]
                        shape = tensor_info['shape']
                        size = torch.tensor([], dtype=dtype).element_size() * torch.prod(torch.tensor(shape))

                        tensor_slice = gpu_weight_buffer.narrow(0, current_offset, int(size)).view(dtype).reshape(shape)
                        gpu_handles[param_name] = tensor_slice
                        current_offset += int(size)
                        param_idx += 1

                    # --- Actual forward pass logic ---
                    # (Activations are now implicitly managed by the global custom allocator)
                    if position_bias is None and 'block' in layer_name:
                        attention_layer = submodule.layer[0].SelfAttention
                        mask_shape = hidden_states.shape[:2]
                        attention_mask = torch.ones(mask_shape, device=self.device)
                        extended_attention_mask = attention_mask[:, None, None, :]
                        model_dtype = next(iter(gpu_handles.values())).dtype
                        extended_attention_mask = (1.0 - extended_attention_mask.to(dtype=model_dtype)) * torch.finfo(model_dtype).min
                        position_bias = attention_layer.compute_bias(hidden_states.shape[1], hidden_states.shape[1])

                    if layer_name == 'shared':
                        hidden_states = functional_call(submodule, gpu_handles, (hidden_states,))
                    elif 'block' in layer_name:
                        output_tensor = torch.empty_like(hidden_states)
                        custom_t5_cpp.t5_block_forward(
                            hidden_states, position_bias, extended_attention_mask,
                            gpu_handles['layer.0.layer_norm.weight'], gpu_handles['layer.0.SelfAttention.q.weight'],
                            gpu_handles['layer.0.SelfAttention.k.weight'], gpu_handles['layer.0.SelfAttention.v.weight'],
                            gpu_handles['layer.0.SelfAttention.o.weight'], gpu_handles['layer.1.layer_norm.weight'],
                            gpu_handles['layer.1.DenseReluDense.wi_0.weight'], gpu_handles['layer.1.DenseReluDense.wi_1.weight'],
                            gpu_handles['layer.1.DenseReluDense.wo.weight'], output_tensor
                        )
                        hidden_states = output_tensor
                    else: # final_layer_norm
                        hidden_states = functional_call(submodule, gpu_handles, (hidden_states,))

            self.stream_comp.synchronize()
            
            # --- 3. Free Memory ---
            # The GPU buffer will be freed by Python's garbage collector when it goes
            # out of scope at the end of the loop, and our custom allocator will
            # correctly track this deallocation.
            del gpu_weight_buffer
            del cpu_weight_buffer

        return hidden_states


class CLIPScheduler(BaseScheduler):
    """
    Runtime engine for the CLIP text encoder.
    """
    def run_encoder_inference(self, input_ids):
        # This logic will be very similar to T5's, but without the custom kernel.
        print("CLIP inference not yet implemented.")
        return None


class VAEScheduler(BaseScheduler):
    """
    Runtime engine for the VAE decoder.
    """
    def run_decoder_inference(self, latents):
        print("VAE inference not yet implemented.")
        return None


class FluxScheduler(BaseScheduler):
    """
    Runtime engine for the FLUX model.
    """
    def run_inference(self, *args, **kwargs):
        print("FLUX inference not yet implemented.")
        return None