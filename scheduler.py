import torch
import json
from torch.func import functional_call

import custom_t5_cpp
from safetensor_loader import SafetensorLoader, SAFETENSORS_DTYPE_MAP
from mem_allocator import BudgetedAllocator

class T5Scheduler:
    """
    Runtime engine for the T5 text encoder that executes a pre-computed schedule.
    """
    def __init__(self, blueprint: torch.nn.Module, allocator: BudgetedAllocator, schedule_path: str, model_dir: str, device: str = "cuda"):
        self.device = device
        self.blueprint = blueprint.to("meta")
        self.allocator = allocator
        
        with open(schedule_path, 'r') as f:
            self.schedule = json.load(f)
        
        self.is_quantized = self.schedule['metadata'].get('t5_quantized', False)
        
        if self.is_quantized:
            # For quantized models, the schedule builder saves them in a known location
            quantized_dir = "t5_quantized"
            self.loader = SafetensorLoader(quantized_dir)
        else:
            t5_weight_map_path = f"{model_dir}/model.safetensors.index.json"
            with open(t5_weight_map_path, 'r') as f:
                model_config = json.load(f)
            self.loader = SafetensorLoader(model_dir, model_config=model_config)
        
        self.stream_copy = torch.cuda.Stream()
        self.stream_comp = torch.cuda.Stream()

    def _load_segment_to_cpu(self, segment: list, cpu_buffer: torch.Tensor):
        """Loads all weights for a segment into a single pre-allocated CPU buffer."""
        current_offset = 0
        for layer_info in segment:
            layer_name = layer_info['name']
            submodule = self.blueprint.get_submodule(layer_name)
            for param_name, param in submodule.named_parameters(recurse=True):
                full_name = f"{layer_name}.{param_name}"
                info = self.loader.get_tensor_info(full_name)
                dtype = SAFETENSORS_DTYPE_MAP[info['dtype']]
                size = torch.tensor([], dtype=dtype).element_size() * torch.prod(torch.tensor(info['shape']))
                
                tensor_slice = cpu_buffer.narrow(0, current_offset, int(size)).view(dtype).reshape(param.shape)
                self.loader.load_tensor_into(full_name, tensor_slice)
                current_offset += int(size)
        return

    def run_encoder_inference(self, input_ids):
        hidden_states = input_ids.to(self.device)
        position_bias = None
        extended_attention_mask = None
        
        segments = self.schedule['t5_segments']
        
        # This is a simplified synchronous implementation of the two-stream design.
        # A fully async version would prefetch the next segment while computing the current one.
        for seg in segments:
            segment_size = sum(layer['size'] for layer in seg)

            # --- Load and Transfer ---
            with torch.cuda.stream(self.stream_copy):
                cpu_buffer = self.allocator.malloc_cpu(segment_size)
                gpu_buffer = self.allocator.malloc_gpu(segment_size)
                
                self._load_segment_to_cpu(seg, cpu_buffer)
                gpu_buffer.copy_(cpu_buffer, non_blocking=True)

            # --- Compute ---
            self.stream_comp.wait_stream(self.stream_copy)
            with torch.cuda.stream(self.stream_comp):
                current_offset = 0
                for layer_info in seg:
                    layer_name = layer_info['name']
                    submodule = self.blueprint.get_submodule(layer_name)
                    
                    # Create handles by slicing the single GPU buffer for the segment
                    gpu_handles = {}
                    for param_name, param in submodule.named_parameters(recurse=True):
                        full_name = f"{layer_name}.{param_name}"
                        info = self.loader.get_tensor_info(full_name)
                        dtype = SAFETENSORS_DTYPE_MAP[info['dtype']]
                        size = torch.tensor([], dtype=dtype).element_size() * torch.prod(torch.tensor(info['shape']))
                        
                        tensor_slice = gpu_buffer.narrow(0, current_offset, int(size)).view(dtype).reshape(param.shape)
                        gpu_handles[param_name] = tensor_slice
                        current_offset += int(size)

                    # --- Actual forward pass logic ---
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
                        # This assumes a workspace allocator is integrated separately or handled by PyTorch
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
            
            # --- Free Memory ---
            self.allocator.free_cpu(cpu_buffer)
            self.allocator.free_gpu(gpu_buffer)

        return hidden_states