import torch
import json
from torch.func import functional_call

import custom_t5_cpp
from safetensor_loader import SafetensorLoader, SAFETENSORS_DTYPE_MAP
from mem_allocator import CUDAMemoryAllocator

class BaseScheduler:
    """
    A generic, schedule-driven runtime engine. It iterates through a list of
    operations defined in a JSON schedule and dispatches them to the
    appropriate implementation provided by a subclass.
    """
    def __init__(self, blueprint: torch.nn.Module, schedule_path: str, model_dir: str, device: str = "cuda", loader: SafetensorLoader = None):
        self.device = device
        self.blueprint = blueprint.to("meta")
        self.model_dir = model_dir

        with open(schedule_path, 'r') as f:
            self.schedule = json.load(f)

        self.is_quantized = self.schedule['metadata'].get('quantized', False)
        self.loader = loader if loader is not None else SafetensorLoader(self.model_dir)
        self.op_map = self._get_op_map()
        self.live_tensors = {}
        self.gpu_handles = {}

    def run_schedule(self, allocator: CUDAMemoryAllocator, **initial_inputs):
        """
        Executes the schedule defined in the JSON file.
        """
        self.live_tensors = {k: v.to(self.device) for k, v in initial_inputs.items()}

        for op in self.schedule['ops']:
            op_name = op['op_name']
            
            # Free tensors that are no longer needed
            for tensor_name in op.get('free', []):
                if tensor_name in self.live_tensors:
                    del self.live_tensors[tensor_name]
                if tensor_name in self.gpu_handles:
                    del self.gpu_handles[tensor_name]

            if op_name not in self.op_map:
                raise NotImplementedError(f"Operation '{op_name}' is not implemented in this scheduler.")
            
            # Execute the operation
            self.op_map[op_name](op, allocator)

        final_output_name = self.schedule['metadata']['output']
        return self.live_tensors[final_output_name]

    def _get_op_map(self):
        """
        Returns a dictionary mapping operation names to their implementation.
        Subclasses must override this.
        """
        raise NotImplementedError

    def _load_weights(self, op_info: dict, allocator: CUDAMemoryAllocator):
        """Loads all weights for a given operation into a single buffer."""
        weight_size = op_info['weight_size']
        weight_buffer = allocator.allocate_buffer(weight_size)
        
        cpu_staging = torch.empty(weight_size, dtype=torch.uint8, device='cpu').pin_memory()

        current_offset = 0
        for tensor_info in op_info['tensors']:
            dtype = SAFETENSORS_DTYPE_MAP[tensor_info['dtype']]
            shape = tensor_info['shape']
            size = torch.tensor([], dtype=dtype).element_size() * torch.prod(torch.tensor(shape))
            
            tensor_slice = cpu_staging.narrow(0, current_offset, int(size)).view(dtype).reshape(shape)
            self.loader.load_tensor_into(tensor_info['name'], tensor_slice)
            current_offset += int(size)
        
        weight_buffer.copy_(cpu_staging, non_blocking=True)
        torch.cuda.current_stream().synchronize() # Ensure copy is complete

        # Create views into the buffer
        current_offset = 0
        for tensor_info in op_info['tensors']:
            dtype = SAFETENSORS_DTYPE_MAP[tensor_info['dtype']]
            shape = tensor_info['shape']
            size = torch.tensor([], dtype=dtype).element_size() * torch.prod(torch.tensor(shape))
            
            view = weight_buffer.narrow(0, current_offset, int(size)).view(dtype).reshape(shape)
            self.gpu_handles[tensor_info['param_name']] = view
            current_offset += int(size)
            
        return weight_buffer


class T5Scheduler(BaseScheduler):
    """
    Schedule-driven runtime engine for the T5 text encoder.
    """
    def __init__(self, blueprint: torch.nn.Module, schedule_path: str, model_dir: str, device: str = "cuda"):
        # T5 has a sharded model structure, so we need to load its weight map
        t5_weight_map_path = f"{model_dir}/model.safetensors.index.json"
        with open(t5_weight_map_path, 'r') as f:
            model_config = json.load(f)
        
        # Initialize the loader with the specific weight map for T5
        loader = SafetensorLoader(model_dir, model_config=model_config)
        super().__init__(blueprint, schedule_path, model_dir, device, loader=loader)

    def _get_op_map(self):
        return {
            "Shared_Embed": self._execute_shared_embed,
            "T5Block": self._execute_t5_block,
            "RMSNorm": self._execute_rmsnorm,
            "Bias_Setup": self._execute_bias_setup,
        }

    def _execute_shared_embed(self, op, allocator):
        weight_buffer = self._load_weights(op, allocator)
        input_tensor = self.live_tensors[op['inputs'][0]]
        
        embedding_weight = self.gpu_handles['weight']
        output = torch.nn.functional.embedding(input_tensor, embedding_weight)
        
        self.live_tensors[op['output']] = output
        del weight_buffer

    def _execute_bias_setup(self, op, allocator):
        hidden_states = self.live_tensors[op['inputs'][0]]
        mask_shape = hidden_states.shape[:2]
        attention_mask = torch.ones(mask_shape, device=self.device)
        extended_attention_mask = attention_mask[:, None, None, :]
        
        model_dtype = hidden_states.dtype
        extended_attention_mask = (1.0 - extended_attention_mask.to(dtype=model_dtype)) * torch.finfo(model_dtype).min
        
        attention_layer = self.blueprint.get_submodule(op['submodule'])
        position_bias = attention_layer.compute_bias(hidden_states.shape[1], hidden_states.shape[1])

        self.live_tensors['position_bias'] = position_bias
        self.live_tensors['extended_attention_mask'] = extended_attention_mask

    def _execute_t5_block(self, op, allocator):
        weight_buffer = self._load_weights(op, allocator)
        workspace_buffer = allocator.allocate_buffer(op['workspace_size'])

        hidden_states = self.live_tensors['hidden_states']
        position_bias = self.live_tensors['position_bias']
        extended_attention_mask = self.live_tensors['extended_attention_mask']
        
        output_tensor = torch.empty_like(hidden_states)

        custom_t5_cpp.t5_block_forward(
            hidden_states, position_bias, extended_attention_mask,
            self.gpu_handles['layer.0.layer_norm.weight'], self.gpu_handles['layer.0.SelfAttention.q.weight'],
            self.gpu_handles['layer.0.SelfAttention.k.weight'], self.gpu_handles['layer.0.SelfAttention.v.weight'],
            self.gpu_handles['layer.0.SelfAttention.o.weight'], self.gpu_handles['layer.1.layer_norm.weight'],
            self.gpu_handles['layer.1.DenseReluDense.wi_0.weight'], self.gpu_handles['layer.1.DenseReluDense.wi_1.weight'],
            self.gpu_handles['layer.1.DenseReluDense.wo.weight'],
            output_tensor,
            workspace_buffer
        )
        
        self.live_tensors['hidden_states'] = output_tensor
        del weight_buffer
        del workspace_buffer

    def _execute_rmsnorm(self, op, allocator):
        weight_buffer = self._load_weights(op, allocator)
        hidden_states = self.live_tensors['hidden_states']
        
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + 1e-6)
        output = hidden_states * self.gpu_handles['weight']

        self.live_tensors[op['output']] = output
        del weight_buffer

class CLIPScheduler(BaseScheduler):
    def __init__(self, blueprint: torch.nn.Module, schedule_path: str, model_dir: str, device: str = "cuda"):
        loader = SafetensorLoader(model_dir, safetensor_filename="model.safetensors")
        super().__init__(blueprint, schedule_path, model_dir, device, loader=loader)

    def _get_op_map(self):
        print("CLIP scheduler not implemented.")
        return {}

class VAEScheduler(BaseScheduler):
    def __init__(self, blueprint: torch.nn.Module, schedule_path: str, model_dir: str, device: str = "cuda"):
        loader = SafetensorLoader(model_dir, safetensor_filename="ae.safetensors")
        super().__init__(blueprint, schedule_path, model_dir, device, loader=loader)

    def _get_op_map(self):
        print("VAE scheduler not implemented.")
        return {}

class FluxScheduler(BaseScheduler):
    def __init__(self, blueprint: torch.nn.Module, schedule_path: str, model_dir: str, device: str = "cuda"):
        loader = SafetensorLoader(model_dir, safetensor_filename="flux1-dev.safetensors")
        super().__init__(blueprint, schedule_path, model_dir, device, loader=loader)

    def _get_op_map(self):
        print("FLUX scheduler not implemented.")
        return {}