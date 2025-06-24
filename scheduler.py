# jitstandalone/scheduler.py

import torch
from torch import nn
from torch.func import functional_call
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import custom_t5_cpp
from safetensor_loader import SafetensorLoader, SAFETENSORS_DTYPE_MAP
from mem_allocator import MemoryAllocator
from utils.flux_layers import timestep_embedding
from einops import rearrange, repeat
from utils import common as standalone_common

class BaseScheduler:
    """
    Base class for all schedulers, providing asynchronous pre-fetching,
    manual module materialization, and CUDA graph integration.
    """
    def __init__(self, path: str, model_blueprint: nn.Module, allocator: MemoryAllocator, device: str = "cuda", model_config: dict = None, quant_config: str = None, prefetch_depth: int = 2):
        self.device = device
        self.loader = SafetensorLoader(path, model_config=model_config, quant_config=quant_config)
        self.blueprint = model_blueprint.to("meta")
        self.allocator = allocator
        self.prefetch_depth = prefetch_depth
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.prefetch_queue = Queue(maxsize=prefetch_depth)
        self.execution_plan = []
        
        self.cuda_graphs = {}

        if self.device == 'cuda':
            self.stream_a = torch.cuda.Stream()
            self.stream_b = torch.cuda.Stream()

    def _prepare_execution_plan(self, plan: list):
        self.execution_plan = plan
        for layer_name in self.execution_plan[:self.prefetch_depth]:
            future = self.executor.submit(self._load_layer_to_pool, layer_name)
            self.prefetch_queue.put(future)

    def _load_layer_to_pool(self, layer_name: str):
        """
        Loads all tensors for a layer into pre-allocated pinned CPU memory.
        Returns a dictionary of handles (allocated tensors) and their names.
        """
        self.allocator.reset('cpu')
        handles = {}
        submodule = self.blueprint.get_submodule(layer_name)

        for name, param in submodule.named_parameters(recurse=True):
            full_name = f"{layer_name}.{name}"
            info = self.loader.get_tensor_info(full_name)
            dtype = SAFETENSORS_DTYPE_MAP[info['dtype']]
            size = torch.tensor([], dtype=dtype).element_size() * torch.prod(torch.tensor(info['shape']))

            cpu_buffer = self.allocator.allocate(size, 'cpu').view(dtype).reshape(param.shape)
            self.loader.load_tensor_into(full_name, cpu_buffer)
            handles[name] = cpu_buffer

        for name, buffer in submodule.named_buffers(recurse=True):
            full_name = f"{layer_name}.{name}"
            try:
                info = self.loader.get_tensor_info(full_name)
                dtype = SAFETENSORS_DTYPE_MAP[info['dtype']]
                size = torch.tensor([], dtype=dtype).element_size() * torch.prod(torch.tensor(info['shape']))
                
                cpu_buffer = self.allocator.allocate(size, 'cpu').view(dtype).reshape(buffer.shape)
                self.loader.load_tensor_into(full_name, cpu_buffer)
                handles[name] = cpu_buffer
            except KeyError:
                pass
        return layer_name, handles

    def _load_layer_to_gpu(self, layer_name_and_handles, buffer_id: int, stream: torch.cuda.Stream):
        """
        Copies a layer's weights from CPU to the designated GPU weight buffer.
        Returns handles to the GPU tensors.
        """
        layer_name, cpu_handles = layer_name_and_handles
        gpu_handles = {}
        with torch.cuda.stream(stream):
            for name, cpu_tensor in cpu_handles.items():
                gpu_tensor = self.allocator.allocate(
                    cpu_tensor.nbytes, 'gpu_weights', buffer_id=buffer_id
                ).view(cpu_tensor.dtype).reshape(cpu_tensor.shape)
                gpu_tensor.copy_(cpu_tensor, non_blocking=True)
                gpu_handles[name] = gpu_tensor
        return layer_name, gpu_handles


class FluxScheduler(BaseScheduler):
    """
    Scheduler for the main FLUX transformer model.
    """
    def run_inference(self, x, timestep, context, y=None, guidance=None, **kwargs):
        plan = ['img_in', 'time_in']
        if self.blueprint.params.guidance_embed and guidance is not None:
            plan.append('guidance_in')
        plan.extend(['vector_in', 'txt_in', 'pe_embedder'])
        plan.extend([f'double_blocks.{i}' for i in range(len(self.blueprint.double_blocks))])
        plan.extend([f'single_blocks.{i}' for i in range(len(self.blueprint.single_blocks))])
        plan.append('final_layer')
        self._prepare_execution_plan(plan)
        
        bs, c, h, w = x.shape
        patch_size = self.blueprint.patch_size
        x = standalone_common.pad_to_patch_size(x, (patch_size, patch_size))
        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        h_len, w_len = h // patch_size, w // patch_size
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 1] = torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids[:, :, 2] = torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        txt = context
        txt_ids = torch.zeros((bs, txt.shape[1], 3), device=x.device, dtype=x.dtype)
        
        if y is None:
            y = torch.zeros((x.shape[0], self.blueprint.params.vec_in_dim), device=x.device, dtype=x.dtype)

        streams = [self.stream_a, self.stream_b]
        
        # Initial fetch and load to GPU
        cpu_future = self.prefetch_queue.get()
        gpu_future = self.executor.submit(self._load_layer_to_gpu, cpu_future.result(), 0, streams[0])

        for i in range(len(plan)):
            buffer_id = i % 2
            stream = streams[buffer_id]
            
            # Wait for the current layer's weights to be loaded to the GPU
            layer_name, gpu_handles = gpu_future.result()
            
            # Prefetch the next layer to CPU, and then schedule its copy to the other GPU buffer
            if i + 1 < len(plan):
                cpu_future = self.prefetch_queue.get()
                gpu_future = self.executor.submit(self._load_layer_to_gpu, cpu_future.result(), (i + 1) % 2, streams[(i + 1) % 2])

            # Now, wait for the previous computation on the current stream to finish before resetting buffers
            stream.synchronize()
            self.allocator.reset('gpu_weights', buffer_id=buffer_id)
            
            # Start the next CPU load for a future layer
            if i + self.prefetch_depth < len(plan):
                future = self.executor.submit(self._load_layer_to_pool, plan[i + self.prefetch_depth])
                self.prefetch_queue.put(future)

            with self.allocator.scope() as workspace, torch.cuda.stream(stream):
                submodule = self.blueprint.get_submodule(layer_name)
                
                # NOTE: Stateless execution requires manually allocating output tensors.
                # This is a conceptual implementation. The exact shapes need to be derived
                # from the model architecture.
                if layer_name == 'img_in':
                    img = functional_call(submodule, gpu_handles, (img,))
                elif layer_name == 'time_in':
                    vec = functional_call(submodule, gpu_handles, (timestep_embedding(timestep, 256).to(img.dtype),))
                elif layer_name == 'guidance_in':
                    guidance_embed = functional_call(submodule, gpu_handles, (timestep_embedding(guidance, 256).to(img.dtype),))
                    vec = vec + guidance_embed
                elif layer_name == 'vector_in':
                    y_embed = functional_call(submodule, gpu_handles, (y[:,:self.blueprint.params.vec_in_dim],))
                    vec = vec + y_embed
                elif layer_name == 'txt_in':
                    txt = functional_call(submodule, gpu_handles, (txt,))
                elif layer_name == 'pe_embedder':
                    ids = torch.cat((txt_ids, img_ids), dim=1)
                    pe = functional_call(submodule, gpu_handles, (ids,))
                elif 'double_blocks' in layer_name:
                    img, txt = functional_call(submodule, gpu_handles, args=(img, txt, vec, pe))
                elif 'single_blocks' in layer_name:
                    if i > 0 and 'double_blocks' in plan[i-1]: # First single block
                        img = torch.cat((txt, img), 1)
                    img = functional_call(submodule, gpu_handles, args=(img,), kwargs={'vec': vec, 'pe': pe})
                elif layer_name == 'final_layer':
                    img = img[:, txt.shape[1] :, ...]
                    img = functional_call(submodule, gpu_handles, args=(img, vec))

        torch.cuda.synchronize()
        return rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=patch_size, pw=patch_size)[:,:,:h,:w]

class VAEScheduler(BaseScheduler):
    """
    Scheduler for the VAE model.
    """
    def run_decoder_inference(self, latents):
        latents = latents / self.blueprint.config.get("scaling_factor", 0.18215)
        
        plan = ['post_quant_conv', 'decoder.conv_in', 'decoder.mid.block_1', 'decoder.mid.attn_1', 'decoder.mid.block_2']
        for i in reversed(range(self.blueprint.decoder.num_resolutions)):
            for j in range(self.blueprint.decoder.num_res_blocks + 1):
                plan.append(f'decoder.up.{i}.block.{j}')
            if self.blueprint.decoder.up[i].attn:
                plan.append(f'decoder.up.{i}.attn.0')
            if i != 0:
                plan.append(f'decoder.up.{i}.upsample')
        plan.extend(['decoder.norm_out', 'decoder.conv_out'])
        self._prepare_execution_plan(plan)

        streams = [self.stream_a, self.stream_b]
        h = latents

        for i, layer_name in enumerate(plan):
            buffer_id = i % 2
            stream = streams[buffer_id]
            
            stream.synchronize()
            self.allocator.reset('gpu_weights', buffer_id=buffer_id)
            
            # NOTE: This part is not fully implemented with the new design
            # submodule = self._materialize_module(layer_name, buffer_id, stream)
            raise NotImplementedError("VAEScheduler not yet refactored for stateless execution.")

            if i + self.prefetch_depth < len(plan):
                self.executor.submit(self._load_layer_to_pool, plan[i + self.prefetch_depth])

            with torch.cuda.stream(stream):
                # h = submodule(h)
                pass

        torch.cuda.synchronize()
        return h

class T5Scheduler(BaseScheduler):
    """
    Scheduler for the T5 text encoder with CUDA Graph optimization.
    """
    def run_encoder_inference(self, input_ids):
        plan = ['shared'] + [f'encoder.block.{i}' for i in range(len(self.blueprint.encoder.block))] + ['encoder.final_layer_norm']
        self._prepare_execution_plan(plan)

        streams = [self.stream_a, self.stream_b]
        hidden_states = input_ids.to(self.device)
        position_bias = None
        extended_attention_mask = None

        for i, layer_name in enumerate(plan):
            buffer_id = i % 2
            stream = streams[buffer_id]
            
            other_stream = streams[(i + 1) % 2]
            other_stream.synchronize()
            
            # Wait for the current layer's weights to be loaded to the GPU
            layer_name, gpu_handles = gpu_future.result()

            # Prefetch the next layer to CPU, and then schedule its copy to the other GPU buffer
            if i + 1 < len(plan):
                cpu_future = self.prefetch_queue.get()
                gpu_future = self.executor.submit(self._load_layer_to_gpu, cpu_future.result(), (i + 1) % 2, streams[(i + 1) % 2])

            self.allocator.reset('gpu_weights', buffer_id=buffer_id)
            
            if i + self.prefetch_depth < len(plan):
                future = self.executor.submit(self._load_layer_to_pool, plan[i + self.prefetch_depth])
                self.prefetch_queue.put(future)

            with self.allocator.scope() as workspace, torch.cuda.stream(stream):
                submodule = self.blueprint.get_submodule(layer_name)

                if position_bias is None and 'block' in layer_name:
                    # This calculation is still needed to generate the bias for the first block
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
                    # For the block, we use our custom kernel
                    output_tensor = workspace.allocate(
                        hidden_states.nbytes, 'workspace'
                    ).view(hidden_states.dtype).reshape(hidden_states.shape)
                    
                    # NOTE: The C++ signature for t5_block_forward must be updated to accept
                    # all the tensors below. This call serves as the specification.
                    custom_t5_cpp.t5_block_forward(
                        # Input Tensors
                        hidden_states,
                        position_bias,
                        extended_attention_mask,
                        
                        # Layer 0 (Self-Attention) Weights
                        gpu_handles['layer.0.layer_norm.weight'],
                        gpu_handles['layer.0.SelfAttention.q.weight'],
                        gpu_handles['layer.0.SelfAttention.k.weight'],
                        gpu_handles['layer.0.SelfAttention.v.weight'],
                        gpu_handles['layer.0.SelfAttention.o.weight'],

                        # Layer 1 (Feed-Forward) Weights
                        gpu_handles['layer.1.layer_norm.weight'],
                        gpu_handles['layer.1.DenseReluDense.wi.weight'],
                        gpu_handles['layer.1.DenseReluDense.wo.weight'],

                        # Output Tensor
                        output_tensor
                    )
                    hidden_states = output_tensor
                else: # final_layer_norm
                    hidden_states = functional_call(submodule, gpu_handles, (hidden_states,))

        torch.cuda.synchronize()
        return hidden_states

class CLIPScheduler(BaseScheduler):
    """
    Scheduler for the CLIP text encoder.
    """
    def run_encoder_inference(self, input_ids):
        plan = ['text_model.embeddings'] + [f'text_model.encoder.layers.{i}' for i in range(len(self.blueprint.text_model.encoder.layers))] + ['text_model.final_layer_norm']
        self._prepare_execution_plan(plan)

        streams = [self.stream_a, self.stream_b]
        hidden_states = input_ids

        for i, layer_name in enumerate(plan):
            buffer_id = i % 2
            stream = streams[buffer_id]
            
            stream.synchronize()
            self.allocator.reset('gpu_weights', buffer_id=buffer_id)
            
            # NOTE: This part is not fully implemented with the new design
            # submodule = self._materialize_module(layer_name, buffer_id, stream)
            raise NotImplementedError("CLIPScheduler not yet refactored for stateless execution.")

            if i + self.prefetch_depth < len(plan):
                self.executor.submit(self._load_layer_to_pool, plan[i + self.prefetch_depth])

            with torch.cuda.stream(stream):
                pass

        torch.cuda.synchronize()
        return hidden_states