# jitstandalone/scheduler.py

import torch
from torch import nn
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

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

        for name, param in submodule.named_parameters(recurse=False):
            full_name = f"{layer_name}.{name}"
            info = self.loader.get_tensor_info(full_name)
            dtype = SAFETENSORS_DTYPE_MAP[info['dtype']]
            size = torch.tensor([], dtype=dtype).element_size() * torch.prod(torch.tensor(info['shape']))

            cpu_buffer = self.allocator.allocate(size, 'cpu').view(dtype).reshape(param.shape)
            self.loader.load_tensor_into(full_name, cpu_buffer)
            handles[name] = cpu_buffer

        for name, buffer in submodule.named_buffers(recurse=False):
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
        return handles

    def _materialize_module(self, layer_name: str, buffer_id: int, stream: torch.cuda.Stream):
        """
        Manually creates a module on the GPU by allocating memory for its parameters
        and buffers from our custom allocator and copying weights asynchronously.
        """
        blueprint_submodule = self.blueprint.get_submodule(layer_name)
        cpu_handles = self.prefetch_queue.get().result()

        with torch.cuda.stream(stream):
            for name, meta_param in blueprint_submodule.named_parameters(recurse=False):
                cpu_tensor = cpu_handles[name]
                gpu_tensor = self.allocator.allocate(cpu_tensor.nbytes, 'cuda', buffer_id=buffer_id).view(cpu_tensor.dtype).reshape(cpu_tensor.shape)
                gpu_tensor.copy_(cpu_tensor, non_blocking=True)
                
                delattr(blueprint_submodule, name)
                setattr(blueprint_submodule, name, torch.nn.Parameter(gpu_tensor, requires_grad=False))

            for name, meta_buffer in blueprint_submodule.named_buffers(recurse=False):
                if name not in cpu_handles: continue
                cpu_tensor = cpu_handles[name]
                gpu_buffer = self.allocator.allocate(cpu_tensor.nbytes, 'cuda', buffer_id=buffer_id).view(cpu_tensor.dtype).reshape(cpu_tensor.shape)
                gpu_buffer.copy_(cpu_tensor, non_blocking=True)
                
                delattr(blueprint_submodule, name)
                setattr(blueprint_submodule, name, gpu_buffer)

        return blueprint_submodule


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
        
        for i, layer_name in enumerate(plan):
            buffer_id = i % 2
            stream = streams[buffer_id]
            
            stream.synchronize()
            self.allocator.reset('gpu', buffer_id=buffer_id)
            
            submodule = self._materialize_module(layer_name, buffer_id, stream)

            if i + self.prefetch_depth < len(plan):
                future = self.executor.submit(self._load_layer_to_pool, plan[i + self.prefetch_depth])
                self.prefetch_queue.put(future)

            with torch.cuda.stream(stream):
                if layer_name == 'img_in':
                    img = submodule(img)
                elif layer_name == 'time_in':
                    vec = submodule(timestep_embedding(timestep, 256).to(img.dtype))
                elif layer_name == 'guidance_in':
                    vec = vec + submodule(timestep_embedding(guidance, 256).to(img.dtype))
                elif layer_name == 'vector_in':
                    vec = vec + submodule(y[:,:self.blueprint.params.vec_in_dim])
                elif layer_name == 'txt_in':
                    txt = submodule(txt)
                elif layer_name == 'pe_embedder':
                    ids = torch.cat((txt_ids, img_ids), dim=1)
                    pe = submodule(ids)
                elif 'double_blocks' in layer_name:
                    img, txt = submodule(img=img, txt=txt, vec=vec, pe=pe)
                elif 'single_blocks' in layer_name:
                    if i > 0 and 'double_blocks' in plan[i-1]: # First single block
                        img = torch.cat((txt, img), 1)
                    img = submodule(img, vec=vec, pe=pe)
                elif layer_name == 'final_layer':
                    img = img[:, txt.shape[1] :, ...]
                    img = submodule(img, vec)

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
            self.allocator.reset('gpu', buffer_id=buffer_id)
            
            submodule = self._materialize_module(layer_name, buffer_id, stream)

            if i + self.prefetch_depth < len(plan):
                self.executor.submit(self._load_layer_to_pool, plan[i + self.prefetch_depth])

            with torch.cuda.stream(stream):
                h = submodule(h)

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
        hidden_states = input_ids
        position_bias = None

        # --- Pre-computation of Position Bias ---
        first_block_name = 'encoder.block.0'
        self.allocator.reset('gpu', buffer_id=0)
        first_block_module = self._materialize_module(first_block_name, 0, self.stream_a)
        self.stream_a.synchronize() # Wait for copy to finish

        with torch.cuda.stream(self.stream_a):
            attention_layer = first_block_module.layer[0].SelfAttention
            
            mask_shape = hidden_states.shape[:2]
            mask_size = mask_shape[0] * mask_shape[1]
            attention_mask_buffer = self.allocator.allocate(mask_size, 'cuda', buffer_id=0)
            attention_mask = attention_mask_buffer.view(torch.uint8).reshape(mask_shape)
            attention_mask.fill_(1)
            
            extended_attention_mask = attention_mask[:, None, None, :]
            model_dtype = first_block_module.layer[0].SelfAttention.q.weight.dtype
            
            extended_attention_mask = extended_attention_mask.to(dtype=model_dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(model_dtype).min
            
            position_bias = attention_layer.compute_bias(hidden_states.shape[1], hidden_states.shape[1])

        self.prefetch_queue.get() # Consume the future for the first block

        for i, layer_name in enumerate(plan):
            buffer_id = i % 2
            stream = streams[buffer_id]
            
            stream.synchronize()
            self.allocator.reset('gpu', buffer_id=buffer_id)
            
            # --- Graph Replay or Capture ---
            graph_key = (layer_name, hidden_states.shape, hidden_states.dtype)
            if graph_key in self.cuda_graphs:
                self.cuda_graphs[graph_key].replay()
            else:
                # --- Capture Phase ---
                submodule = self._materialize_module(layer_name, buffer_id, stream)
                
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    if layer_name == 'shared':
                        output_states = submodule(hidden_states)
                    elif 'block' in layer_name:
                        output_states = submodule(
                            hidden_states,
                            attention_mask=extended_attention_mask,
                            position_bias=position_bias,
                            use_cache=False
                        )[0]
                    else: # final_layer_norm
                        output_states = submodule(hidden_states)
                
                self.cuda_graphs[graph_key] = graph
                hidden_states = output_states

            if i + self.prefetch_depth < len(plan):
                future = self.executor.submit(self._load_layer_to_pool, plan[i + self.prefetch_depth])
                self.prefetch_queue.put(future)

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
            self.allocator.reset('gpu', buffer_id=buffer_id)
            
            submodule = self._materialize_module(layer_name, buffer_id, stream)

            if i + self.prefetch_depth < len(plan):
                self.executor.submit(self._load_layer_to_pool, plan[i + self.prefetch_depth])

            with torch.cuda.stream(stream):
                if 'layers' in layer_name:
                    hidden_states = submodule(hidden_states)[0]
                else:
                    hidden_states = submodule(hidden_states)

        torch.cuda.synchronize()
        return hidden_states