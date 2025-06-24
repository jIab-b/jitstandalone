# jitstandalone/mem_allocator.py

import torch
from contextlib import contextmanager

class MemoryAllocator:
    """
    A managed memory allocator that partitions a total VRAM limit between
    double-buffered weight pools and a reusable activation workspace.
    """
    def __init__(self, total_vram_limit: int, max_module_size: int, cpu_pool_size: int, device: str = "cuda"):
        self.device = device

        # 1. Calculate the size needed for the double-buffered weights.
        # This holds two copies of the largest module.
        weights_pool_size = 2 * int(max_module_size)
        
        if weights_pool_size >= total_vram_limit:
            raise MemoryError(
                f"The provided VRAM limit ({total_vram_limit / 1e9:.2f} GB) is not even large enough "
                f"to hold two copies of the largest module ({weights_pool_size / 1e9:.2f} GB)."
            )

        # 2. The rest of the VRAM is dedicated to the activation workspace.
        workspace_pool_size = total_vram_limit - weights_pool_size

        print(f"VRAM Budget: {total_vram_limit / 1e9:.2f} GB")
        print(f" -> Weights (2x {max_module_size / 1e9:.2f} GB): {weights_pool_size / 1e9:.2f} GB")
        print(f" -> Workspace: {workspace_pool_size / 1e9:.2f} GB")

        # 3. Allocate all the memory pools.
        self.cpu_pool = torch.empty(int(cpu_pool_size), dtype=torch.uint8, device='cpu').pin_memory()
        
        self.gpu_weights_pool_a = torch.empty(int(max_module_size), dtype=torch.uint8, device=self.device)
        self.gpu_weights_pool_b = torch.empty(int(max_module_size), dtype=torch.uint8, device=self.device)
        self.gpu_workspace_pool = torch.empty(int(workspace_pool_size), dtype=torch.uint8, device=self.device)

        self.cpu_offset = 0
        self.gpu_weights_offsets = [0, 0]
        self.gpu_workspace_offset = 0

    def _allocate(self, size: int, pool: torch.Tensor, offset: int):
        # Align to a common boundary like 128 bytes for performance
        aligned_offset = (offset + 127) & -128
        if aligned_offset + size > pool.numel():
             raise MemoryError(f"Pool out of memory after alignment. Requested {size}, available {pool.numel() - offset}")
        
        tensor_slice = pool.narrow(0, aligned_offset, size)
        return tensor_slice, aligned_offset + size

    def allocate(self, size: int, pool_name: str, buffer_id: int = 0):
        """Allocates memory from a named pool."""
        size = int(size)
        if pool_name == 'cpu':
            tensor_slice, self.cpu_offset = self._allocate(size, self.cpu_pool, self.cpu_offset)
            return tensor_slice
        elif pool_name == 'gpu_weights':
            pool = self.gpu_weights_pool_a if buffer_id == 0 else self.gpu_weights_pool_b
            offset_ref = self.gpu_weights_offsets[buffer_id]
            tensor_slice, new_offset = self._allocate(size, pool, offset_ref)
            self.gpu_weights_offsets[buffer_id] = new_offset
            return tensor_slice
        elif pool_name == 'workspace':
            tensor_slice, self.gpu_workspace_offset = self._allocate(size, self.gpu_workspace_pool, self.gpu_workspace_offset)
            return tensor_slice
        else:
            raise ValueError(f"Unknown pool name: {pool_name}")

    def reset(self, pool_name: str, buffer_id: int = None):
        """Resets the offset for a specific pool."""
        if pool_name == 'cpu':
            self.cpu_offset = 0
        elif pool_name == 'gpu_weights':
            if buffer_id is None: self.gpu_weights_offsets = [0, 0]
            else: self.gpu_weights_offsets[buffer_id] = 0
        elif pool_name == 'workspace':
            self.gpu_workspace_offset = 0
        else:
            raise ValueError(f"Unknown pool name: {pool_name}")

    @contextmanager
    def scope(self):
        """A context manager to handle workspace allocation for a single forward pass."""
        self.reset('workspace')
        try:
            yield self
        finally:
            self.reset('workspace')