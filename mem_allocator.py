# jitstandalone/mem_allocator.py

import torch

class MemoryAllocator:
    """
    A simple bump-allocator for managing pre-allocated CPU and GPU memory pools.
    It allocates memory sequentially and is reset after each layer's execution.
    """
    def __init__(self, cpu_pool_size: int, gpu_pool_size: int, device: str = "cuda"):
        self.device = device
        # Pinned memory is essential for asynchronous GPU copies
        self.cpu_pool = torch.empty(cpu_pool_size, dtype=torch.uint8, device='cpu').pin_memory()
        
        # Double buffer for GPU to overlap computation and data transfer
        self.gpu_pool_a = torch.empty(gpu_pool_size // 2, dtype=torch.uint8, device=self.device)
        self.gpu_pool_b = torch.empty(gpu_pool_size // 2, dtype=torch.uint8, device=self.device)

        self.cpu_offset = 0
        self.gpu_offsets = [0, 0]

    def allocate(self, size: int, device: str, buffer_id: int = 0):
        """
        Allocates a block of memory of a given size from the specified pool.
        For CUDA devices, buffer_id (0 or 1) selects the double buffer.
        """
        size = int(size)
        if device == 'cpu':
            if self.cpu_offset + size > self.cpu_pool.numel():
                raise MemoryError(f"CPU memory pool out of memory. Requested {size}, available {self.cpu_pool.numel() - self.cpu_offset}")
            tensor_slice = self.cpu_pool.narrow(0, self.cpu_offset, size)
            self.cpu_offset += size
            return tensor_slice
        
        elif device == 'cuda':
            pool = self.gpu_pool_a if buffer_id == 0 else self.gpu_pool_b
            offset = self.gpu_offsets[buffer_id]
            
            if offset + size > pool.numel():
                raise MemoryError(f"GPU memory pool {buffer_id} out of memory. Requested {size}, available {pool.numel() - offset}")
            
            tensor_slice = pool.narrow(0, offset, size)
            self.gpu_offsets[buffer_id] += size
            return tensor_slice
        else:
            raise ValueError(f"Unknown device for allocation: {device}")

    def reset(self, device: str = None, buffer_id: int = None):
        """
        Resets the offset for a specific pool or all pools, making them fully available.
        For CUDA, can reset a specific buffer_id or both.
        """
        if device is None or device == 'cpu':
            self.cpu_offset = 0
        if device is None or device == 'cuda':
            if buffer_id is None:
                self.gpu_offsets = [0, 0]
            else:
                self.gpu_offsets[buffer_id] = 0