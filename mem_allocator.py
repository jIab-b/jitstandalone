import torch

class CUDAMemoryAllocator:
    """
    A custom memory allocator that can be registered with PyTorch to manage
    all GPU memory allocations under a strict budget.

    Usage:
      alloc = CUDAMemoryAllocator(limit_bytes)
      torch.cuda.memory.change_allocator(alloc.malloc, alloc.free)
    """
    def __init__(self, limit_bytes: int, device: torch.device):
        self.device = device
        self.limit = int(limit_bytes)
        self.used = 0
        self.peak = 0

    def malloc(self, size: int, stream: torch.cuda.Stream):
        """
        The allocation function that PyTorch will call.
        It must return an integer representing the memory address.
        """
        if self.used + size > self.limit:
            raise MemoryError(
                f"Custom Allocator OOM: Cannot allocate {size/1e6:.2f}MB. "
                f"Used: {self.used/1e6:.2f}MB, Peak: {self.peak/1e6:.2f}MB, "
                f"Limit: {self.limit/1e6:.2f}MB"
            )
        
        # PyTorch's underlying allocator is still used to get the actual memory,
        # but it happens under our budget control.
        address = torch.cuda.memory._get_cuda_memory_allocator().malloc(size, stream.ptr)
        
        self.used += size
        if self.used > self.peak:
            self.peak = self.used
            
        return address

    def free(self, address: int):
        """
        The deallocation function that PyTorch will call.
        It must take an integer memory address.
        """
        try:
            # We need to know the size of the block being freed.
            # PyTorch's allocator backend can provide this.
            block_size = torch.cuda.memory._get_cuda_memory_allocator().get_allocation_size(address)
            self.used -= block_size
        except RuntimeError:
            # This can happen if the block is already freed or invalid.
            # It's safer to ignore than to crash.
            pass
        
        torch.cuda.memory._get_cuda_memory_allocator().free(address)

    def get_usage_str(self) -> str:
        return f"{self.used/1e6:.2f}MB / {self.limit/1e6:.2f}MB"

    def get_peak_usage_str(self) -> str:
        return f"{self.peak/1e6:.2f}MB"

# Note: A CPU budget allocator is simpler as it doesn't need to be hooked.
# We can use a simple wrapper class if needed, but direct tracking in the
# scheduler is often sufficient.
class CPUAllocator:
    def __init__(self, limit_bytes: int):
        self.limit = int(limit_bytes)
        self.used = 0

    def malloc(self, nbytes: int, pinned: bool = True):
        if self.used + nbytes > self.limit:
            raise MemoryError(f"CPU Allocator OOM: Cannot allocate {nbytes/1e6:.2f}MB.")
        self.used += nbytes
        tensor = torch.empty(int(nbytes), dtype=torch.uint8, device='cpu')
        return tensor.pin_memory() if pinned else tensor

    def free(self, tensor: torch.Tensor):
        self.used -= tensor.nbytes
        del tensor