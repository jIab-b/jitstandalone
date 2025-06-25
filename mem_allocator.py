import torch
import threading

class CUDAMemoryAllocator:
    """
    A custom memory allocator that manages GPU memory under a strict budget
    and uses a condition variable to block when memory is unavailable.
    """
    def __init__(self, limit_bytes: int, device: torch.device):
        self.device = device
        self.limit = int(limit_bytes)
        self.used = 0
        self.peak = 0
        self.condition = threading.Condition()

    def malloc(self, size: int, stream: torch.cuda.Stream = None):
        """
        The allocation function that PyTorch will call. It blocks if memory is not available.
        """
        with self.condition:
            while not self.can_allocate(size):
                print(f"Allocator: Waiting to allocate {size/1e6:.2f}MB...")
                self.condition.wait()
            
            stream_ptr = stream.ptr if stream else 0
            address = torch.cuda.memory._get_cuda_memory_allocator().malloc(size, stream_ptr)
            
            self.used += size
            if self.used > self.peak:
                self.peak = self.used
                
            return address

    def free(self, address: int):
        """
        The deallocation function. It notifies waiting threads after freeing memory.
        """
        with self.condition:
            try:
                block_size = torch.cuda.memory._get_cuda_memory_allocator().get_allocation_size(address)
                self.used -= block_size
            except RuntimeError:
                pass
            
            torch.cuda.memory._get_cuda_memory_allocator().free(address)
            self.condition.notify_all()

    def can_allocate(self, size: int) -> bool:
        """Checks if a block of a given size can be allocated."""
        return self.used + size <= self.limit

    def allocate_buffer(self, size: int) -> torch.Tensor:
        """
        Explicitly allocates a tensor buffer, blocking if necessary.
        """
        with self.condition:
            while not self.can_allocate(size):
                print(f"Allocator: Waiting to allocate buffer of {size/1e6:.2f}MB...")
                self.condition.wait()

            buffer = torch.empty(int(size), dtype=torch.uint8, device=self.device)
            
            self.used += buffer.nbytes
            if self.used > self.peak:
                self.peak = self.used
                
            return buffer

    def get_usage_str(self) -> str:
        with self.condition:
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