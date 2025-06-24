import torch

class BudgetedAllocator:
    """
    A simple, budgeted memory allocator that tracks memory usage against hard
    limits for both GPU and CPU. It relies on an external scheduler to manage
    the lifecycle of tensors and does not manage memory pools itself.
    """
    def __init__(self, gpu_limit_bytes: int, cpu_limit_bytes: int, device: str = "cuda"):
        self.device = device
        self.gpu_limit = gpu_limit_bytes
        self.cpu_limit = cpu_limit_bytes
        self.gpu_bytes_used = 0
        self.cpu_bytes_used = 0

        # Optional: Set PyTorch's memory fraction as a first line of defense
        # Note: This is a soft limit and doesn't account for fragmentation.
        try:
            device_props = torch.cuda.get_device_properties(self.device)
            fraction = self.gpu_limit / device_props.total_memory
            if fraction > 1.0:
                print(f"Warning: Requested GPU budget ({self.gpu_limit/1e9:.2f}GB) exceeds device capacity ({device_props.total_memory/1e9:.2f}GB).")
            torch.cuda.set_per_process_memory_fraction(min(fraction, 1.0), self.device)
        except Exception as e:
            print(f"Could not set per-process memory fraction: {e}")

    def malloc_gpu(self, nbytes: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        """Allocates a tensor on the GPU after checking the budget."""
        nbytes = int(nbytes)
        if self.gpu_bytes_used + nbytes > self.gpu_limit:
            raise MemoryError(
                f"GPU OOM: Cannot allocate {nbytes/1e6:.2f}MB. "
                f"Used: {self.gpu_bytes_used/1e6:.2f}MB, Limit: {self.gpu_limit/1e6:.2f}MB"
            )
        self.gpu_bytes_used += nbytes
        # Using a raw byte tensor is common for buffer management
        return torch.empty(nbytes // dtype.itemsize, dtype=dtype, device=self.device)

    def free_gpu(self, tensor: torch.Tensor):
        """Frees a GPU tensor and updates the budget."""
        self.gpu_bytes_used -= tensor.nbytes
        del tensor

    def malloc_cpu(self, nbytes: int, dtype: torch.dtype = torch.uint8, pinned: bool = True) -> torch.Tensor:
        """Allocates a tensor on the CPU (pinned) after checking the budget."""
        nbytes = int(nbytes)
        if self.cpu_bytes_used + nbytes > self.cpu_limit:
            raise MemoryError(
                f"CPU OOM: Cannot allocate {nbytes/1e6:.2f}MB. "
                f"Used: {self.cpu_bytes_used/1e6:.2f}MB, Limit: {self.cpu_limit/1e6:.2f}MB"
            )
        self.cpu_bytes_used += nbytes
        tensor = torch.empty(nbytes // dtype.itemsize, dtype=dtype, device='cpu')
        return tensor.pin_memory() if pinned else tensor

    def free_cpu(self, tensor: torch.Tensor):
        """Frees a CPU tensor and updates the budget."""
        self.cpu_bytes_used -= tensor.nbytes
        del tensor

    def get_gpu_usage_str(self) -> str:
        return f"{self.gpu_bytes_used/1e6:.2f}MB / {self.gpu_limit/1e6:.2f}MB"

    def get_cpu_usage_str(self) -> str:
        return f"{self.cpu_bytes_used/1e6:.2f}MB / {self.cpu_limit/1e6:.2f}MB"