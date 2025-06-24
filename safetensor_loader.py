# jitstandalone/safetensor_loader.py

import os
import json
import struct
import mmap
import torch
import numpy as np

SAFETENSORS_DTYPE_MAP = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}

NUMPY_DTYPE_MAP = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16,
    "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_,
}

class SafetensorLoader:
    """
    On-demand safetensor loader using memory-mapping for zero-copy reads.
    Can handle single-file or multi-file (sharded) models.
    """
    def __init__(self, path: str, model_config: dict = None, quant_config: str = None):
        self.path = path
        self.model_config = model_config or {}
        self.weight_map = self.model_config.get('weight_map')
        self.quant_config = quant_config
        if self.quant_config:
            try:
                import bitsandbytes.functional as F
                self.F = F
            except ImportError:
                raise ImportError("bitsandbytes is required for quantization. Please install it.")

        self.file_handles = {}
        self.mmaps = {}
        self.headers = {}
        self.data_start_offsets = {}

        if self.weight_map:
            # Multi-file (sharded) model. `path` is a directory.
            unique_files = set(self.weight_map.values())
            for filename in unique_files:
                shard_path = os.path.join(path, filename)
                file = open(shard_path, 'rb')
                self.file_handles[filename] = file
                
                header_len_bytes = file.read(8)
                header_len = struct.unpack('<Q', header_len_bytes)[0]
                header_json_bytes = file.read(header_len)
                header = json.loads(header_json_bytes.decode('utf-8'))
                
                self.headers[filename] = header
                self.data_start_offsets[filename] = 8 + header_len
                self.mmaps[filename] = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            # Single-file model. `path` is a file path.
            file = open(path, 'rb')
            self.file_handles['__single__'] = file
            
            header_len_bytes = file.read(8)
            header_len = struct.unpack('<Q', header_len_bytes)[0]
            header_json_bytes = file.read(header_len)
            header = json.loads(header_json_bytes.decode('utf-8'))
            
            self.headers['__single__'] = header
            self.data_start_offsets['__single__'] = 8 + header_len
            self.mmaps['__single__'] = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)

    def __del__(self):
        for mmap_obj in self.mmaps.values():
            mmap_obj.close()
        for file_handle in self.file_handles.values():
            file_handle.close()

    def get_tensor_info(self, name: str) -> dict:
        """
        Returns the metadata for a single tensor.
        """
        if self.weight_map:
            if name not in self.weight_map:
                raise KeyError(f"Tensor '{name}' not found in the weight map.")
            filename = self.weight_map[name]
            header = self.headers[filename]
        else:
            filename = '__single__'
            header = self.headers[filename]

        if name not in header:
            raise KeyError(f"Tensor '{name}' not found in header of '{filename}'.")
        
        return header[name]

    def load_tensor_into(self, name: str, buffer: torch.Tensor):
        """
        Loads a tensor from disk directly into a pre-allocated buffer using a
        zero-copy view into the memory-mapped file.
        """
        info = self.get_tensor_info(name)
        filename = self.weight_map.get(name, '__single__')
        mmap_obj = self.mmaps[filename]
        data_start_offset = self.data_start_offsets[filename]
        dtype = SAFETENSORS_DTYPE_MAP.get(info['dtype'])
        if dtype is None:
            raise TypeError(f"Unsupported dtype '{info['dtype']}' for tensor '{name}'")

        shape = info['shape']
        offsets = info['data_offsets']
        data_len = offsets[1] - offsets[0]
        
        data_offset = data_start_offset + offsets[0]

        # Create a NumPy array that is a zero-copy view into the mmap object.
        # This avoids creating intermediate copies in memory.
        if info['dtype'] == 'BF16':
            # NumPy doesn't have bfloat16, so we view the raw uint16 bytes.
            np_dtype = np.uint16
        else:
            np_dtype = NUMPY_DTYPE_MAP[info['dtype']]

        np_view = np.frombuffer(mmap_obj, dtype=np_dtype, count=data_len // np.dtype(np_dtype).itemsize, offset=data_offset).reshape(shape)
        
        # Create a torch tensor that views the numpy array, still no copy.
        tensor_view = torch.from_numpy(np_view)

        if info['dtype'] == 'BF16':
            tensor_view = tensor_view.view(torch.bfloat16)

        # The only copy happens here, directly into the target buffer.
        buffer.copy_(tensor_view)

    def debug_print_all_tensor_info(self):
        """
        Prints a detailed report of every tensor the loader has metadata for,
        including its file location and byte offsets.
        """
        print("\n--- SafetensorLoader Debug Report ---")
        for filename, header in self.headers.items():
            print(f"\n[File: {filename}]")
            data_start_offset = self.data_start_offsets[filename]
            
            # Sort tensors by their starting offset for readability
            sorted_tensors = sorted(header.items(), key=lambda item: item[1].get('data_offsets', [0])[0])
            
            for name, info in sorted_tensors:
                if name == '__metadata__':
                    continue
                
                offsets = info.get('data_offsets')
                if not offsets:
                    print(f"  - {name}: (Metadata only, no data offsets)")
                    continue
                    
                start_offset, end_offset = offsets
                absolute_start = data_start_offset + start_offset
                size_mb = (end_offset - start_offset) / 1e6
                
                print(f"  - {name}")
                print(f"    - Shape: {info['shape']}")
                print(f"    - DType: {info['dtype']}")
                print(f"    - Size: {size_mb:.4f} MB")
                print(f"    - Relative Offsets: [{start_offset}, {end_offset}]")
                print(f"    - Absolute Offsets: [{absolute_start}, {absolute_start + (end_offset - start_offset)}]")

        print("\n--- End of Report ---\n")

def extract_safetensor_metadata(path: str) -> dict:
    """
    Reads only the header of a safetensor file to extract metadata.
    """
    with open(path, 'rb') as f:
        header_len_bytes = f.read(8)
        header_len = struct.unpack('<Q', header_len_bytes)[0]
        header_json_bytes = f.read(header_len)
        header = json.loads(header_json_bytes.decode('utf-8'))
    return header

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Extract safetensor metadata to a file.")
    parser.add_argument("input_file", type=str, help="Path to the .safetensors file.")
    parser.add_argument("output_file", type=str, help="Path to save the metadata JSON file.")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found at {args.input_file}", file=sys.stderr)
        sys.exit(1)

    metadata = extract_safetensor_metadata(args.input_file)
    
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Metadata from {args.input_file} saved to {args.output_file}")