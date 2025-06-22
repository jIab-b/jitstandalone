import torch
import logging
from . import model_management
from . import rmsnorm
import contextlib

def cast_to_input(weight, input, non_blocking=False, copy=True):
    return model_management.cast_to(weight, input.dtype, input.device, non_blocking=non_blocking, copy=copy)

def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
    if input is not None:
        if dtype is None:
            dtype = input.dtype
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    offload_stream = model_management.get_offload_stream(device)
    if offload_stream is not None:
        wf_context = offload_stream
    else:
        wf_context = contextlib.nullcontext()

    bias = None
    non_blocking = model_management.device_supports_non_blocking(device)
    if s.bias is not None:
        has_function = len(s.bias_function) > 0
        bias = model_management.cast_to(s.bias, bias_dtype, device, non_blocking=non_blocking, copy=has_function, stream=offload_stream)

        if has_function:
            with wf_context:
                for f in s.bias_function:
                    bias = f(bias)

    has_function = len(s.weight_function) > 0
    weight = model_management.cast_to(s.weight, dtype, device, non_blocking=non_blocking, copy=has_function, stream=offload_stream)
    if has_function:
        with wf_context:
            for f in s.weight_function:
                weight = f(weight)

    model_management.sync_stream(device, offload_stream)
    return weight, bias

class CastWeightBiasOp:
    comfy_cast_weights = False
    weight_function = []
    bias_function = []

class disable_weight_init:
    class Linear(torch.nn.Linear, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv3d(torch.nn.Conv3d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            if self.weight is not None:
                weight, bias = cast_bias_weight(self, input)
            else:
                weight = None
                bias = None
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class RMSNorm(rmsnorm.RMSNorm, CastWeightBiasOp):
        def reset_parameters(self):
            self.bias = None
            return None

        def forward_comfy_cast_weights(self, input):
            if self.weight is not None:
                weight, bias = cast_bias_weight(self, input)
            else:
                weight = None
            return rmsnorm.rms_norm(input, weight, self.eps)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)