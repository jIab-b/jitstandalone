# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
import logging

from . import model_management
from . import ops
ops = ops.disable_weight_init

if model_management.xformers_enabled_vae():
    try:
        import xformers
        import xformers.ops
    except ImportError:
        pass


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return ops.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def interpolate_up(x, scale_factor):
    try:
        return torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode="nearest")
    except: #operation not implemented for bf16
        orig_shape = list(x.shape)
        out_shape = orig_shape[:2]
        for i in range(len(orig_shape) - 2):
            out_shape.append(round(orig_shape[i + 2] * scale_factor[i]))
        out = torch.empty(out_shape, dtype=x.dtype, layout=x.layout, device=x.device)
        split = 8
        l = out.shape[1] // split
        for i in range(0, out.shape[1], l):
            out[:,i:i+l] = torch.nn.functional.interpolate(x[:,i:i+l].to(torch.float32), scale_factor=scale_factor, mode="nearest").to(x.dtype)
        return out

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, conv_op=ops.Conv2d, scale_factor=2.0):
        super().__init__()
        self.with_conv = with_conv
        self.scale_factor = scale_factor

        if self.with_conv:
            self.conv = conv_op(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        scale_factor = self.scale_factor
        if isinstance(scale_factor, (int, float)):
            scale_factor = (scale_factor,) * (x.ndim - 2)
        x = interpolate_up(x, scale_factor)
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, stride=2, conv_op=ops.Conv2d):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = conv_op(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, conv_op=ops.Conv2d):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.swish = torch.nn.SiLU(inplace=True)
        self.norm1 = Normalize(in_channels)
        self.conv1 = conv_op(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = ops.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout, inplace=True)
        self.conv2 = conv_op(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv_op(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = conv_op(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.swish(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.swish(temb))[:,:,None,None]

        h = self.norm2(h)
        h = self.swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

def slice_attention(q, k, v):
    r1 = torch.zeros_like(k, device=q.device)
    scale = (int(q.shape[-1])**(-0.5))

    mem_free_total = model_management.get_free_memory(q.device)

    tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
    modifier = 3 if q.element_size() == 2 else 2.5
    mem_required = tensor_size * modifier
    steps = 1

    if mem_required > mem_free_total:
        steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))

    while True:
        try:
            slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
            for i in range(0, q.shape[1], slice_size):
                end = i + slice_size
                s1 = torch.bmm(q[:, i:end], k) * scale

                s2 = torch.nn.functional.softmax(s1, dim=2).permute(0,2,1)
                del s1

                r1[:, :, i:end] = torch.bmm(v, s2)
                del s2
            break
        except model_management.OOM_EXCEPTION as e:
            model_management.soft_empty_cache(True)
            steps *= 2
            if steps > 128:
                raise e
            logging.warning("out of memory error, increasing steps and trying again {}".format(steps))

    return r1

def normal_attention(q, k, v):
    # compute attention
    orig_shape = q.shape
    b = orig_shape[0]
    c = orig_shape[1]

    q = q.reshape(b, c, -1)
    q = q.permute(0, 2, 1)   # b,hw,c
    k = k.reshape(b, c, -1) # b,c,hw
    v = v.reshape(b, c, -1)

    r1 = slice_attention(q, k, v)
    h_ = r1.reshape(orig_shape)
    del r1
    return h_

def xformers_attention(q, k, v):
    # compute attention
    orig_shape = q.shape
    B = orig_shape[0]
    C = orig_shape[1]
    q, k, v = map(
        lambda t: t.view(B, C, -1).transpose(1, 2).contiguous(),
        (q, k, v),
    )

    try:
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
        out = out.transpose(1, 2).reshape(orig_shape)
    except NotImplementedError:
        out = slice_attention(q.view(B, -1, C), k.view(B, -1, C).transpose(1, 2), v.view(B, -1, C).transpose(1, 2)).reshape(orig_shape)
    return out

def pytorch_attention(q, k, v):
    # compute attention
    orig_shape = q.shape
    B = orig_shape[0]
    C = orig_shape[1]
    q, k, v = map(
        lambda t: t.view(B, 1, C, -1).transpose(2, 3).contiguous(),
        (q, k, v),
    )

    try:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(2, 3).reshape(orig_shape)
    except model_management.OOM_EXCEPTION:
        logging.warning("scaled_dot_product_attention OOMed: switched to slice attention")
        out = slice_attention(q.view(B, -1, C), k.view(B, -1, C).transpose(1, 2), v.view(B, -1, C).transpose(1, 2)).reshape(orig_shape)
    return out


def vae_attention():
    if model_management.xformers_enabled_vae():
        logging.info("Using xformers attention in VAE")
        return xformers_attention
    elif model_management.pytorch_attention_enabled_vae():
        logging.info("Using pytorch attention in VAE")
        return pytorch_attention
    else:
        logging.info("Using split attention in VAE")
        return normal_attention

class AttnBlock(nn.Module):
    def __init__(self, in_channels, conv_op=ops.Conv2d):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = conv_op(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = conv_op(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = conv_op(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = conv_op(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

        self.optimized_attention = vae_attention()

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        h_ = self.optimized_attention(q, k, v)

        h_ = self.proj_out(h_)

        return x+h_