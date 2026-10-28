"""Simulated (fake) integer quantization of nn.Linear layers.

Weights are rounded once to a symmetric integer grid and stored back in the
layer's dtype. Activations, when quantized, are rounded per token on every
Linear input by a forward pre-hook. Biases, embeddings, convolutions, norms and
attention softmax are left unchanged.

- W8A8: 8-bit weights per output channel, 8-bit activations per token (dynamic).
- W4A16: 4-bit weights in groups of 128 input channels, activations unchanged.
"""
import torch
from torch import nn
import torch.nn.functional as F

SCHEMES = {'W8A8': dict(weight_bits=8, group=None, act_bits=8),
           'W4A16': dict(weight_bits=4, group=128, act_bits=None)}


def quantize_weight_(linear, bits, group):
    w = linear.weight.data.float()
    out_features, in_features = w.shape
    if group is None:
        blocks = w.view(out_features, 1, in_features)
    else:
        # The last group is zero-padded when in_features is not a multiple of the group size.
        blocks = F.pad(w, (0, (-in_features) % group)).view(out_features, -1, group)
    qmax = 2 ** (bits - 1) - 1
    scale = blocks.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    q = (blocks / scale).round().clamp(-qmax - 1, qmax) * scale
    linear.weight.data = q.view(out_features, -1)[:, :in_features].to(linear.weight.dtype).contiguous()


def quantize_activation(bits):
    qmax = 2 ** (bits - 1) - 1

    def hook(module, args):
        x = args[0]
        scale = x.float().abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
        return ((x.float() / scale).round().clamp(-qmax - 1, qmax) * scale).to(x.dtype), *args[1:]
    return hook


def apply(modules, scheme):
    """Quantize every nn.Linear inside `modules`; returns the number of layers changed."""
    spec = SCHEMES[scheme]
    count = 0
    for root in modules:
        for layer in root.modules():
            if isinstance(layer, nn.Linear):
                quantize_weight_(layer, spec['weight_bits'], spec['group'])
                if spec['act_bits']:
                    layer.register_forward_pre_hook(quantize_activation(spec['act_bits']))
                count += 1
    return count


def max_levels(linear, group):
    """Largest number of distinct weight values in one quantization block (for checks)."""
    w = linear.weight.data.float()
    if group is None:
        blocks = w.view(w.shape[0], 1, w.shape[1])
    else:
        blocks = F.pad(w, (0, (-w.shape[1]) % group)).view(w.shape[0], -1, group)
    rows = blocks.reshape(-1, blocks.shape[-1])[:64]
    return max(len(torch.unique(r)) for r in rows)
