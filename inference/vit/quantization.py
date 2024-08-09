import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention
from functools import partial
import torch.nn.functional as F

def percentile_quantize_fn(inp, percentile, num_dist_vals):
    original_type = inp.dtype
    if percentile == 1:
        alpha = inp.abs().max()
    else:
        alpha = torch.topk(inp.abs().reshape(-1), k = int(inp.numel() * (1 - percentile)), dim = -1).values.min()
    if alpha.item() < 1e-20:
        alpha = 1e-20
    scale = (0.5 * num_dist_vals) / alpha
    inp = (torch.round(inp * scale).to(original_type) / scale).detach()
    return inp

def percentile_clip_fn(inp, percentile):
    original_type = inp.dtype
    assert percentile < 1
    alpha = torch.topk(inp.abs().reshape(-1), k = int(inp.numel() * (1 - percentile)), dim = -1).values.min()
    if alpha.item() < 1e-20:
        alpha = 1e-20
    inp = torch.clamp(inp, min = - alpha, max = alpha)
    return inp

def quantize_fn(inp, config):
    if "clip" in config and config["clip"]:
        return percentile_clip_fn(inp, config["percentile"])
    else:
        return percentile_quantize_fn(inp, config["percentile"], config["num_dist_vals"])
    
def quantize_linear_params(module, config):
    module.weight.data = quantize_fn(module.weight.data, config = config["weight"])

def quantize_linear(x, module, config):
    x = quantize_fn(x, config = config["input"])
    y = F.linear(x, module.weight, module.bias)
    return y

def quantize_matmul(A, B, config):
    A = quantize_fn(A, config)
    B = quantize_fn(B, config)
    C = torch.matmul(A, B)
    return C

def quantize_attention_forward(x, module, config):
    B, N, C = x.shape
    qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = module.q_norm(q), module.k_norm(k)

    q = q * module.scale
    attn = quantize_matmul(q, k.transpose(-2, -1), config["attention"])
    attn = attn.softmax(dim=-1)
    attn = module.attn_drop(attn)
    x = quantize_matmul(attn, v, config["attention"])

    x = x.transpose(1, 2).reshape(B, N, C)
    x = module.proj(x)
    x = module.proj_drop(x)
    return x

def quantize(model, config):
    for name, module in model.named_modules():
        if isinstance(module, Attention) and "attention" in config:
            print(name, module)
            module.forward = partial(quantize_attention_forward, module = module, config = config)

        if isinstance(module, nn.Linear):
            if "weight" in config:
                print(name, module, "quantize_linear_params")
                quantize_linear_params(module, config = config)

            if "input" in config:
                print(name, module, "quantize_linear")
                module.forward = partial(quantize_linear, module = module, config = config)

