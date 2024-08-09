import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial

def percentile_quantize_fn(inp, percentile, num_dist_vals):
    original_type = inp.dtype
    inp = inp.float()
    if percentile == 1:
        alpha = inp.abs().max()
    else:
        alpha = torch.topk(inp.abs().reshape(-1), k = int(inp.numel() * (1 - percentile)), dim = -1).values.min()
        # alpha = torch.quantile(inp.abs().reshape(-1).float(), percentile)
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
    # assert module.bias is None

def quantize_embed_params(module, config):
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

def quantize(model, config):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(name, module)
            
            if "weight" in config:
                print("quantize_linear_params")
                quantize_linear_params(module, config = config)

            if "input" in config:
                print("quantize_linear")
                module.forward = partial(quantize_linear, module = module, config = config)

        if isinstance(module, nn.Embedding):
            print(name, module)

            if "weight" in config:
                print("quantize_embed_params")
                quantize_embed_params(module, config = config)

        if isinstance(module, LlamaAttention):
            print(name, module)

            if "attention" in config:
                print("quantize_attention_forward")
                module.forward = partial(quantize_attention_forward, module = module, quantize_config = config["attention"])


from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
from typing import List, Optional, Tuple, Union
import math

def quantize_attention_forward(
    module: LlamaAttention,
    quantize_config: dict,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = module.q_proj(hidden_states).view(bsz, q_len, module.num_heads, module.head_dim).transpose(1, 2)
    key_states = module.k_proj(hidden_states).view(bsz, q_len, module.num_heads, module.head_dim).transpose(1, 2)
    value_states = module.v_proj(hidden_states).view(bsz, q_len, module.num_heads, module.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = module.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    attn_weights = quantize_matmul(query_states, key_states.transpose(2, 3), quantize_config) / math.sqrt(module.head_dim)

    if attn_weights.size() != (bsz, module.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, module.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        )

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = quantize_matmul(attn_weights, value_states, quantize_config)

    if attn_output.size() != (bsz, module.num_heads, q_len, module.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, module.num_heads, q_len, module.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, module.hidden_size)

    attn_output = module.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value