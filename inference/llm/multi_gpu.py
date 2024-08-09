import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial

def forward(*args, **kwargs):
    module = kwargs["_module"]
    del kwargs["_module"]
    module.to(module._device)

    new_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            arg = arg.to(module._device)
        new_args.append(arg)
    new_kwargs = {}
    for key, arg in kwargs.items():
        if isinstance(arg, torch.Tensor):
            arg = arg.to(module._device)
        new_kwargs[key] = arg

    outputs = module._original_forward(*new_args, **new_kwargs)

    if isinstance(outputs, torch.Tensor):
        new_outputs = outputs.to("cpu")
    elif isinstance(outputs, tuple):
        new_outputs = []
        for out in outputs:
            if isinstance(out, torch.Tensor):
                out = out.to("cpu")
            new_outputs.append(out)
        new_outputs = tuple(new_outputs)
    elif isinstance(outputs, list):
        new_outputs = []
        for out in outputs:
            if isinstance(out, torch.Tensor):
                out = out.to("cpu")
            new_outputs.append(out)
    
    return new_outputs

def make_multi_gpu_llama(model, num_gpus):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    current_gpu = 0
    for name, module in model.named_modules():
        if isinstance(module, LlamaDecoderLayer) or isinstance(module, nn.Embedding) or "lm_head" in name or "model.norm" in name:
            if not hasattr(module, "_device"):
                print(name, module, f"cuda:{current_gpu}")
                module._device = f"cuda:{current_gpu}"
                module._original_forward = module.forward
                module.forward = partial(forward, _module = module)
                current_gpu = (current_gpu + 1) % num_gpus

def make_multi_gpu(model, num_gpus):
    from transformers import LlamaForCausalLM
    if isinstance(model, LlamaForCausalLM):
        make_multi_gpu_llama(model, num_gpus)
    else:
        raise Exception()