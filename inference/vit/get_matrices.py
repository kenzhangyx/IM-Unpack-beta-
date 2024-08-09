import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention
from functools import partial
import torch.nn.functional as F
import timm

from torch.nn.modules import module
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import pickle as pkl
import types
import time
import datasets
import argparse
import timm
import json
import torch

def output_linear(x, module, name):
    name = name.replace(".", "-")
    print(name)
    torch.save({"x":x, "weight":module.weight.data, "name":name}, f"{name}.th")
    
    y = F.linear(x, module.weight, module.bias)
    return y

def output_matmul(A, B, name):
    name = name.replace(".", "-")
    print(name)
    torch.save({"A":A, "B":B, "name":name}, f"{name}.th")
    
    C = torch.matmul(A, B)
    return C

def output_attention_forward(x, module, name):
    B, N, C = x.shape
    qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = module.q_norm(q), module.k_norm(k)

    q = q * module.scale
    attn = output_matmul(q, k.transpose(-2, -1), f"{name}.attn_score")
    attn = attn.softmax(dim=-1)
    attn = module.attn_drop(attn)
    x = output_matmul(attn, v, f"{name}.attn_out")

    x = x.transpose(1, 2).reshape(B, N, C)
    x = module.proj(x)
    x = module.proj_drop(x)
    return x

def modify_forward(model):
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            print(name, module, "output_attention_forward")
            module.forward = partial(output_attention_forward, module = module, name = name)

        if isinstance(module, nn.Linear):
            print(name, module, "output_linear")
            module.forward = partial(output_linear, module = module, name = name)


name = "vit_large_patch16_224.augreg_in21k_ft_in1k"
net = timm.create_model(name, pretrained=True)
modify_forward(net)
net.cuda()
net.eval()
    
g = datasets.ViTImageNetLoaderGenerator('/data/imagenet','imagenet',1,1,16, kwargs = {"model":net})
test_loader = g.test_loader()

for inp, target in test_loader:
    break

inp = inp.cuda()
target = target.cuda()
out = net(inp)

