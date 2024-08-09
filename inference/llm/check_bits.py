from transformers import LlamaForCausalLM, LlamaTokenizer
from functools import partial
import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial

def huffman_code_tree(node, bin_string = ''):
    if type(node) is str:
        return {node: bin_string}
    (l, r) = node
    d = dict()
    d.update(huffman_code_tree(l, bin_string + '0'))
    d.update(huffman_code_tree(r, bin_string + '1'))
    return d

def make_tree(nodes):
    while len(nodes) > 1:
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]
        nodes = nodes[:-2]
        nodes.append(((key1, key2), c1 + c2))
        nodes = sorted(nodes, key = lambda x: x[1], reverse = True)
    return nodes[0][0]

def percentile_quantize_fn(inp, percentile, num_dist_vals):
    original_type = inp.dtype
    scale = (0.5 * num_dist_vals) / torch.topk(inp.abs().reshape(-1), k = int(inp.numel() * (1 - percentile)), dim = -1).values.min()
    if scale == 0:
        print("encounter scale = 0")
        scale = 1e-4
    inp = torch.round(inp * scale).int()
    return inp

def quantize_fn(inp, config):
    return percentile_quantize_fn(inp, config["percentile"], config["num_dist_vals"])

def quantize_and_stats(weight, config):
    weight = weight.cuda()
    weight = quantize_fn(weight, config)

    vals, counts = torch.unique(weight, return_counts = True)
    freq = {str(val):count for val, count in zip(vals.tolist(), counts.tolist())}
    node = make_tree(sorted(freq.items(), key = lambda x: x[1], reverse = True))
    encoding = huffman_code_tree(node)

    bits = sum([freq[key] * len(encoding[key]) for key in freq])
    return bits, weight.numel()


import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, required = True)
parser.add_argument("--config", type = str, required = True)
args = parser.parse_args()

model = LlamaForCausalLM.from_pretrained(args.model)
config = json.load(open(args.config, "r"))

total_params = 0
total_bits = 0
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
        bits, params = quantize_and_stats(module.weight.data, config["weight"])
        total_params = total_params + params
        total_bits = total_bits + bits
        print(name, module, total_bits / total_params)
