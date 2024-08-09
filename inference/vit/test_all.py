
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

def test_classification(net,test_loader,max_iteration=None, description=None):
    pos=0
    tot=0
    i = 0
    max_iteration = len(test_loader) if max_iteration is None else max_iteration
    with torch.no_grad():
        q=tqdm(test_loader, desc=description)
        for inp,target in q:
            i+=1
            inp=inp.cuda()
            target=target.cuda()
            out=net(inp)
            pos_num=torch.sum(out.argmax(1)==target).item()
            pos+=pos_num
            tot+=inp.size(0)
            q.set_postfix({"acc":pos/tot})
            if i >= max_iteration:
                break
    print(pos/tot)
    return pos/tot

def eval(name, args):

    net = timm.create_model(name, pretrained=True)

    if args.config is not None:
        config = json.load(open(args.config))
        from quantization import quantize
        quantize(net, config)
        
    net.cuda()
    net.eval()
    
    g = datasets.ViTImageNetLoaderGenerator('/data/imagenet','imagenet',256,256,16, kwargs = {"model":net})
    test_loader = g.test_loader()
    
    acc = test_classification(net,test_loader)
    print(f"model: {name} accuracy: {acc}\n")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, default = None)
    args = parser.parse_args()

    names = [
        "vit_tiny_patch16_224",
        "vit_small_patch16_224",
        "vit_medium_patch16_gap_256.sw_in12k_ft_in1k",
        "vit_base_patch16_224",
        "vit_large_patch16_224.augreg_in21k_ft_in1k",
        "vit_huge_patch14_clip_224.laion2b_ft_in1k",
    ]
    for name in names:
        eval(name, args)