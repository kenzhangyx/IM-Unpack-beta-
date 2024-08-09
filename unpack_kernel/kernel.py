
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import math

curr_path = os.path.dirname(os.path.realpath(__file__))
src_files = [os.path.join(curr_path, 'extension', file) for file in ['cuda_kernel.cu', 'torch_extension.cpp']]
unpack_v2 = load('unpack_v2', src_files, verbose = True)

import unpack_v2 as unpack

def row_unpack(inp, scale, bits = 4):
    L, D = inp.shape
    S = int(2 ** (bits - 1))
    assert D % S == 0
    
    cum_cnt = torch.cumsum(torch.ceil(torch.log2(inp.max(dim = -1).values * scale + 1) / (bits - 1)).int(), dim = -1, dtype = torch.int)
    X = cum_cnt[-1].item()
    
    # Convert X to int
    X = int(X)
    
    # Ensure that inp is of type float
    inp = inp.float()
    
    # bits should be passed as int
    bits = int(bits)

    out = unpack.row_unpack_fn(inp, cum_cnt, scale, X, bits)
    return out

def col_unpack(A_inp, B_inp, scale, bits = 4):
    D, L = A_inp.shape
    S = int(2 ** (bits - 1))
    assert L % S == 0

    A_bits = torch.ceil(torch.log2(A_inp.max(dim = -1).values * scale + 1) / (bits - 1)).int()
    B_bits = torch.ceil(torch.log2(B_inp.max(dim = -1).values * scale + 1) / (bits - 1)).int()
    cum_cnt = torch.cumsum(A_bits * B_bits, dim = -1, dtype = torch.int)
    X = cum_cnt[-1].item()
    
    A_out, B_out = unpack.col_unpack_fn(A_inp, B_inp, cum_cnt, A_bits, scale, X, bits)
    return A_out, B_out

def both_unpack(A_inp, B_inp, scale, bits=4):
  
    # Step 1: 对列进行解压，得到新 A 和 B
    A_out, B_out = col_unpack(A_inp, B_inp, scale, bits)

    # Step 2: 对得到的新 A 进行行解压
    A_final = row_unpack(A_out, scale, bits)

    # 返回最终的解压结果
    return A_final, B_out

def construct_matrix(H, W, bits, ratio):
    S = int(2 ** (bits - 1)) - 1
    A = torch.randint(-S, S + 1, size = (H, W), device = "cuda", dtype = torch.float)
    mask = (torch.randperm(H) < ((ratio - 1) * H)).float().cuda()
    A = A * (mask[:, None] + 1)
    return A

def profile(fn):
    N = 20
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        fn()
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / N

if __name__ == "__main__":

    #begin to test both

    A = construct_matrix(512, 512, 8, 1.1)
    B = construct_matrix(512, 512, 8, 1.1)
    A_final, B_final = both_unpack(A, B, 1, bits=8)
    print(A_final.shape, (A_final != 0).float().mean(dim=-1))
    print(B_final.shape, (B_final != 0).float().mean(dim=-1))

    A = construct_matrix(512, 512, 4, 1.1)
    B = construct_matrix(512, 512, 4, 1.1)
    A_final, B_final = both_unpack(A, B, 1, bits=4)
    print(A_final.shape, (A_final != 0).float().mean(dim=-1))
    print(B_final.shape, (B_final != 0).float().mean(dim=-1))
    
    A = construct_matrix(512, 512, 2, 1.1)
    B = construct_matrix(512, 512, 2, 1.1)
    A_final, B_final = both_unpack(A, B, 1, bits=2)
    print(A_final.shape, (A_final != 0).float().mean(dim=-1))
    print(B_final.shape, (B_final != 0).float().mean(dim=-1))