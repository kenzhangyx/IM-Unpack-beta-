import torch
import math
import json
from unpack import unpack_row, unpack_column, unpack_both, unpack, scaled_matmul, pack_row, pack_transposed_row

A = (torch.randn(512, 1024) * 8).int()
B = (torch.randn(2048, 1024) * 8).int()
C = torch.matmul(A, B.T)
bit_width = 4

# 测试 both_unpack
A_final, B_final = unpack_both(A, B, 1, bits=bit_width)

# 进行矩阵乘法
result = torch.matmul(A_final.float(), B_final.T.float())

# 检查结果
print((C - result).abs().max())

scales = torch.ones(A.shape[1], device = A.device, dtype = A.dtype)
Au, Be, APi_indices, APi_scales, scales_u = unpack(A, B, scales, bit_width, unpack_row)
Beu, Aue, BPi_indices, BPi_scales, scales_uu = unpack(Be, Au, scales_u, bit_width, unpack_row)

AueSuuBeu = scaled_matmul(Aue, Beu, scales_uu)
APiAueSuuBeu = pack_row(AueSuuBeu, APi_indices, APi_scales)
APiAueSuuBeuBPi = pack_transposed_row(APiAueSuuBeu, BPi_indices, BPi_scales)

print((C - APiAueSuuBeuBPi).abs().max())
