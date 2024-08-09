import torch

def unpack_column(A, B, scales, bit_width):
    scale = 2 ** (bit_width - 1)
    unpacked_A = [A.clone().detach()]
    unpacked_B = [B.clone().detach()]
    scales = [scales]
    
    while True:
        low_bit_vals = unpacked_A[-1] % scale
        high_bit_vals = unpacked_A[-1] // scale

        sparsity = torch.mean((unpacked_A[-1].abs() >= scale).float(), dim = 0)
        sparsity_mask = sparsity > 0
        count_sparsity = sparsity_mask.int().sum().item()

        if count_sparsity == 0:
            break
        
        unpacked_A[-1][:, sparsity_mask] = low_bit_vals[:, sparsity_mask]
        unpacked_A.append(high_bit_vals[:, sparsity_mask])
        unpacked_B.append(unpacked_B[-1][:, sparsity_mask])
        scales.append(scales[-1][sparsity_mask] * scale)
    
    unpacked_A = torch.cat(unpacked_A, dim = 1)
    unpacked_B = torch.cat(unpacked_B, dim = 1)
    scales = torch.cat(scales, dim = 0)
    return unpacked_A, unpacked_B, scales

def unpack_row(A, bit_width):
    scale = 2 ** (bit_width - 1)
    unpacked_A = [A.clone().detach()]
    indices = [torch.arange(A.shape[0], device = A.device)]
    scales = [torch.ones(A.shape[0], device = A.device, dtype = torch.int)]
    
    while True:
        
        low_bit_vals = unpacked_A[-1] % scale
        high_bit_vals = unpacked_A[-1] // scale
    
        sparsity = torch.mean((unpacked_A[-1].abs() >= scale).float(), dim = 1)
        sparsity_mask = sparsity > 0
        count_sparsity = sparsity_mask.int().sum().item()

        if count_sparsity == 0:
            break
        
        unpacked_A[-1][sparsity_mask, :] = low_bit_vals[sparsity_mask, :]
        unpacked_A.append(high_bit_vals[sparsity_mask, :])
        indices.append(indices[-1][sparsity_mask])
        scales.append(scales[-1][sparsity_mask] * scale)

    unpacked_A = torch.cat(unpacked_A, dim = 0)
    indices = torch.cat(indices, dim = 0)
    scales = torch.cat(scales, dim = 0)
    return unpacked_A, indices, scales

def expend_mat(M, size, dim):
    if dim == 0:
        extra = torch.zeros(size, M.shape[1], dtype = M.dtype, device = M.device)
        return torch.cat([M, extra], dim = 0)
    elif dim == 1:
        extra = torch.zeros(M.shape[0], size, dtype = M.dtype, device = M.device)
        return torch.cat([M, extra], dim = 1)
    else:
        raise Exception()

def expend_vec(v, size):
    extra = torch.zeros(size, dtype = v.dtype, device = v.device)
    return torch.cat([v, extra], dim = 0)

def unpack_both(A, B, scales, bit_width):
    scale = 2 ** (bit_width - 1)
    unpacked_A = A.clone().detach()
    unpacked_B = B.clone().detach()
    Pi_indices = torch.arange(A.shape[0], device = A.device)
    Pi_scales = torch.ones(A.shape[0], device = A.device, dtype = torch.int)

    insert_pointer_i = A.shape[0]
    insert_pointer_j = A.shape[1]
    
    while True:
        sparsity_mask = unpacked_A.abs() >= scale
        col_sparsity = torch.sum(sparsity_mask.int(), dim = 1)
        row_sparsity = torch.sum(sparsity_mask.int(), dim = 0)
    
        col_val, col_idx = torch.max(col_sparsity, dim = 0)
        row_val, row_idx = torch.max(row_sparsity, dim = 0)

        if col_val == 0 and row_val == 0:
            break
    
        if col_val >= row_val:
            if insert_pointer_i >= unpacked_A.shape[0]:
                unpacked_A = expend_mat(unpacked_A, A.shape[0], 0)
                Pi_indices = expend_vec(Pi_indices, A.shape[0])
                Pi_scales = expend_vec(Pi_scales, A.shape[0])
                
            vals = unpacked_A[col_idx, :]
            
            unpacked_A[insert_pointer_i, :] = vals // scale
            unpacked_A[col_idx, :] = vals % scale
            
            Pi_indices[insert_pointer_i] = Pi_indices[col_idx]
            Pi_scales[insert_pointer_i] = Pi_scales[col_idx] * scale
            insert_pointer_i = insert_pointer_i + 1
        else:
            if insert_pointer_j >= unpacked_A.shape[1]:
                unpacked_A = expend_mat(unpacked_A, A.shape[1], 1)
                unpacked_B = expend_mat(unpacked_B, B.shape[1], 1)
                scales = expend_vec(scales, A.shape[1])
                
            vals = unpacked_A[:, row_idx]
            
            unpacked_A[:, insert_pointer_j] = vals // scale
            unpacked_A[:, row_idx] = vals % scale
            
            unpacked_B[:, insert_pointer_j] = unpacked_B[:, row_idx]
            scales[insert_pointer_j] = scales[row_idx] * scale
            insert_pointer_j = insert_pointer_j + 1

    unpacked_A = unpacked_A[:insert_pointer_i, :insert_pointer_j]
    unpacked_B = unpacked_B[:, :insert_pointer_j]
    Pi_indices = Pi_indices[:insert_pointer_i]
    Pi_scales = Pi_scales[:insert_pointer_i]
    scales = scales[:insert_pointer_j]
    return unpacked_A, unpacked_B, Pi_indices, Pi_scales, scales

def unpack(A, B, scales, bit_width, strategy):
    if strategy == unpack_row:
        A, Pi_indices, Pi_scales = unpack_row(A, bit_width)
    elif strategy == unpack_column:
        A, B, scales = unpack_column(A, B, scales, bit_width)
        Pi_indices = torch.arange(A.shape[0], device = A.device)
        Pi_scales = torch.ones(A.shape[0], device = A.device, dtype = A.dtype)
    elif strategy == unpack_both:
        A, B, Pi_indices, Pi_scales, scales = unpack_both(A, B, scales, bit_width)
    return A, B, Pi_indices, Pi_scales, scales

def scaled_matmul(unpacked_A, unpacked_B, scales):
    return torch.matmul(unpacked_A.float() * scales.float(), unpacked_B.T.float())

def pack_row(A, indices, scales):
    A, scales = A.float(), scales.float()
    packed_A = torch.zeros(indices.max() + 1, A.shape[1], device = A.device, dtype = A.dtype)
    packed_A.index_add_(0, indices, A * scales[:, None])
    return packed_A

def pack_transposed_row(A, indices, scales):
    A, scales = A.float(), scales.float()
    packed_A = torch.zeros(A.shape[0], indices.max() + 1, device = A.device, dtype = A.dtype)
    packed_A.index_add_(1, indices, A * scales)
    return packed_A