#include "cuda_kernel.h"
#include <stdio.h>
#include <cuda.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
using namespace std;

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

#define INT8R 4
#define INT8D 5
#define INT8B 8
#define INT8S 128

#define INT4R 8
#define INT4D 11
#define INT4B 4
#define INT4S 8

#define INT2R 16
#define INT2D 31
#define INT2B 2
#define INT2S 2

__global__ void A_col_unpack_8bit(
  float *A_inp,    // [D, L]
  int *cum_cnt,  // [D]
  int *A_cnt,    // [D]
  int *A_out,      // [X, T = L / INT8R]
  float scale,
  int X, int L
) {
  const int D_idx = blockIdx.y;
  const int T_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int T = L / INT8R;

  int offset = 0;
  if (D_idx != 0) {
    offset = cum_cnt[D_idx - 1];
  }
  int count = cum_cnt[D_idx] - offset;
  int A_count = A_cnt[D_idx];
  int B_count = count / A_count;

  int vals[INT8R];
  int pack_vals = 0;
  int tmp_val;
  int tmp_sign;

  #pragma unroll
  for (int idx = 0; idx < INT8R; idx++) {
    vals[idx] = round(scale * A_inp[D_idx * L + idx * T + T_idx]);
  }

  for (int A_jdx = 0; A_jdx < A_count; A_jdx++) {
    pack_vals = 0;
    #pragma unroll
    for (int idx = 0; idx < INT8R; idx++) {
      tmp_val = vals[idx] % INT8S;
      tmp_sign = tmp_val < 0;
      pack_vals = (pack_vals << INT8B) | (tmp_sign << (INT8B - 1)) | (tmp_val & 127);
      vals[idx] = vals[idx] / INT8S;
    }
    for (int B_jdx = 0; B_jdx < B_count; B_jdx++) {
      A_out[(offset + A_jdx * B_count + B_jdx) * T + T_idx] = pack_vals;
    }
  }
}

__global__ void B_col_unpack_8bit(
  float *B_inp,    // [D, L]
  int *cum_cnt,  // [D]
  int *A_cnt,    // [D]
  int *B_out,      // [X, T = L / INT8R]
  float scale,
  int X, int L
) {
  const int D_idx = blockIdx.y;
  const int T_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int T = L / INT8R;

  int offset = 0;
  if (D_idx != 0) {
    offset = cum_cnt[D_idx - 1];
  }
  int count = cum_cnt[D_idx] - offset;
  int A_count = A_cnt[D_idx];
  int B_count = count / A_count;

  int vals[INT8R];
  int pack_vals = 0;
  int tmp_val;
  int tmp_sign;

  #pragma unroll
  for (int idx = 0; idx < INT8R; idx++) {
    vals[idx] = round(scale * B_inp[D_idx * L + idx * T + T_idx]);
  }

  for (int B_jdx = 0; B_jdx < B_count; B_jdx++) {
    pack_vals = 0;
    #pragma unroll
    for (int idx = 0; idx < INT8R; idx++) {
      tmp_val = vals[idx] % INT8S;
      tmp_sign = tmp_val < 0;
      pack_vals = (pack_vals << INT8B) | (tmp_sign << (INT8B - 1)) | (tmp_val & 127);
      vals[idx] = vals[idx] / INT8S;
    }
    for (int A_jdx = 0; A_jdx < A_count; A_jdx++) {
      B_out[(offset + A_jdx * B_count + B_jdx) * T + T_idx] = pack_vals;
    }
  }
}

__global__ void A_col_unpack_4bit(
  float *A_inp,    // [D, L]
  int *cum_cnt,  // [D]
  int *A_cnt,    // [D]
  int *A_out,      // [X, T = L / INT4R]
  float scale,
  int X, int L
) {
  const int D_idx = blockIdx.y;
  const int T_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int T = L / INT4R;

  int offset = 0;
  if (D_idx != 0) {
    offset = cum_cnt[D_idx - 1];
  }
  int count = cum_cnt[D_idx] - offset;
  int A_count = A_cnt[D_idx];
  int B_count = count / A_count;

  int vals[INT4R];
  int pack_vals = 0;
  int tmp_val;
  int tmp_sign;

  #pragma unroll
  for (int idx = 0; idx < INT4R; idx++) {
    vals[idx] = round(scale * A_inp[D_idx * L + idx * T + T_idx]);
  }

  for (int A_jdx = 0; A_jdx < A_count; A_jdx++) {
    pack_vals = 0;
    #pragma unroll
    for (int idx = 0; idx < INT4R; idx++) {
      tmp_val = vals[idx] % INT4S;
      tmp_sign = tmp_val < 0;
      pack_vals = (pack_vals << INT4B) | (tmp_sign << (INT4B - 1)) | (tmp_val & 7);
      vals[idx] = vals[idx] / INT4S;
    }
    for (int B_jdx = 0; B_jdx < B_count; B_jdx++) {
      A_out[(offset + A_jdx * B_count + B_jdx) * T + T_idx] = pack_vals;
    }
  }
}

__global__ void B_col_unpack_4bit(
  float *B_inp,    // [D, L]
  int *cum_cnt,  // [D]
  int *A_cnt,    // [D]
  int *B_out,      // [X, T = L / INT4R]
  float scale,
  int X, int L
) {
  const int D_idx = blockIdx.y;
  const int T_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int T = L / INT4R;

  int offset = 0;
  if (D_idx != 0) {
    offset = cum_cnt[D_idx - 1];
  }
  int count = cum_cnt[D_idx] - offset;
  int A_count = A_cnt[D_idx];
  int B_count = count / A_count;

  int vals[INT4R];
  int pack_vals = 0;
  int tmp_val;
  int tmp_sign;

  #pragma unroll
  for (int idx = 0; idx < INT4R; idx++) {
    vals[idx] = round(scale * B_inp[D_idx * L + idx * T + T_idx]);
  }

  for (int B_jdx = 0; B_jdx < B_count; B_jdx++) {
    pack_vals = 0;
    #pragma unroll
    for (int idx = 0; idx < INT4R; idx++) {
      tmp_val = vals[idx] % INT4S;
      tmp_sign = tmp_val < 0;
      pack_vals = (pack_vals << INT4B) | (tmp_sign << (INT4B - 1)) | (tmp_val & 7);
      vals[idx] = vals[idx] / INT4S;
    }
    for (int A_jdx = 0; A_jdx < A_count; A_jdx++) {
      B_out[(offset + A_jdx * B_count + B_jdx) * T + T_idx] = pack_vals;
    }
  }
}

__global__ void A_col_unpack_2bit(
  float *A_inp,    // [D, L]
  int *cum_cnt,  // [D]
  int *A_cnt,    // [D]
  int *A_out,      // [X, T = L / INT2R]
  float scale,
  int X, int L
) {
  const int D_idx = blockIdx.y;
  const int T_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int T = L / INT2R;

  int offset = 0;
  if (D_idx != 0) {
    offset = cum_cnt[D_idx - 1];
  }
  int count = cum_cnt[D_idx] - offset;
  int A_count = A_cnt[D_idx];
  int B_count = count / A_count;

  int vals[INT2R];
  int pack_vals = 0;
  int tmp_val;
  int tmp_sign;

  #pragma unroll
  for (int idx = 0; idx < INT2R; idx++) {
    vals[idx] = round(scale * A_inp[D_idx * L + idx * T + T_idx]);
  }

  for (int A_jdx = 0; A_jdx < A_count; A_jdx++) {
    pack_vals = 0;
    #pragma unroll
    for (int idx = 0; idx < INT2R; idx++) {
      tmp_val = vals[idx] % INT2S;
      tmp_sign = tmp_val < 0;
      pack_vals = (pack_vals << INT2B) | (tmp_sign << (INT2B - 1)) | (tmp_val & 1);
      vals[idx] = vals[idx] / INT2S;
    }
    for (int B_jdx = 0; B_jdx < B_count; B_jdx++) {
      A_out[(offset + A_jdx * B_count + B_jdx) * T + T_idx] = pack_vals;
    }
  }
}

__global__ void B_col_unpack_2bit(
  float *B_inp,    // [D, L]
  int *cum_cnt,  // [D]
  int *A_cnt,    // [D]
  int *B_out,      // [X, T = L / INT2R]
  float scale,
  int X, int L
) {
  const int D_idx = blockIdx.y;
  const int T_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int T = L / INT2R;

  int offset = 0;
  if (D_idx != 0) {
    offset = cum_cnt[D_idx - 1];
  }
  int count = cum_cnt[D_idx] - offset;
  int A_count = A_cnt[D_idx];
  int B_count = count / A_count;

  int vals[INT2R];
  int pack_vals = 0;
  int tmp_val;
  int tmp_sign;

  #pragma unroll
  for (int idx = 0; idx < INT2R; idx++) {
    vals[idx] = round(scale * B_inp[D_idx * L + idx * T + T_idx]);
  }

  for (int B_jdx = 0; B_jdx < B_count; B_jdx++) {
    pack_vals = 0;
    #pragma unroll
    for (int idx = 0; idx < INT2R; idx++) {
      tmp_val = vals[idx] % INT2S;
      tmp_sign = tmp_val < 0;
      pack_vals = (pack_vals << INT2B) | (tmp_sign << (INT2B - 1)) | (tmp_val & 1);
      vals[idx] = vals[idx] / INT2S;
    }
    for (int A_jdx = 0; A_jdx < A_count; A_jdx++) {
      B_out[(offset + A_jdx * B_count + B_jdx) * T + T_idx] = pack_vals;
    }
  }
}

std::vector<at::Tensor> col_unpack_launch(at::Tensor A_inp, at::Tensor B_inp, at::Tensor cum_cnt, at::Tensor A_cnt, float scale, int X, int bits) {
  int D = A_inp.size(0);
  int A_L = A_inp.size(1);
  int B_L = B_inp.size(1);
  
  if (bits == 8) {
    int A_T = A_L / INT8R;
    at::Tensor A_out = at::empty({X, A_T}, cum_cnt.options());
    dim3 A_threads(min(A_T, 1024));
    dim3 A_blocks(max(A_T / 1024, 1), D);
    A_col_unpack_8bit<<<A_blocks, A_threads>>>(
      A_inp.data_ptr<float>(), 
      cum_cnt.data_ptr<int>(), 
      A_cnt.data_ptr<int>(), 
      A_out.data_ptr<int>(),
      scale, X, A_L
    );

    int B_T = B_L / INT8R;
    at::Tensor B_out = at::empty({X, B_T}, cum_cnt.options());
    dim3 B_threads(min(B_T, 1024));
    dim3 B_blocks(max(B_T / 1024, 1), D);
    B_col_unpack_8bit<<<B_blocks, B_threads>>>(
      B_inp.data_ptr<float>(), 
      cum_cnt.data_ptr<int>(), 
      A_cnt.data_ptr<int>(), 
      B_out.data_ptr<int>(),
      scale, X, B_L
    );

    return {A_out, B_out};
  } else if (bits == 4) {
    int A_T = A_L / INT4R;
    at::Tensor A_out = at::empty({X, A_T}, cum_cnt.options());
    dim3 A_threads(min(A_T, 1024));
    dim3 A_blocks(max(A_T / 1024, 1), D);
    A_col_unpack_4bit<<<A_blocks, A_threads>>>(
      A_inp.data_ptr<float>(), 
      cum_cnt.data_ptr<int>(), 
      A_cnt.data_ptr<int>(), 
      A_out.data_ptr<int>(),
      scale, X, A_L
    );

    int B_T = B_L / INT4R;
    at::Tensor B_out = at::empty({X, B_T}, cum_cnt.options());
    dim3 B_threads(min(B_T, 1024));
    dim3 B_blocks(max(B_T / 1024, 1), D);
    B_col_unpack_4bit<<<B_blocks, B_threads>>>(
      B_inp.data_ptr<float>(), 
      cum_cnt.data_ptr<int>(), 
      A_cnt.data_ptr<int>(), 
      B_out.data_ptr<int>(),
      scale, X, B_L
    );

    return {A_out, B_out};
  } else if (bits == 2) {
    int A_T = A_L / INT2R;
    at::Tensor A_out = at::empty({X, A_T}, cum_cnt.options());
    dim3 A_threads(min(A_T, 1024));
    dim3 A_blocks(max(A_T / 1024, 1), D);
    A_col_unpack_2bit<<<A_blocks, A_threads>>>(
      A_inp.data_ptr<float>(), 
      cum_cnt.data_ptr<int>(), 
      A_cnt.data_ptr<int>(), 
      A_out.data_ptr<int>(),
      scale, X, A_L
    );

    int B_T = B_L / INT2R;
    at::Tensor B_out = at::empty({X, B_T}, cum_cnt.options());
    dim3 B_threads(min(B_T, 1024));
    dim3 B_blocks(max(B_T / 1024, 1), D);
    B_col_unpack_2bit<<<B_blocks, B_threads>>>(
      B_inp.data_ptr<float>(), 
      cum_cnt.data_ptr<int>(), 
      A_cnt.data_ptr<int>(), 
      B_out.data_ptr<int>(),
      scale, X, B_L
    );

    return {A_out, B_out};
  } else {
    return {A_inp, B_inp};
  }
}


__global__ void row_unpack_8bit(
  float *inp,    // [L, D]
  int *cum_cnt,  // [L]
  int *out,      // [X, T = D / INT8R]
  float scale,
  int X, int D
) {
  const int L_idx = blockIdx.y;
  const int T_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int T = D / INT8R;

  int offset = 0;
  if (L_idx != 0) {
    offset = cum_cnt[L_idx - 1];
  }
  int count = cum_cnt[L_idx] - offset;

  int vals[INT8R];
  int pack_vals = 0;
  int tmp_val;
  int tmp_sign;
  #pragma unroll
  for (int idx = 0; idx < INT8R; idx++) {
    vals[idx] = round(scale * inp[L_idx * D + idx * T + T_idx]);
  }

  for (int jdx = 0; jdx < count; jdx++) {
    pack_vals = 0;
    #pragma unroll
    for (int idx = 0; idx < INT8R; idx++) {
      tmp_val = vals[idx] % INT8S;
      tmp_sign = tmp_val < 0;
      pack_vals = (pack_vals << INT8B) | (tmp_sign << (INT8B - 1)) | (tmp_val & 127);
      vals[idx] = vals[idx] / INT8S;
    }
    out[(offset + jdx) * T + T_idx] = pack_vals;
  }
}

__global__ void row_unpack_4bit(
  float *inp,    // [L, D]
  int *cum_cnt,  // [L]
  int *out,      // [X, T = D / INT4R]
  float scale,
  int X, int D
) {
  const int L_idx = blockIdx.y;
  const int T_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int T = D / INT4R;

  int offset = 0;
  if (L_idx != 0) {
    offset = cum_cnt[L_idx - 1];
  }
  int count = cum_cnt[L_idx] - offset;

  int vals[INT4R];
  int pack_vals = 0;
  int tmp_val;
  int tmp_sign;
  #pragma unroll
  for (int idx = 0; idx < INT4R; idx++) {
    vals[idx] = round(scale * inp[L_idx * D + idx * T + T_idx]);
  }

  for (int jdx = 0; jdx < count; jdx++) {
    pack_vals = 0;
    #pragma unroll
    for (int idx = 0; idx < INT4R; idx++) {
      tmp_val = vals[idx] % INT4S;
      tmp_sign = tmp_val < 0;
      pack_vals = (pack_vals << INT4B) | (tmp_sign << (INT4B - 1)) | (tmp_val & 7);
      vals[idx] = vals[idx] / INT4S;
    }
    out[(offset + jdx) * T + T_idx] = pack_vals;
  }
}

__global__ void row_unpack_2bit(
  float *inp,    // [L, D]
  int *cum_cnt,  // [L]
  int *out,      // [X, T = D / INT2R]
  float scale,
  int X, int D
) {
  const int L_idx = blockIdx.y;
  const int T_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int T = D / INT2R;

  int offset = 0;
  if (L_idx != 0) {
    offset = cum_cnt[L_idx - 1];
  }
  int count = cum_cnt[L_idx] - offset;

  int vals[INT2R];
  int pack_vals = 0;
  int tmp_val;
  int tmp_sign;
  #pragma unroll
  for (int idx = 0; idx < INT2R; idx++) {
    vals[idx] = round(scale * inp[L_idx * D + idx * T + T_idx]);
  }

  for (int jdx = 0; jdx < count; jdx++) {
    pack_vals = 0;
    #pragma unroll
    for (int idx = 0; idx < INT2R; idx++) {
      tmp_val = vals[idx] % INT2S;
      tmp_sign = tmp_val < 0;
      pack_vals = (pack_vals << INT2B) | (tmp_sign << (INT2B - 1)) | (tmp_val & 1);
      vals[idx] = vals[idx] / INT2S;
    }
    out[(offset + jdx) * T + T_idx] = pack_vals;
  }
}


at::Tensor row_unpack_launch(at::Tensor inp, at::Tensor cum_cnt, float scale, int X, int bits) {
  int L = inp.size(0);
  int D = inp.size(1);
  
  if (bits == 8) {
    int T = D / INT8R;
    at::Tensor out = at::empty({X, T}, cum_cnt.options());
    dim3 threads(min(T, 1024));
    dim3 blocks(max(T / 1024, 1), L);
    row_unpack_8bit<<<blocks, threads>>>(
      inp.data_ptr<float>(), 
      cum_cnt.data_ptr<int>(), 
      out.data_ptr<int>(),
      scale, X, D
    );
    return out;
  } else if (bits == 4) {
    int T = D / INT4R;
    at::Tensor out = at::empty({X, T}, cum_cnt.options());
    dim3 threads(min(T, 1024));
    dim3 blocks(max(T / 1024, 1), L);
    row_unpack_4bit<<<blocks, threads>>>(
      inp.data_ptr<float>(), 
      cum_cnt.data_ptr<int>(), 
      out.data_ptr<int>(),
      scale, X, D
    );
    return out;
  } else if (bits == 2) {
    int T = D / INT2R;
    at::Tensor out = at::empty({X, T}, cum_cnt.options());
    dim3 threads(min(T, 1024));
    dim3 blocks(max(T / 1024, 1), L);
    row_unpack_2bit<<<blocks, threads>>>(
      inp.data_ptr<float>(), 
      cum_cnt.data_ptr<int>(), 
      out.data_ptr<int>(),
      scale, X, D
    );
    return out;
  } else {
    return inp;
  }
}
std::vector<at::Tensor> both_unpack_launch(at::Tensor A_inp, at::Tensor B_inp, at::Tensor cum_cnt, at::Tensor A_cnt, float scale, int X, int bits) {
  int D = A_inp.size(0);
  int A_L = A_inp.size(1);
  int B_L = B_inp.size(1);

  std::vector<at::Tensor> col_results = col_unpack_launch(A_inp, B_inp, cum_cnt, A_cnt, scale, X, bits);
  at::Tensor A_out = col_results[0];
  at::Tensor B_out = col_results[1];

  // Now perform row unpacking on the resulting A_out
  at::Tensor A_final = row_unpack_launch(A_out, cum_cnt, scale, X, bits);

  // The final B matrix remains as it is after col_unpack
  return {A_final, B_out};
}
