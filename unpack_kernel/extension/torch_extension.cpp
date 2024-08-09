#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_kernel.h"
using namespace std;

at::Tensor row_unpack_fn(at::Tensor inp, at::Tensor cum_cnt, float scale, int X, int bits) {
  return row_unpack_launch(inp, cum_cnt, scale, X, bits);
}

std::vector<at::Tensor> col_unpack_fn(at::Tensor A_inp, at::Tensor B_inp, at::Tensor cum_cnt, at::Tensor A_cnt, float scale, int X, int bits) {
  return col_unpack_launch(A_inp, B_inp, cum_cnt, A_cnt, scale, X, bits);
}

std::vector<at::Tensor> both_unpack_fn(at::Tensor A_inp, at::Tensor B_inp, at::Tensor cum_cnt, at::Tensor A_cnt, float scale, int X, int bits) {
  return both_unpack_launch(A_inp, B_inp, cum_cnt, A_cnt, scale, X, bits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("row_unpack_fn", &row_unpack_fn, "row_unpack_fn");
  m.def("col_unpack_fn", &col_unpack_fn, "col_unpack_fn");
  m.def("both_unpack_fn", &both_unpack_fn, "both_unpack_fn");
}
