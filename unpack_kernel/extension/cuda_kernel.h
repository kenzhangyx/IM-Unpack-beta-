#include <torch/extension.h>
#include <ATen/ATen.h>
using namespace std;

at::Tensor row_unpack_launch(at::Tensor inp, at::Tensor cum_cnt, float scale, int X, int bits);
std::vector<at::Tensor> col_unpack_launch(at::Tensor A_inp, at::Tensor B_inp, at::Tensor cum_cnt, at::Tensor A_cnt, float scale, int X, int bits);
std::vector<at::Tensor> both_unpack_launch(at::Tensor A_inp, at::Tensor B_inp, at::Tensor cum_cnt, at::Tensor A_cnt, float scale, int X, int bits);
