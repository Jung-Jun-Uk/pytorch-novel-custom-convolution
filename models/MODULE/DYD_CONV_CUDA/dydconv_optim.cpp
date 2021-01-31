#include <torch/extension.h>
#include <stdio.h>

//#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

torch::Tensor dydconv2d_op(const torch::Tensor &input,
                          const torch::Tensor &kernel, int up_h, int up_w,
                          int down_h, int down_w, int pad_h0, int pad_h1,
                          int pad_w0, int pad_w1, bool forward);


torch::Tensor dydconv2d(const torch::Tensor &input, const torch::Tensor &kernel,
                       int up_h, int up_w, int down_h, int down_w, int pad_h0,
                       int pad_h1, int pad_w0, int pad_w1, bool forward) {
  CHECK_CUDA(input);
  CHECK_CUDA(kernel);

  return dydconv2d_op(input, kernel, up_h, up_w, down_h, down_w, pad_h0, pad_h1,
                      pad_w0, pad_w1, forward);
}                    

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dydconv2d", &dydconv2d, "dydconv2d (CUDA)");
    //m.def("dydconv2d_backward_kernel", &dydconv2d_backward_kernel, "dydconv2d backward kernel (CUDA)");
}