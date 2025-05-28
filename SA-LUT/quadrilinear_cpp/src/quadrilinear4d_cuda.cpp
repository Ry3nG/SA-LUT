#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "quadrilinear4d_kernel.h"

int quadrilinear4d_forward_cuda(torch::Tensor lut, torch::Tensor image,
                                torch::Tensor output, int lut_dim, int shift,
                                float binsize, int width, int height,
                                int batch) {
  const float* lut_flat = lut.data_ptr<float>();
  const float* image_flat = image.data_ptr<float>();
  float* output_flat = output.data_ptr<float>();

  QuadriLinearForwardLaucher(lut_flat, image_flat, output_flat, lut_dim, shift,
                             binsize, width, height, batch,
                             at::cuda::getCurrentCUDAStream());
  return 1;
}

int quadrilinear4d_backward_cuda(torch::Tensor image, torch::Tensor image_grad,
                                 torch::Tensor lut, torch::Tensor lut_grad,
                                 int lut_dim, int shift, float binsize,
                                 int width, int height, int batch) {
  const float* image_flat = image.data_ptr<float>();
  float* image_grad_flat = image_grad.data_ptr<float>();
  const float* lut_flat = lut.data_ptr<float>();
  float* lut_grad_flat = lut_grad.data_ptr<float>();

  QuadriLinearBackwardLaucher(image_flat, image_grad_flat, lut_flat,
                              lut_grad_flat, lut_dim, shift, binsize, width,
                              height, batch, at::cuda::getCurrentCUDAStream());
  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &quadrilinear4d_forward_cuda, "Quadrilinear forward (CUDA)");
  m.def("backward", &quadrilinear4d_backward_cuda,
        "Quadrilinear backward (CUDA)");
}
