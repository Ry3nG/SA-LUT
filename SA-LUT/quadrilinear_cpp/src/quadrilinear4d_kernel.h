#ifndef QUADRILINEAR_CUDA_H
#define QUADRILINEAR_CUDA_H

#include <cuda_runtime.h>

int QuadriLinearForwardLaucher(const float* lut, const float* image,
                               float* output, const int lut_dim,
                               const int shift, const float binsize,
                               const int width, const int height,
                               const int batch, cudaStream_t stream);

int QuadriLinearBackwardLaucher(const float* image, float* image_grad,
                                const float* lut, float* lut_grad,
                                const int lut_dim, const int shift,
                                const float binsize, const int width,
                                const int height, const int batch,
                                cudaStream_t stream);

#endif
