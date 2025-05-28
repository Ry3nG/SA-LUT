#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#include "quadrilinear4d_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__global__ void QuadriLinearForward(const int nthreads, const float* lut,
                                    const float* image, float* output,
                                    const int dim, const int shift,
                                    const float binsize, const int width,
                                    const int height, const int batch) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    float context = image[index];
    float r = image[index + width * height * batch];
    float g = image[index + width * height * batch * 2];
    float b = image[index + width * height * batch * 3];

    int context_id = 0;
    int r_id = floor(r / binsize);
    int g_id = floor(g / binsize);
    int b_id = floor(b / binsize);

    float context_d = context;
    float r_d = fmod(r, binsize) / binsize;
    float g_d = fmod(g, binsize) / binsize;
    float b_d = fmod(b, binsize) / binsize;

    int id0000 =
        context_id * dim * dim * dim + r_id + g_id * dim + b_id * dim * dim;
    int id0100 = context_id * dim * dim * dim + (r_id + 1) + g_id * dim +
                 b_id * dim * dim;
    int id0010 = context_id * dim * dim * dim + r_id + (g_id + 1) * dim +
                 b_id * dim * dim;
    int id0001 = context_id * dim * dim * dim + r_id + g_id * dim +
                 (b_id + 1) * dim * dim;
    int id0110 = context_id * dim * dim * dim + (r_id + 1) + (g_id + 1) * dim +
                 b_id * dim * dim;
    int id0011 = context_id * dim * dim * dim + r_id + (g_id + 1) * dim +
                 (b_id + 1) * dim * dim;
    int id0101 = context_id * dim * dim * dim + (r_id + 1) + g_id * dim +
                 (b_id + 1) * dim * dim;
    int id0111 = context_id * dim * dim * dim + (r_id + 1) + (g_id + 1) * dim +
                 (b_id + 1) * dim * dim;

    int id1000 = (context_id + 1) * dim * dim * dim + r_id + g_id * dim +
                 b_id * dim * dim;
    int id1100 = (context_id + 1) * dim * dim * dim + (r_id + 1) + g_id * dim +
                 b_id * dim * dim;
    int id1010 = (context_id + 1) * dim * dim * dim + r_id + (g_id + 1) * dim +
                 b_id * dim * dim;
    int id1001 = (context_id + 1) * dim * dim * dim + r_id + g_id * dim +
                 (b_id + 1) * dim * dim;
    int id1110 = (context_id + 1) * dim * dim * dim + (r_id + 1) +
                 (g_id + 1) * dim + b_id * dim * dim;
    int id1011 = (context_id + 1) * dim * dim * dim + r_id + (g_id + 1) * dim +
                 (b_id + 1) * dim * dim;
    int id1101 = (context_id + 1) * dim * dim * dim + (r_id + 1) + g_id * dim +
                 (b_id + 1) * dim * dim;
    int id1111 = (context_id + 1) * dim * dim * dim + (r_id + 1) +
                 (g_id + 1) * dim + (b_id + 1) * dim * dim;

    float w0000 = (1 - context_d) * (1 - r_d) * (1 - g_d) * (1 - b_d);
    float w0100 = (1 - context_d) * r_d * (1 - g_d) * (1 - b_d);
    float w0010 = (1 - context_d) * (1 - r_d) * g_d * (1 - b_d);
    float w0001 = (1 - context_d) * (1 - r_d) * (1 - g_d) * b_d;
    float w0110 = (1 - context_d) * r_d * g_d * (1 - b_d);
    float w0011 = (1 - context_d) * (1 - r_d) * g_d * b_d;
    float w0101 = (1 - context_d) * r_d * (1 - g_d) * b_d;
    float w0111 = (1 - context_d) * r_d * g_d * b_d;

    float w1000 = context_d * (1 - r_d) * (1 - g_d) * (1 - b_d);
    float w1100 = context_d * r_d * (1 - g_d) * (1 - b_d);
    float w1010 = context_d * (1 - r_d) * g_d * (1 - b_d);
    float w1001 = context_d * (1 - r_d) * (1 - g_d) * b_d;
    float w1110 = context_d * r_d * g_d * (1 - b_d);
    float w1011 = context_d * (1 - r_d) * g_d * b_d;
    float w1101 = context_d * r_d * (1 - g_d) * b_d;
    float w1111 = context_d * r_d * g_d * b_d;

    output[index] =
        w0000 * lut[id0000] + w0100 * lut[id0100] + w0010 * lut[id0010] +
        w0001 * lut[id0001] + w0110 * lut[id0110] + w0011 * lut[id0011] +
        w0101 * lut[id0101] + w0111 * lut[id0111] + w1000 * lut[id1000] +
        w1100 * lut[id1100] + w1010 * lut[id1010] + w1001 * lut[id1001] +
        w1110 * lut[id1110] + w1011 * lut[id1011] + w1101 * lut[id1101] +
        w1111 * lut[id1111];

    output[index + width * height * batch] =
        w0000 * lut[id0000 + shift] + w0100 * lut[id0100 + shift] +
        w0010 * lut[id0010 + shift] + w0001 * lut[id0001 + shift] +
        w0110 * lut[id0110 + shift] + w0011 * lut[id0011 + shift] +
        w0101 * lut[id0101 + shift] + w0111 * lut[id0111 + shift] +
        w1000 * lut[id1000 + shift] + w1100 * lut[id1100 + shift] +
        w1010 * lut[id1010 + shift] + w1001 * lut[id1001 + shift] +
        w1110 * lut[id1110 + shift] + w1011 * lut[id1011 + shift] +
        w1101 * lut[id1101 + shift] + w1111 * lut[id1111 + shift];

    output[index + width * height * batch * 2] =
        w0000 * lut[id0000 + shift * 2] + w0100 * lut[id0100 + shift * 2] +
        w0010 * lut[id0010 + shift * 2] + w0001 * lut[id0001 + shift * 2] +
        w0110 * lut[id0110 + shift * 2] + w0011 * lut[id0011 + shift * 2] +
        w0101 * lut[id0101 + shift * 2] + w0111 * lut[id0111 + shift * 2] +
        w1000 * lut[id1000 + shift * 2] + w1100 * lut[id1100 + shift * 2] +
        w1010 * lut[id1010 + shift * 2] + w1001 * lut[id1001 + shift * 2] +
        w1110 * lut[id1110 + shift * 2] + w1011 * lut[id1011 + shift * 2] +
        w1101 * lut[id1101 + shift * 2] + w1111 * lut[id1111 + shift * 2];
  }
}

int QuadriLinearForwardLaucher(const float* lut, const float* image,
                               float* output, const int lut_dim,
                               const int shift, const float binsize,
                               const int width, const int height,
                               const int batch, cudaStream_t stream) {
  const int kThreadsPerBlock = 1024;
  const int output_size = height * width * batch;
  cudaError_t err;

  QuadriLinearForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                        kThreadsPerBlock, 0, stream>>>(
      output_size, lut, image, output, lut_dim, shift, binsize, width, height,
      batch);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    printf("CUDA kernel failed : %s\n", cudaGetErrorString(err));
    return 0;
  }

  return 1;
}

__global__ void QuadriLinearBackward(const int nthreads, const float* image,
                                     float* image_grad, const float* lut,
                                     float* lut_grad, const int dim,
                                     const int shift, const float binsize,
                                     const int width, const int height,
                                     const int batch) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    float context = image[index];
    float r = image[index + width * height * batch];
    float g = image[index + width * height * batch * 2];
    float b = image[index + width * height * batch * 3];

    int context_id = 0;
    int r_id = floor(r / binsize);
    int g_id = floor(g / binsize);
    int b_id = floor(b / binsize);

    float context_d = context;
    float r_d = fmod(r, binsize) / binsize;
    float g_d = fmod(g, binsize) / binsize;
    float b_d = fmod(b, binsize) / binsize;

    int id0000 =
        context_id * dim * dim * dim + r_id + g_id * dim + b_id * dim * dim;
    int id0100 = context_id * dim * dim * dim + (r_id + 1) + g_id * dim +
                 b_id * dim * dim;
    int id0010 = context_id * dim * dim * dim + r_id + (g_id + 1) * dim +
                 b_id * dim * dim;
    int id0001 = context_id * dim * dim * dim + r_id + g_id * dim +
                 (b_id + 1) * dim * dim;
    int id0110 = context_id * dim * dim * dim + (r_id + 1) + (g_id + 1) * dim +
                 b_id * dim * dim;
    int id0011 = context_id * dim * dim * dim + r_id + (g_id + 1) * dim +
                 (b_id + 1) * dim * dim;
    int id0101 = context_id * dim * dim * dim + (r_id + 1) + g_id * dim +
                 (b_id + 1) * dim * dim;
    int id0111 = context_id * dim * dim * dim + (r_id + 1) + (g_id + 1) * dim +
                 (b_id + 1) * dim * dim;

    int id1000 = (context_id + 1) * dim * dim * dim + r_id + g_id * dim +
                 b_id * dim * dim;
    int id1100 = (context_id + 1) * dim * dim * dim + (r_id + 1) + g_id * dim +
                 b_id * dim * dim;
    int id1010 = (context_id + 1) * dim * dim * dim + r_id + (g_id + 1) * dim +
                 b_id * dim * dim;
    int id1001 = (context_id + 1) * dim * dim * dim + r_id + g_id * dim +
                 (b_id + 1) * dim * dim;
    int id1110 = (context_id + 1) * dim * dim * dim + (r_id + 1) +
                 (g_id + 1) * dim + b_id * dim * dim;
    int id1011 = (context_id + 1) * dim * dim * dim + r_id + (g_id + 1) * dim +
                 (b_id + 1) * dim * dim;
    int id1101 = (context_id + 1) * dim * dim * dim + (r_id + 1) + g_id * dim +
                 (b_id + 1) * dim * dim;
    int id1111 = (context_id + 1) * dim * dim * dim + (r_id + 1) +
                 (g_id + 1) * dim + (b_id + 1) * dim * dim;

    float w0000 = (1 - context_d) * (1 - r_d) * (1 - g_d) * (1 - b_d);
    float w0100 = (1 - context_d) * r_d * (1 - g_d) * (1 - b_d);
    float w0010 = (1 - context_d) * (1 - r_d) * g_d * (1 - b_d);
    float w0001 = (1 - context_d) * (1 - r_d) * (1 - g_d) * b_d;
    float w0110 = (1 - context_d) * r_d * g_d * (1 - b_d);
    float w0011 = (1 - context_d) * (1 - r_d) * g_d * b_d;
    float w0101 = (1 - context_d) * r_d * (1 - g_d) * b_d;
    float w0111 = (1 - context_d) * r_d * g_d * b_d;

    float w1000 = context_d * (1 - r_d) * (1 - g_d) * (1 - b_d);
    float w1100 = context_d * r_d * (1 - g_d) * (1 - b_d);
    float w1010 = context_d * (1 - r_d) * g_d * (1 - b_d);
    float w1001 = context_d * (1 - r_d) * (1 - g_d) * b_d;
    float w1110 = context_d * r_d * g_d * (1 - b_d);
    float w1011 = context_d * (1 - r_d) * g_d * b_d;
    float w1101 = context_d * r_d * (1 - g_d) * b_d;
    float w1111 = context_d * r_d * g_d * b_d;

    atomicAdd(lut_grad + id0000,
              image_grad[index + width * height * batch] * w0000);
    atomicAdd(lut_grad + id0100,
              image_grad[index + width * height * batch] * w0100);
    atomicAdd(lut_grad + id0010,
              image_grad[index + width * height * batch] * w0010);
    atomicAdd(lut_grad + id0001,
              image_grad[index + width * height * batch] * w0001);
    atomicAdd(lut_grad + id0110,
              image_grad[index + width * height * batch] * w0110);
    atomicAdd(lut_grad + id0011,
              image_grad[index + width * height * batch] * w0011);
    atomicAdd(lut_grad + id0101,
              image_grad[index + width * height * batch] * w0101);
    atomicAdd(lut_grad + id0111,
              image_grad[index + width * height * batch] * w0111);

    atomicAdd(lut_grad + id1000,
              image_grad[index + width * height * batch] * w1000);
    atomicAdd(lut_grad + id1100,
              image_grad[index + width * height * batch] * w1100);
    atomicAdd(lut_grad + id1010,
              image_grad[index + width * height * batch] * w1010);
    atomicAdd(lut_grad + id1001,
              image_grad[index + width * height * batch] * w1001);
    atomicAdd(lut_grad + id1110,
              image_grad[index + width * height * batch] * w1110);
    atomicAdd(lut_grad + id1011,
              image_grad[index + width * height * batch] * w1011);
    atomicAdd(lut_grad + id1101,
              image_grad[index + width * height * batch] * w1101);
    atomicAdd(lut_grad + id1111,
              image_grad[index + width * height * batch] * w1111);

    atomicAdd(lut_grad + id0000 + shift,
              image_grad[index + width * height * batch * 2] * w0000);
    atomicAdd(lut_grad + id0100 + shift,
              image_grad[index + width * height * batch * 2] * w0100);
    atomicAdd(lut_grad + id0010 + shift,
              image_grad[index + width * height * batch * 2] * w0010);
    atomicAdd(lut_grad + id0001 + shift,
              image_grad[index + width * height * batch * 2] * w0001);
    atomicAdd(lut_grad + id0110 + shift,
              image_grad[index + width * height * batch * 2] * w0110);
    atomicAdd(lut_grad + id0011 + shift,
              image_grad[index + width * height * batch * 2] * w0011);
    atomicAdd(lut_grad + id0101 + shift,
              image_grad[index + width * height * batch * 2] * w0101);
    atomicAdd(lut_grad + id0111 + shift,
              image_grad[index + width * height * batch * 2] * w0111);

    atomicAdd(lut_grad + id1000 + shift,
              image_grad[index + width * height * batch * 2] * w1000);
    atomicAdd(lut_grad + id1100 + shift,
              image_grad[index + width * height * batch * 2] * w1100);
    atomicAdd(lut_grad + id1010 + shift,
              image_grad[index + width * height * batch * 2] * w1010);
    atomicAdd(lut_grad + id1001 + shift,
              image_grad[index + width * height * batch * 2] * w1001);
    atomicAdd(lut_grad + id1110 + shift,
              image_grad[index + width * height * batch * 2] * w1110);
    atomicAdd(lut_grad + id1011 + shift,
              image_grad[index + width * height * batch * 2] * w1011);
    atomicAdd(lut_grad + id1101 + shift,
              image_grad[index + width * height * batch * 2] * w1101);
    atomicAdd(lut_grad + id1111 + shift,
              image_grad[index + width * height * batch * 2] * w1111);

    atomicAdd(lut_grad + id0000 + shift * 2,
              image_grad[index + width * height * batch * 3] * w0000);
    atomicAdd(lut_grad + id0100 + shift * 2,
              image_grad[index + width * height * batch * 3] * w0100);
    atomicAdd(lut_grad + id0010 + shift * 2,
              image_grad[index + width * height * batch * 3] * w0010);
    atomicAdd(lut_grad + id0001 + shift * 2,
              image_grad[index + width * height * batch * 3] * w0001);
    atomicAdd(lut_grad + id0110 + shift * 2,
              image_grad[index + width * height * batch * 3] * w0110);
    atomicAdd(lut_grad + id0011 + shift * 2,
              image_grad[index + width * height * batch * 3] * w0011);
    atomicAdd(lut_grad + id0101 + shift * 2,
              image_grad[index + width * height * batch * 3] * w0101);
    atomicAdd(lut_grad + id0111 + shift * 2,
              image_grad[index + width * height * batch * 3] * w0111);

    atomicAdd(lut_grad + id1000 + shift * 2,
              image_grad[index + width * height * batch * 3] * w1000);
    atomicAdd(lut_grad + id1100 + shift * 2,
              image_grad[index + width * height * batch * 3] * w1100);
    atomicAdd(lut_grad + id1010 + shift * 2,
              image_grad[index + width * height * batch * 3] * w1010);
    atomicAdd(lut_grad + id1001 + shift * 2,
              image_grad[index + width * height * batch * 3] * w1001);
    atomicAdd(lut_grad + id1110 + shift * 2,
              image_grad[index + width * height * batch * 3] * w1110);
    atomicAdd(lut_grad + id1011 + shift * 2,
              image_grad[index + width * height * batch * 3] * w1011);
    atomicAdd(lut_grad + id1101 + shift * 2,
              image_grad[index + width * height * batch * 3] * w1101);
    atomicAdd(lut_grad + id1111 + shift * 2,
              image_grad[index + width * height * batch * 3] * w1111);

    // atomicAdd(image_grad + index, (image_grad[index + width * height * batch]
    // + image_grad[index + width * height * batch * 2] + image_grad[index +
    // width * height * batch * 3]) / 3);

    // atomicAdd(image_grad + index, image_grad[index + width * height * batch]
    // *
    //            (   (-1)*(1-r_d)*(1-g_d)*(1-b_d) * lut[id0000] +
    //                1*(1-r_d)*(1-g_d)*(1-b_d) * lut[id1000] +
    //                (-1)*r_d*(1-g_d)*(1-b_d) * lut[id0100] +
    //                (-1)*(1-r_d)*g_d*(1-b_d) * lut[id0010] +
    //                (-1)*(1-r_d)*(1-g_d)*b_d * lut[id0001] +
    //                1*r_d*(1-g_d)*(1-b_d) * lut[id1100] +
    //                (-1)*r_d*g_d*(1-b_d) * lut[id0110] +
    //                (-1)*(1-r_d)*g_d*b_d * lut[id0011] +
    //                1*(1-r_d)*g_d*(1-b_d) * lut[id1010] +
    //                1*(1-r_d)*(1-g_d)*b_d * lut[id1001] +
    //                (-1)*r_d*(1-g_d)*b_d * lut[id0101] +
    //                1*r_d*g_d*(1-b_d) * lut[id1110] +
    //                (-1)*r_d*g_d*b_d * lut[id0111] +
    //                1*r_d*(1-g_d)*b_d * lut[id1101] +
    //                1*(1-r_d)*g_d*b_d * lut[id1011] +
    //                1*r_d*g_d*b_d * lut[id1111]
    //             )
    //          );

    // atomicAdd(image_grad + index, image_grad[index + width * height * batch]
    // * ((-1)*(1-r_d)*(1-g_d)*(1-b_d)*lut[id0000])); atomicAdd(image_grad +
    // index, image_grad[index + width * height * batch] *
    // (1*(1-r_d)*(1-g_d)*(1-b_d) * lut[id1000])); atomicAdd(image_grad + index,
    // image_grad[index + width * height * batch] * ((-1)*r_d*(1-g_d)*(1-b_d) *
    // lut[id0100])); atomicAdd(image_grad + index, image_grad[index + width *
    // height * batch] * ((-1)*(1-r_d)*g_d*(1-b_d) * lut[id0010]));
    // atomicAdd(image_grad + index, image_grad[index + width * height * batch]
    // * ((-1)*(1-r_d)*(1-g_d)*b_d * lut[id0001])); atomicAdd(image_grad +
    // index, image_grad[index + width * height * batch] *
    // (1*r_d*(1-g_d)*(1-b_d) * lut[id1100])); atomicAdd(image_grad + index,
    // image_grad[index + width * height * batch] * ((-1)*r_d*g_d*(1-b_d) *
    // lut[id0110])); atomicAdd(image_grad + index, image_grad[index + width *
    // height * batch] * ((-1)*(1-r_d)*g_d*b_d * lut[id0011]));
    // atomicAdd(image_grad + index, image_grad[index + width * height * batch]
    // * (1*(1-r_d)*g_d*(1-b_d) * lut[id1010])); atomicAdd(image_grad + index,
    // image_grad[index + width * height * batch] * (1*(1-r_d)*(1-g_d)*b_d *
    // lut[id1001])); atomicAdd(image_grad + index, image_grad[index + width *
    // height * batch] * ((-1)*r_d*(1-g_d)*b_d * lut[id0101]));
    // atomicAdd(image_grad + index, image_grad[index + width * height * batch]
    // * (1*r_d*g_d*(1-b_d) * lut[id1110])); atomicAdd(image_grad + index,
    // image_grad[index + width * height * batch] * ((-1)*r_d*g_d*b_d *
    // lut[id0111])); atomicAdd(image_grad + index, image_grad[index + width *
    // height * batch] * (1*r_d*(1-g_d)*b_d * lut[id1101]));
    // atomicAdd(image_grad + index, image_grad[index + width * height * batch]
    // * (1*(1-r_d)*g_d*b_d * lut[id1011])); atomicAdd(image_grad + index,
    // image_grad[index + width * height * batch] * (1*r_d*g_d*b_d *
    // lut[id1111]));

    // atomicAdd(image_grad + index, image_grad[index + width * height * batch *
    // 2] *
    //            (   (-1)*(1-r_d)*(1-g_d)*(1-b_d) * lut[id0000] +
    //                1*(1-r_d)*(1-g_d)*(1-b_d) * lut[id1000] +
    //                (-1)*r_d*(1-g_d)*(1-b_d) * lut[id0100] +
    //                (-1)*(1-r_d)*g_d*(1-b_d) * lut[id0010] +
    //                (-1)*(1-r_d)*(1-g_d)*b_d * lut[id0001] +
    //                1*r_d*(1-g_d)*(1-b_d) * lut[id1100] +
    //                (-1)*r_d*g_d*(1-b_d) * lut[id0110] +
    //                (-1)*(1-r_d)*g_d*b_d * lut[id0011] +
    //                1*(1-r_d)*g_d*(1-b_d) * lut[id1010] +
    //                1*(1-r_d)*(1-g_d)*b_d * lut[id1001] +
    //                (-1)*r_d*(1-g_d)*b_d * lut[id0101] +
    //                1*r_d*g_d*(1-b_d) * lut[id1110] +
    //                (-1)*r_d*g_d*b_d * lut[id0111] +
    //                1*r_d*(1-g_d)*b_d * lut[id1101] +
    //                1*(1-r_d)*g_d*b_d * lut[id1011] +
    //                1*r_d*g_d*b_d * lut[id1111]
    //             )
    //          );
    // atomicAdd(image_grad + index, image_grad[index + width * height * batch *
    // 3] *
    //            (   (-1)*(1-r_d)*(1-g_d)*(1-b_d) * lut[id0000] +
    //                1*(1-r_d)*(1-g_d)*(1-b_d) * lut[id1000] +
    //                (-1)*r_d*(1-g_d)*(1-b_d) * lut[id0100] +
    //                (-1)*(1-r_d)*g_d*(1-b_d) * lut[id0010] +
    //                (-1)*(1-r_d)*(1-g_d)*b_d * lut[id0001] +
    //                1*r_d*(1-g_d)*(1-b_d) * lut[id1100] +
    //                (-1)*r_d*g_d*(1-b_d) * lut[id0110] +
    //                (-1)*(1-r_d)*g_d*b_d * lut[id0011] +
    //                1*(1-r_d)*g_d*(1-b_d) * lut[id1010] +
    //                1*(1-r_d)*(1-g_d)*b_d * lut[id1001] +
    //                (-1)*r_d*(1-g_d)*b_d * lut[id0101] +
    //                1*r_d*g_d*(1-b_d) * lut[id1110] +
    //                (-1)*r_d*g_d*b_d * lut[id0111] +
    //                1*r_d*(1-g_d)*b_d * lut[id1101] +
    //                1*(1-r_d)*g_d*b_d * lut[id1011] +
    //                1*r_d*g_d*b_d * lut[id1111]
    //             )
    //          );
    // atomicAdd(image_grad + index, image_grad[index + width * height * batch *
    // 2]); atomicAdd(image_grad + index, image_grad[index + width * height *
    // batch * 3]);

    // float i000 = lut[id1000]-lut[id0000];
    // float i100 = lut[id1100]-lut[id0100];
    // float i010 = lut[id1010]-lut[id0010];
    // float i001 = lut[id1001]-lut[id0001];
    // float i110 = lut[id1110]-lut[id0110];
    // float i011 = lut[id1011]-lut[id0011];
    // float i101 = lut[id1101]-lut[id0101];
    // float i111 = lut[id1111]-lut[id0111];

    float w000 = (1 - r_d) * (1 - g_d) * (1 - b_d);
    float w100 = r_d * (1 - g_d) * (1 - b_d);
    float w010 = (1 - r_d) * g_d * (1 - b_d);
    float w001 = (1 - r_d) * (1 - g_d) * b_d;
    float w110 = r_d * g_d * (1 - b_d);
    float w011 = (1 - r_d) * g_d * b_d;
    float w101 = r_d * (1 - g_d) * b_d;
    float w111 = r_d * g_d * b_d;

    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch] * w000 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch] * w100 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch] * w010 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch] * w001 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch] * w110 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch] * w011 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch] * w101 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch] * w111 * binsize);

    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch * 2] * w000 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch * 2] * w100 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch * 2] * w010 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch * 2] * w001 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch * 2] * w110 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch * 2] * w011 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch * 2] * w101 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch * 2] * w111 * binsize);

    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch * 3] * w000 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch * 3] * w100 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch * 3] * w010 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch * 3] * w001 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch * 3] * w110 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch * 3] * w011 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch * 3] * w101 * binsize);
    atomicAdd(image_grad + index,
              image_grad[index + width * height * batch * 3] * w111 * binsize);
  }
}

int QuadriLinearBackwardLaucher(const float* image, float* image_grad,
                                const float* lut, float* lut_grad,
                                const int lut_dim, const int shift,
                                const float binsize, const int width,
                                const int height, const int batch,
                                cudaStream_t stream) {
  const int kThreadsPerBlock = 1024;
  const int output_size = height * width * batch;
  cudaError_t err;

  QuadriLinearBackward<<<(output_size + kThreadsPerBlock - 1) /
                             kThreadsPerBlock,
                         kThreadsPerBlock, 0, stream>>>(
      output_size, image, image_grad, lut, lut_grad, lut_dim, shift, binsize,
      width, height, batch);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    printf("CUDA kernel failed : %s\n", cudaGetErrorString(err));
    return 0;
  }

  return 1;
}
