#include "quadrilinear.h"


void QuadriLinearForwardCpu(const float* lut, const float* image, float* output, const int dim, const int shift, const float binsize, const int width, const int height, const int channels);

void QuadriLinearBackwardCpu(const float* image, float* image_grad,const float* lut, float* lut_grad, const int dim, const int shift, const float binsize, const int width, const int height, const int channels);

int quadrilinear_forward(torch::Tensor lut, torch::Tensor image, torch::Tensor output,
                      int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    // Grab the input tensor
    float * lut_flat = lut.data<float>();
    float * image_flat = image.data<float>();
    float * output_flat = output.data<float>();

    // whether color image
    auto image_size = image.sizes();
    int channels = image_size[1];
    
    QuadriLinearForwardCpu(lut_flat, image_flat, output_flat, lut_dim, shift, binsize, width, height, channels);

    return 1;
}

int quadrilinear_backward(torch::Tensor image, torch::Tensor image_grad, torch::Tensor lut, torch::Tensor lut_grad,
                       int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    // Grab the input tensor
    float * image_grad_flat = image_grad.data<float>();
    float * lut_flat = lut.data<float>();
    float * image_flat = image.data<float>();
    float * lut_grad_flat = lut_grad.data<float>();

    // whether color image
    auto image_size = image.sizes();
    int channels = image_size[1];
    if (channels != 3)
    {
        return 0;
    }

    TriLinearBackwardCpu(image_flat, image_grad_flat,lut_flat, lut_grad_flat, lut_dim, shift, binsize, width, height, channels);

    return 1;
}

void QuadriLinearForwardCpu(const float* lut, const float* image, float* output, const int dim, const int shift, const float binsize, const int width, const int height, const int channels)
{
    const int output_size = height * width;;

    int index = 0;

    #pragma omp parallel
    #pragma omp for
    for (index = 0; index < output_size; ++index)
    {
    float context = image[index];
    float r = image[index + width * height];
    float g = image[index + width * height * 2];
    float b = image[index + width * height * 3];

    int r_id = floor(r / binsize);
    int g_id = floor(g / binsize);
    int b_id = floor(b / binsize);
    int context_id = floor(context / binsize);

    float r_d = fmod(r,binsize) / binsize;
    float g_d = fmod(g,binsize) / binsize;
    float b_d = fmod(b,binsize) / binsize;
    float context_d = fmod(context,binsize) / binsize;

    int id0000 = context_id + r_id * dim + g_id * dim * dim + b_id * dim * dim * dim;
    int id1000 = context_id + 1 + r_id * dim + g_id * dim * dim + b_id * dim * dim * dim;
    int id0100 = context_id + (r_id + 1) * dim + g_id * dim * dim + b_id * dim * dim * dim;
    int id0010 = context_id + r_id * dim + (g_id + 1) * dim * dim + b_id * dim * dim * dim;
    int id0001 = context_id + r_id * dim + g_id * dim * dim + (b_id + 1) * dim * dim * dim;
    int id1100 = context_id + 1 + (r_id + 1) * dim + g_id * dim * dim + b_id * dim * dim * dim;
    int id0110 = context_id + (r_id + 1) * dim + (g_id + 1) * dim * dim + b_id * dim * dim * dim;
    int id0011 = context_id + r_id * dim + (g_id + 1) * dim * dim + (b_id + 1) * dim * dim * dim;
    int id1010 = context_id + 1 + r_id * dim + (g_id + 1) * dim * dim + b_id * dim * dim * dim;
    int id1001 = context_id + 1 + r_id * dim + g_id * dim * dim + (b_id + 1) * dim * dim * dim;
    int id0101 = context_id + (r_id + 1) * dim + g_id * dim * dim + (b_id + 1) * dim * dim * dim;
    int id1110 = context_id + 1 + (r_id + 1) * dim + (g_id + 1) * dim * dim + b_id * dim * dim * dim;
    int id1011 = context_id + 1 + r_id * dim + (g_id + 1) * dim * dim + (b_id + 1) * dim * dim * dim;
    int id1101 = context_id + 1 + (r_id + 1) * dim + g_id * dim * dim + (b_id + 1) * dim * dim * dim;
    int id0111 = context_id + (r_id + 1) * dim + (g_id + 1) * dim * dim + (b_id + 1) * dim * dim * dim;
    int id1111 = context_id + 1 + (r_id + 1) * dim + (g_id + 1) * dim * dim + (b_id + 1) * dim * dim * dim;


    float w0000 = (1-context_d)*(1-r_d)*(1-g_d)*(1-b_d);
    float w1000 = context_d*(1-r_d)*(1-g_d)*(1-b_d);
    float w0100 = (1-context_d)*r_d*(1-g_d)*(1-b_d);
    float w0010 = (1-context_d)*(1-r_d)*g_d*(1-b_d);
    float w0001 = (1-context_d)*(1-r_d)*(1-g_d)*b_d;
    float w1100 = context_d*r_d*(1-g_d)*(1-b_d);
    float w0110 = (1-context_d)*r_d*g_d*(1-b_d);
    float w0011 = (1-context_d)*(1-r_d)*g_d*b_d;
    float w1010 = context_d*(1-r_d)*g_d*(1-b_d);
    float w1001 = context_d*(1-r_d)*(1-g_d)*b_d;
    float w0101 = (1-context_d)*r_d*(1-g_d)*b_d;
    float w1110 = context_d*r_d*g_d*(1-b_d);
    float w0111 = (1-context_d)*r_d*g_d*b_d;
    float w1101 = context_d*r_d*(1-g_d)*b_d;
    float w1011 = context_d*(1-r_d)*g_d*b_d;
    float w1111 = context_d*r_d*g_d*b_d;

    output[index] = w0000 * lut[id0000] + w1000 * lut[id1000] + w0100 * lut[id0100] + w0010 * lut[id0010] + 
                    w0001 * lut[id0001] + w1100 * lut[id1100] + w0110 * lut[id0110] + w0011 * lut[id0011] + 
                    w1010 * lut[id1010] + w1001 * lut[id1001] + w0101 * lut[id0101] + w1110 * lut[id1110] + 
                    w0111 * lut[id0111] + w1101 * lut[id1101] + w1011 * lut[id1011] + w1111 * lut[id1111];

    output[index + width * height] = w0000 * lut[id0000 + shift] + w1000 * lut[id1000 + shift] + w0100 * lut[id0100 + shift] + w0010 * lut[id0010 + shift] + 
                                             w0001 * lut[id0001 + shift] + w1100 * lut[id1100 + shift] + w0110 * lut[id0110 + shift] + w0011 * lut[id0011 + shift] + 
                                             w1010 * lut[id1010 + shift] + w1001 * lut[id1001 + shift] + w0101 * lut[id0101 + shift] + w1110 * lut[id1110 + shift] + 
                                             w0111 * lut[id0111 + shift] + w1101 * lut[id1101 + shift] + w1011 * lut[id1011 + shift] + w1111 * lut[id1111 + shift];

    output[index + width * height * 2] = w0000 * lut[id0000 + shift * 2] + w1000 * lut[id1000 + shift * 2] + w0100 * lut[id0100 + shift * 2] + w0010 * lut[id0010 + shift * 2] + 
                                                 w0001 * lut[id0001 + shift * 2] + w1100 * lut[id1100 + shift * 2] + w0110 * lut[id0110 + shift * 2] + w0011 * lut[id0011 + shift * 2] + 
                                                 w1010 * lut[id1010 + shift * 2] + w1001 * lut[id1001 + shift * 2] + w0101 * lut[id0101 + shift * 2] + w1110 * lut[id1110 + shift * 2] + 
                                                 w0111 * lut[id0111 + shift * 2] + w1101 * lut[id1101 + shift * 2] + w1011 * lut[id1011 + shift * 2] + w1111 * lut[id1111 + shift * 2];
    }
}

void QuadriLinearBackwardCpu(const float* image, float* image_grad, const float* lut, float* lut_grad, const int dim, const int shift, const float binsize, const int width, const int height, const int channels)
{
    const int output_size = height * width;

    int index = 0;
    #pragma omp parallel
    #pragma omp for
    for (index = 0; index < output_size; ++index)
    {
    float context = image[index];
    float r = image[index + width * height];
    float g = image[index + width * height * 2];
    float b = image[index + width * height * 3];

    int r_id = floor(r / binsize);
    int g_id = floor(g / binsize);
    int b_id = floor(b / binsize);
    int context_id = floor(context / binsize);
    
    float r_d = fmod(r,binsize) / binsize;
    float g_d = fmod(g,binsize) / binsize;
    float b_d = fmod(b,binsize) / binsize;
    float context_d = fmod(context,binsize) / binsize;
    
    int id0000 = context_id + r_id * dim + g_id * dim * dim + b_id * dim * dim * dim;
    int id1000 = context_id + 1 + r_id * dim + g_id * dim * dim + b_id * dim * dim * dim;
    int id0100 = context_id + (r_id + 1) * dim + g_id * dim * dim + b_id * dim * dim * dim;
    int id0010 = context_id + r_id * dim + (g_id + 1) * dim * dim + b_id * dim * dim * dim;
    int id0001 = context_id + r_id * dim + g_id * dim * dim + (b_id + 1) * dim * dim * dim;
    int id1100 = context_id + 1 + (r_id + 1) * dim + g_id * dim * dim + b_id * dim * dim * dim;
    int id0110 = context_id + (r_id + 1) * dim + (g_id + 1) * dim * dim + b_id * dim * dim * dim;
    int id0011 = context_id + r_id * dim + (g_id + 1) * dim * dim + (b_id + 1) * dim * dim * dim;
    int id1010 = context_id + 1 + r_id * dim + (g_id + 1) * dim * dim + b_id * dim * dim * dim;
    int id1001 = context_id + 1 + r_id * dim + g_id * dim * dim + (b_id + 1) * dim * dim * dim;
    int id0101 = context_id + (r_id + 1) * dim + g_id * dim * dim + (b_id + 1) * dim * dim * dim;
    int id1110 = context_id + 1 + (r_id + 1) * dim + (g_id + 1) * dim * dim + b_id * dim * dim * dim;
    int id1011 = context_id + 1 + r_id * dim + (g_id + 1) * dim * dim + (b_id + 1) * dim * dim * dim;
    int id1101 = context_id + 1 + (r_id + 1) * dim + g_id * dim * dim + (b_id + 1) * dim * dim * dim;
    int id0111 = context_id + (r_id + 1) * dim + (g_id + 1) * dim * dim + (b_id + 1) * dim * dim * dim;
    int id1111 = context_id + 1 + (r_id + 1) * dim + (g_id + 1) * dim * dim + (b_id + 1) * dim * dim * dim;


    float w0000 = (1-context_d)*(1-r_d)*(1-g_d)*(1-b_d);
    float w1000 = context_d*(1-r_d)*(1-g_d)*(1-b_d);
    float w0100 = (1-context_d)*r_d*(1-g_d)*(1-b_d);
    float w0010 = (1-context_d)*(1-r_d)*g_d*(1-b_d);
    float w0001 = (1-context_d)*(1-r_d)*(1-g_d)*b_d;
    float w1100 = context_d*r_d*(1-g_d)*(1-b_d);
    float w0110 = (1-context_d)*r_d*g_d*(1-b_d);
    float w0011 = (1-context_d)*(1-r_d)*g_d*b_d;
    float w1010 = context_d*(1-r_d)*g_d*(1-b_d);
    float w1001 = context_d*(1-r_d)*(1-g_d)*b_d;
    float w0101 = (1-context_d)*r_d*(1-g_d)*b_d;
    float w1110 = context_d*r_d*g_d*(1-b_d);
    float w0111 = (1-context_d)*r_d*g_d*b_d;
    float w1101 = context_d*r_d*(1-g_d)*b_d;
    float w1011 = context_d*(1-r_d)*g_d*b_d;
    float w1111 = context_d*r_d*g_d*b_d;

    
    lut_grad[id0000 ] += w0000 * image_grad[index + width * height];
    lut_grad[id1000 ] += w1000 * image_grad[index + width * height];
    lut_grad[id0100 ] += w0100 * image_grad[index + width * height];
    lut_grad[id0010 ] += w0010 * image_grad[index + width * height];
    lut_grad[id0001 ] += w0001 * image_grad[index + width * height];
    lut_grad[id1100 ] += w1100 * image_grad[index + width * height];
    lut_grad[id0110 ] += w0110 * image_grad[index + width * height];
    lut_grad[id0011 ] += w0011 * image_grad[index + width * height];
    lut_grad[id1010 ] += w1010 * image_grad[index + width * height];
    lut_grad[id1001 ] += w1001 * image_grad[index + width * height];
    lut_grad[id0101 ] += w0101 * image_grad[index + width * height];
    lut_grad[id1110 ] += w1110 * image_grad[index + width * height];
    lut_grad[id0111 ] += w0111 * image_grad[index + width * height];
    lut_grad[id1101 ] += w1101 * image_grad[index + width * height];
    lut_grad[id1011 ] += w1011 * image_grad[index + width * height];
    lut_grad[id1111 ] += w1111 * image_grad[index + width * height];

    lut_grad[id0000 + shift] += w0000 * image_grad[index + width * height * 2];
    lut_grad[id1000 + shift] += w1000 * image_grad[index + width * height * 2];
    lut_grad[id0100 + shift] += w0100 * image_grad[index + width * height * 2];
    lut_grad[id0010 + shift] += w0010 * image_grad[index + width * height * 2];
    lut_grad[id0001 + shift] += w0001 * image_grad[index + width * height * 2];
    lut_grad[id1100 + shift] += w1100 * image_grad[index + width * height * 2];
    lut_grad[id0110 + shift] += w0110 * image_grad[index + width * height * 2];
    lut_grad[id0011 + shift] += w0011 * image_grad[index + width * height * 2];
    lut_grad[id1010 + shift] += w1010 * image_grad[index + width * height * 2];
    lut_grad[id1001 + shift] += w1001 * image_grad[index + width * height * 2];
    lut_grad[id0101 + shift] += w0101 * image_grad[index + width * height * 2];
    lut_grad[id1110 + shift] += w1110 * image_grad[index + width * height * 2];
    lut_grad[id0111 + shift] += w0111 * image_grad[index + width * height * 2];
    lut_grad[id1101 + shift] += w1101 * image_grad[index + width * height * 2];
    lut_grad[id1011 + shift] += w1011 * image_grad[index + width * height * 2];
    lut_grad[id1111 + shift] += w1111 * image_grad[index + width * height * 2];

    lut_grad[id0000 + shift* 3] += w0000 * image_grad[index + width * height * 3];
    lut_grad[id1000 + shift* 3] += w1000 * image_grad[index + width * height * 3];
    lut_grad[id0100 + shift* 3] += w0100 * image_grad[index + width * height * 3];
    lut_grad[id0010 + shift* 3] += w0010 * image_grad[index + width * height * 3];
    lut_grad[id0001 + shift* 3] += w0001 * image_grad[index + width * height * 3];
    lut_grad[id1100 + shift* 3] += w1100 * image_grad[index + width * height * 3];
    lut_grad[id0110 + shift* 3] += w0110 * image_grad[index + width * height * 3];
    lut_grad[id0011 + shift* 3] += w0011 * image_grad[index + width * height * 3];
    lut_grad[id1010 + shift* 3] += w1010 * image_grad[index + width * height * 3];
    lut_grad[id1001 + shift* 3] += w1001 * image_grad[index + width * height * 3];
    lut_grad[id0101 + shift* 3] += w0101 * image_grad[index + width * height * 3];
    lut_grad[id1110 + shift* 3] += w1110 * image_grad[index + width * height * 3];
    lut_grad[id0111 + shift* 3] += w0111 * image_grad[index + width * height * 3];
    lut_grad[id1101 + shift* 3] += w1101 * image_grad[index + width * height * 3];
    lut_grad[id1011 + shift* 3] += w1011 * image_grad[index + width * height * 3];
    lut_grad[id1111 + shift* 3] += w1111 * image_grad[index + width * height * 3];


    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &quadrilinear_forward, "Quadrilinear forward");
  m.def("backward", &quadrilinear_backward, "Quadrilinear backward");
}
