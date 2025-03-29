#pragma once 

#include <cufft.h>

void fft_convolve_2d(
    const float *h_image,    // Padding Image (Host)
    const float *h_kernel,   // Padding kernel (Host)
    float **h_result,        // Result (Host)
    int H,                   // Height of padding image
    int W                    // Width of padding image 
);

void check_cufft_status(cufftResult status, const char *msg);

__global__ void fft_pointwise_multiply(cufftComplex *A, cufftComplex *B, cufftComplex *C, int size);

__global__ void normalize(float *data, int size, float factor);