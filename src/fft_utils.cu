#include "fft_utils.cuh"
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include "utils.h"

void check_cufft_status(cufftResult status, const char *msg) {
    if (status != CUFFT_SUCCESS) {
        std::cerr << "cuFFT error: " << msg << " (code: " << status << ")\n";
        exit(EXIT_FAILURE);
    }
}

void fft_convolve_2d(const float *h_image, const float *h_kernel, float **h_result, int H, int W) {
    size_t real_size = sizeof(float) * H * W;
    size_t complex_size = sizeof(cufftComplex) * H * (W / 2 + 1);

    // Device 
    float *d_image, *d_kernel, *d_output;
    cufftComplex *d_image_fft, *d_kernel_fft, *d_output_fft;

    cudaMalloc(&d_image, real_size);
    cudaMalloc(&d_kernel, real_size);
    cudaMalloc(&d_output, real_size);

    cudaMalloc(&d_image_fft, complex_size);
    cudaMalloc(&d_kernel_fft, complex_size);
    cudaMalloc(&d_output_fft, complex_size);

    cudaMemcpy(d_image, h_image, real_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, real_size, cudaMemcpyHostToDevice);

    // cuFFT plan
    cufftHandle forward_plan, inverse_plan;
    check_cufft_status(cufftPlan2d(&forward_plan, H, W, CUFFT_R2C), "cufftPlan2d R2C");
    check_cufft_status(cufftPlan2d(&inverse_plan, H, W, CUFFT_C2R), "cufftPlan2d C2R");

    // FFT image and kernel 
    check_cufft_status(cufftExecR2C(forward_plan, d_image, d_image_fft), "cufftExecR2C image");
    check_cufft_status(cufftExecR2C(forward_plan, d_kernel, d_kernel_fft), "cufftExecR2C kernel");

    int num_elements = H * (W / 2 + 1);
    dim3 block(256);
    dim3 grid((num_elements + block.x - 1) / block.x);
    // TODO : Implement the element-wise multiplication and division
    fft_pointwise_multiply<<<grid, block>>>(d_image_fft, d_kernel_fft, d_output_fft, num_elements);

    // Inverse FFT
    check_cufft_status(cufftExecC2R(inverse_plan, d_output_fft, d_output), "cufftExecC2R");

    // Normalize the result
    float norm_factor = 1.0f / (H * W);
    int real_num_elements = H * W;
    dim3 real_grid((real_num_elements + block.x - 1) / block.x);
    normalize<<<real_grid, block>>>(d_output, real_num_elements, norm_factor);

    // Copy the result back to host
    *h_result = (float *)malloc(real_size);
    cudaMemcpy(*h_result, d_output, real_size, cudaMemcpyDeviceToHost);

    // Clean up 
    cufftDestroy(forward_plan); cufftDestroy(inverse_plan);
    cudaFree(d_image); cudaFree(d_kernel); cudaFree(d_output);
    cudaFree(d_image_fft); cudaFree(d_kernel_fft); cudaFree(d_output_fft);
}

__global__ void fft_pointwise_multiply(cufftComplex *A, cufftComplex *B, cufftComplex *C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float a = A[idx].x;
        float b = A[idx].y;
        float c = B[idx].x;
        float d = B[idx].y;
        
        // z1 = a + b * i   
        // z2 = c + d * i 
        // z3 = (a * c - b * d) + (a * d + b * c) * i 
        C[idx].x = a * c - b * d;
        C[idx].y = a * d + b * c;
    }    
}

__global__ void normalize(float *data, int size, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= factor;
    }
}