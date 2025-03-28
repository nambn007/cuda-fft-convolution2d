#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>

// Helper function to check cuFFT errors
void check(cufftResult status, const char* msg) {
    if (status != CUFFT_SUCCESS) {
        std::cerr << "cuFFT error: " << msg << " (code: " << status << ")\n";
        exit(EXIT_FAILURE);
    }
}

// Helper function to check if arrays are close (similar to np.allclose)
bool allclose(float* arr1, float* arr2, int size, float rtol = 1e-5, float atol = 1e-8) {
    for (int i = 0; i < size; i++) {
        float diff = std::abs(arr1[i] - arr2[i]);
        float tolerance = atol + rtol * std::abs(arr2[i]);
        if (diff > tolerance) return false;
    }
    return true;
}

int main() {
    // Create the same 3x3 image as in Python
    const int H = 3;
    const int W = 3;
    float h_image[H * W] = {
        255.0f, 255.0f, 255.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    
    // Print original image
    std::cout << "Original image:" << std::endl;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            std::cout << h_image[i * W + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Allocate memory
    size_t real_size = sizeof(float) * H * W;
    size_t complex_size = sizeof(cufftComplex) * H * (W/2 + 1);
    
    float *d_image, *d_ifft_result;
    cufftComplex *d_fft_result;
    
    cudaMalloc(&d_image, real_size);
    cudaMalloc(&d_fft_result, complex_size);
    cudaMalloc(&d_ifft_result, real_size);
    
    // Copy data to device
    cudaMemcpy(d_image, h_image, real_size, cudaMemcpyHostToDevice);
    
    // Create FFT plans
    cufftHandle forward_plan, inverse_plan;
    check(cufftPlan2d(&forward_plan, H, W, CUFFT_R2C), "Forward FFT Plan");
    check(cufftPlan2d(&inverse_plan, H, W, CUFFT_C2R), "Inverse FFT Plan");
    
    // Execute forward FFT
    check(cufftExecR2C(forward_plan, d_image, d_fft_result), "Forward FFT");
    
    // Execute inverse FFT
    check(cufftExecC2R(inverse_plan, d_fft_result, d_ifft_result), "Inverse FFT");
    
    // Copy results back to host
    float *h_ifft_result = new float[H * W];
    cudaMemcpy(h_ifft_result, d_ifft_result, real_size, cudaMemcpyDeviceToHost);
    
    // Normalize the result (cuFFT doesn't normalize automatically)
    float norm_factor = 1.0f / (H * W);
    for (int i = 0; i < H * W; i++) {
        h_ifft_result[i] *= norm_factor;
    }
    
    // Print inverse FFT result
    std::cout << "\nInverse FFT result:" << std::endl;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            std::cout << std::fixed << std::setprecision(6) << h_ifft_result[i * W + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Print difference
    std::cout << "\nDifference:" << std::endl;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            std::cout << std::fixed << std::setprecision(10) 
                     << h_image[i * W + j] - h_ifft_result[i * W + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Check if arrays are close
    bool are_close = allclose(h_image, h_ifft_result, H * W);
    std::cout << "\nAre they close? " << (are_close ? "true" : "false") << std::endl;
    
    // Clean up
    delete[] h_ifft_result;
    cufftDestroy(forward_plan);
    cufftDestroy(inverse_plan);
    cudaFree(d_image);
    cudaFree(d_fft_result);
    cudaFree(d_ifft_result);
    
    return 0;
}