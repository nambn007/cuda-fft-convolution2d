#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <fftw3.h>
#include <iostream>
#include <chrono>

void check_fftw_status(bool status, const char *msg) {
    if (!status) {
        std::cerr << "FFTW error: " << msg << std::endl;
        exit(EXIT_FAILURE);
    }
}

void fft_convolve_2d(const float *h_image, const float *h_kernel, double **h_result, int H, int W) {
    size_t real_size = sizeof(double) * H * W;

    // Prepare input vectors
    std::vector<float> f_image(h_image, h_image + H * W);
    std::vector<float> f_kernel(h_kernel, h_kernel + H * W);

    std::vector<double> image(f_image.begin(), f_image.end());
    std::vector<double> kernel(f_kernel.begin(), f_kernel.end());
    
    // Allocate output buffer
    *h_result = (double *)malloc(real_size);

    // Compute FFT and convolution
    fftw_complex *fft_image, *fft_kernel, *fft_output;
    fftw_plan forward_plan_image, forward_plan_kernel, inverse_plan;

    // Allocate memory for complex FFT results
    fft_image = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * H * (W/2 + 1));
    fft_kernel = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * H * (W/2 + 1));
    fft_output = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * H * (W/2 + 1));

    auto start = std::chrono::high_resolution_clock::now();
    // Create forward plans for image and kernel
    forward_plan_image = fftw_plan_dft_r2c_2d(H, W, image.data(), fft_image, FFTW_ESTIMATE);
    forward_plan_kernel = fftw_plan_dft_r2c_2d(H, W, kernel.data(), fft_kernel, FFTW_ESTIMATE);

    // Execute forward FFT for image and kernel
    fftw_execute(forward_plan_image);
    fftw_execute(forward_plan_kernel);

    // Pointwise complex multiplication
    for (int i = 0; i < H * (W/2 + 1); ++i) {
        double a = fft_image[i][0];
        double b = fft_image[i][1];
        double c = fft_kernel[i][0];
        double d = fft_kernel[i][1];
        
        // Complex multiplication: (a + bi) * (c + di)
        fft_output[i][0] = a * c - b * d;
        fft_output[i][1] = a * d + b * c;
    }

    // Prepare output buffer for inverse FFT
    std::vector<float> output(H * W);

    // Create inverse plan
    inverse_plan = fftw_plan_dft_c2r_2d(H, W, fft_output, *h_result, FFTW_ESTIMATE);
    
    // Execute inverse FFT
    fftw_execute(inverse_plan);


    // Normalize result
    float norm_factor = 1.0f / (H * W);
    for (int i = 0; i < H * W; ++i) {
        (*h_result)[i] *= norm_factor;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "...CPU-FFT Convolution time: " << elapsed.count() << " ms\n";

    // Clean up
    fftw_destroy_plan(forward_plan_image);
    fftw_destroy_plan(forward_plan_kernel);
    fftw_destroy_plan(inverse_plan);
    
    fftw_free(fft_image);
    fftw_free(fft_kernel);
    fftw_free(fft_output);
}