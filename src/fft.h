#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <fftw3.h>
#include <iostream>

void check_fftw_status(bool status, const char *msg);
void fft_convolve_2d(const float *h_image, const float *h_kernel, double **h_result, int H, int W);

