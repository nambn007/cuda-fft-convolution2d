#pragma once 

#include <opencv2/opencv.hpp>


void load_image_from_path(const std::string& img_path, float **data, int *height, int *width, int *channels);
void load_image_to_host_buffer(const cv::Mat &img, float **h_buffer, int *height, int *width, int *channels);
void load_kernel_csv(const std::string &file_name, float **kernel, int *height, int *width);

inline 
void pad_image(const float *input, int inH, int inW, float **output, int outH, int outW) {
    *output = (float *)calloc(outH * outW, sizeof(float));
    for (int i = 0; i < inH; ++i) {
        memcpy((*output) + i * outW, input + i * inW, inW * sizeof(float));
    }    
}

inline 
void pad_kernel(const float *kernel, int kerH, int kerW, float **padded_kernel, int outH, int outW) {
    *padded_kernel = (float *)calloc(outH * outW, sizeof(float));
    // int offsetH = (outH - kerH) / 2;
    // int offsetW = (outW - kerW) / 2;
    // for (int i = 0; i < kerH; ++i) {
    //     memcpy((*padded_kernel) + (i + offsetH) * outW + offsetW, kernel + i * kerW, kerW * sizeof(float));
    // } 
    for (int i = 0; i < kerH; ++i) {
        memcpy((*padded_kernel) + i * outW, kernel + i * kerW, kerW * sizeof(float));
    }
}

inline
void crop_result(const float *input, int inH, int inW, int outH, int outW, float **cropped_result) {
    int offsetH = (inH - outH) / 2;
    int offsetW = (inW - outW) / 2;
    *cropped_result = (float *)calloc(outH * outW, sizeof(float));
    for (int i = 0; i < outH; ++i) {
        memcpy((*cropped_result) + i * outW, input + (i + offsetH) * inW + offsetW, outW * sizeof(float));
    }
}

inline
void crop_result(const double *input, int inH, int inW, int outH, int outW, float **cropped_result) {
    int offsetH = (inH - outH) / 2;
    int offsetW = (inW - outW) / 2;
    *cropped_result = (float *)calloc(outH * outW, sizeof(float));
    for (int i = 0; i < outH; ++i) {
        memcpy((*cropped_result) + i * outW, input + (i + offsetH) * inW + offsetW, outW * sizeof(float));
    }
}

inline
void save_image(const char *file_name, const float *data, int height, int width, int channels) {
    cv::Mat out(height, width, CV_8UC1);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float val = data[i * width + j];
            val = std::min(std::max(val, 0.0f), 255.0f);
            out.at<uchar>(i, j) = static_cast<uchar>(val);
        }
    }
    cv::imwrite(file_name, out);
}