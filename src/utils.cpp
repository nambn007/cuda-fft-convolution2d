#include "utils.h"
#include <fstream>

void load_image_from_path(const std::string& img_path, float **data, int *height, int *width, int *channels) {
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    if (img.empty()) {
        std::cerr << "Cannot read image: " << img_path << "\n";
        return;
    }   
    load_image_to_host_buffer(img, data, height, width, channels);
}

void load_image_to_host_buffer(const cv::Mat &img, float **h_buffer, int *height, int *width, int *channels) {
    const size_t data_size = sizeof(float) * img.rows * img.cols * img.channels();
    (*h_buffer) = (float *)malloc(data_size);
    if (h_buffer == nullptr) {
        std::cerr << "Cannot allocate memory for image buffer\n";
        return;
    }

    *height = img.rows;
    *width = img.cols;
    *channels = img.channels();

    // Image from P3C to C3C
    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(row, col);
            size_t channel_size = img.rows * img.cols;
            size_t pitch_size = img.cols; 

            (*h_buffer)[0 * channel_size + row * pitch_size + col] = float(pixel[0]); // R 
            (*h_buffer)[1 * channel_size + row * pitch_size + col] = float(pixel[1]); // G
            (*h_buffer)[2 * channel_size + row * pitch_size + col] = float(pixel[2]); // B
        }
    }
}

void load_kernel_csv(const std::string &file_name, float **kernel, int *height, int *width) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Cannot open kernel file: " << file_name << "\n";
        return;
    }

    std::vector<std::vector<float>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }
        data.push_back(row);
    }

    *height = data.size();
    *width = data[0].size();
    *kernel = (float *)malloc(sizeof(float) * (*height) * (*width));

    for (int i = 0; i < *height; i++) {
        for (int j = 0; j < *width; j++) {
            (*kernel)[i * (*width) + j] = data[i][j];
        }
    }
}
