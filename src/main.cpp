// Main.cu
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <fmt/format.h>
#include <boost/program_options.hpp>
#include "utils.h"
#include "fft.cuh"
#include "fft.h"

namespace po = boost::program_options;
namespace fs = std::filesystem;

float *kernel_h_buffer = nullptr;
int kernel_h;
int kernel_w;


void cuda_fft_convolve_2d_process_image(const std::string &image_path, const std::string &output_folder) {
    float *image_h_buffer = nullptr;
    int img_w, img_h, img_c;
    load_image_from_path(image_path, &image_h_buffer, &img_h, &img_w, &img_c);
    if (image_h_buffer == nullptr) {
        std::cerr << "Cannot load image from path: " << image_path << "\n";
        return;
    }

    const int pad_h = img_h + kernel_h - 1;        
    const int pad_w = img_w + kernel_w - 1;
    float *h_image_padded = nullptr;
    float *h_kernel_padded = nullptr;

    pad_image(image_h_buffer, img_h, img_w, &h_image_padded, pad_h, pad_w);
    pad_kernel(kernel_h_buffer, kernel_h, kernel_w, &h_kernel_padded, pad_h, pad_w);
    
    float *h_result = nullptr;

    auto start = std::chrono::high_resolution_clock::now();
    fft_convolve_2d(h_image_padded, h_kernel_padded, &h_result, pad_h, pad_w);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "CUDA-FFT Convolution time: " << elapsed.count() << " ms\n";

    float *h_cropped = nullptr;
    crop_result(h_result, pad_h, pad_w, img_h, img_w, &h_cropped);

    if (!fs::exists(output_folder)) {
        fs::create_directories(output_folder);
    }
    std::string output_path = fmt::format("{}/{}", output_folder, fs::path(image_path).filename().string());
    std::cout << "... Save image to: " << output_path << "\n";
    save_image(output_path.c_str(), h_cropped, img_h, img_w, img_c);

    if (image_h_buffer) free(image_h_buffer);
    if (h_image_padded) free(h_image_padded);
    if (h_kernel_padded) free(h_kernel_padded);
    if (h_result) free(h_result);
    if (h_cropped) free(h_cropped);
}

void cpu_fft_convolve_2d_process_image(const std::string &image_path, const std::string &output_folder) {
    float *image_h_buffer = nullptr;
    int img_w, img_h, img_c;
    load_image_from_path(image_path, &image_h_buffer, &img_h, &img_w, &img_c);
    if (image_h_buffer == nullptr) {
        std::cerr << "Cannot load image from path: " << image_path << "\n";
        return;
    }

    const int pad_h = img_h + kernel_h - 1;        
    const int pad_w = img_w + kernel_w - 1;
    float *h_image_padded = nullptr;
    float *h_kernel_padded = nullptr;

    pad_image(image_h_buffer, img_h, img_w, &h_image_padded, pad_h, pad_w);
    pad_kernel(kernel_h_buffer, kernel_h, kernel_w, &h_kernel_padded, pad_h, pad_w);
    
    double *h_result = nullptr;

    auto start = std::chrono::high_resolution_clock::now();
    fft_convolve_2d(h_image_padded, h_kernel_padded, &h_result, pad_h, pad_w);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "CPU-FFT Convolution time: " << elapsed.count() << " ms\n";

    float *h_cropped = nullptr;
    crop_result(h_result, pad_h, pad_w, img_h, img_w, &h_cropped);

    if (!fs::exists(output_folder)) {
        fs::create_directories(output_folder);
    }
    std::string output_path = fmt::format("{}/{}", output_folder, fs::path(image_path).filename().string());
    std::cout << "... Save image to: " << output_path << "\n";
    save_image(output_path.c_str(), h_cropped, img_h, img_w, img_c);

    if (image_h_buffer) free(image_h_buffer);
    if (h_image_padded) free(h_image_padded);
    if (h_kernel_padded) free(h_kernel_padded);
    if (h_result) free(h_result);
    if (h_cropped) free(h_cropped);
}

void fft_convolve_2d_process_images(const std::vector<std::string> &image_paths, 
                                    const std::string &output_folder,
                                    bool use_cuda = true) {
    for (const auto &img_path : image_paths) {
        std::cout << "Start process image: " << img_path << "\n";
        if (use_cuda) {
            cuda_fft_convolve_2d_process_image(img_path, output_folder);
        } else {
            cpu_fft_convolve_2d_process_image(img_path, output_folder);
        }
        std::cout << "Finish process image: " << img_path << "\n";
        std::cout << "----------------------------------------\n";
    }
}

int main(int argc, char** argv) {
    
    std::string usage = fmt::format("Usage: {} parameters\n", argv[0]);
    po::options_description desc(usage);
    desc.add_options()
        ("help", "Print help message")
        ("cuda,-c", po::value<bool>()->default_value(true), "Use CUDA")
        ("image,-f", po::value<std::string>(), "Path to the image")
        ("folder,-d", po::value<std::string>(), "Path to folder image")
        ("kernel,-k", po::value<std::string>(), "Path to the kernel")
        ("output,-o", po::value<std::string>()->default_value("./outputs"), "Paht to folder output");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
    } catch (std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    } 

    po::notify(vm);
    if (vm.count("help") || (vm.count("image") == 0 && vm.count("folder") == 0) || vm.count("kernel") == 0) {
        std::cout << desc << "\n";
        return 0;
    }

    std::string img_path;
    if (vm.count("image")) {
        img_path = vm["image"].as<std::string>();
    }

    std::string folder_image_path;
    if (vm.count("folder")) {
        folder_image_path = vm["folder"].as<std::string>();
    }

    if (img_path.empty() && folder_image_path.empty()) {
        std::cout << desc << "\n";
        return -1;
    }

    std::string folder_output_path = vm["output"].as<std::string>();
    bool use_cuda = vm["cuda"].as<bool>();

    // Load Kernel Data
    std::string kernel_path = vm["kernel"].as<std::string>();
    load_kernel_csv(kernel_path, &kernel_h_buffer, &kernel_h, &kernel_w);
    if (kernel_h_buffer == nullptr) {
        std::cerr << "Cannot load kernel from path: " << kernel_path << "\n";
        return 1;
    }

    float sum = 0;
    for (int i = 0; i < kernel_h * kernel_w; i++) {
        sum += kernel_h_buffer[i];
    }
    for (int i = 0; i < kernel_h * kernel_w; i++) {
        kernel_h_buffer[i] /= sum;
    }

    // Load images
    std::vector<std::string> image_paths;
    if (!img_path.empty()) {
        image_paths.push_back(img_path);
    }

    if (!folder_image_path.empty()) {
        if (!fs::exists(folder_image_path) || !fs::is_directory(folder_image_path)) {
            std::cerr << "Folder path is invalid or does not exist: " << folder_image_path << "\n";
            return 1;
        }
        for (const auto& entry : fs::directory_iterator(folder_image_path)) {
            if (entry.is_regular_file()) {
                std::string extension = entry.path().extension().string();
                if (extension == ".jpg" || extension == ".png") {
                    image_paths.push_back(entry.path().string());
                }
            }
        }
    }

    fft_convolve_2d_process_images(image_paths, folder_output_path, use_cuda);

    if (kernel_h_buffer != nullptr) {
        free(kernel_h_buffer);
        kernel_h_buffer = nullptr;
    }

    return 0;
}