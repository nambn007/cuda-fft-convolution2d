// Main.cu
#include <iostream>
#include <filesystem>
#include <fmt/format.h>
#include <boost/program_options.hpp>
#include "utils.h"
#include "fft_utils.cuh"

namespace po = boost::program_options;
namespace fs = std::filesystem;


int main(int argc, char** argv) {
    
    std::string usage = fmt::format("Usage: {} parameters\n", argv[0]);
    po::options_description desc(usage);
    desc.add_options()
        ("help", "Print help message")
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


    // Load Kernel Data
    std::string kernel_path = vm["kernel"].as<std::string>();
    float *kernel_h_buffer = nullptr;
    int kernel_h = 0, kernel_w = 0;
    load_kernel_csv(kernel_path, &kernel_h_buffer, &kernel_h, &kernel_w);
    if (kernel_h_buffer == nullptr) {
        std::cerr << "Cannot load kernel from path: " << kernel_path << "\n";
        return 1;
    }

    if (!img_path.empty()) {
        float *image_h_buffer = nullptr;
        int img_w, img_h, img_c;
        load_image_from_path(img_path, &image_h_buffer, &img_h, &img_w, &img_c);
        if (image_h_buffer == nullptr) {
            std::cerr << "Cannot load image from path: " << img_path << "\n";
            return 1;
        }

        // Calculate padding size
        const int pad_h = img_h + kernel_h - 1;        
        const int pad_w = img_w + kernel_w - 1;
        float *h_image_padded = nullptr;
        float *h_kernel_padded = nullptr;

        pad_image(image_h_buffer, img_h, img_w, &h_image_padded, pad_h, pad_w);
        pad_kernel(kernel_h_buffer, kernel_h, kernel_w, &h_kernel_padded, pad_h, pad_w);
        
        float *h_result = nullptr;
        fft_convolve_2d(h_image_padded, h_kernel_padded, &h_result, pad_h, pad_w);
        
        save_image("output_padding.jpg", h_result, pad_h, pad_w, img_c);

        float *h_cropped = nullptr;
        crop_result(h_result, img_h, img_w, pad_h, pad_w, &h_cropped);

        save_image("output.jpg", h_cropped, img_h, img_w, img_c);
    }

    if (!folder_image_path.empty()) {
        if (!fs::exists(folder_image_path) || !fs::is_directory(folder_image_path)) {
            std::cerr << "Folder path is invalid or does not exist: " << folder_image_path << "\n";
            return 1;
        }
        
        std::vector<std::string> image_paths;
        for (const auto& entry : fs::directory_iterator(folder_image_path)) {
            if (entry.is_regular_file()) {
                std::string extension = entry.path().extension().string();
                if (extension == ".jpg" || extension == ".png") {
                    image_paths.push_back(entry.path().string());
                }
            }
        }
    }

    return 0;
}