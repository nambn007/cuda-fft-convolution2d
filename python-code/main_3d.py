import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

# Load ảnh grayscale bằng OpenCV
image = cv2.imread('../data/test2.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
print("Image shape:", image.shape)

# Tạo kernel làm mịn (5x5 Gaussian blur nhẹ)
kernel = np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1]
], dtype=np.float32)
kernel /= kernel.sum()

# Padding ảnh và kernel
pad_h = image.shape[0] + kernel.shape[0] - 1
pad_w = image.shape[1] + kernel.shape[1] - 1

image_padded = np.zeros((pad_h, pad_w), dtype=np.float32)
kernel_padded = np.zeros((pad_h, pad_w), dtype=np.float32)

image_padded[:image.shape[0], :image.shape[1]] = image
kernel_padded[:kernel.shape[0], :kernel.shape[1]] = kernel

# FFT
fft_img = np.fft.fft2(image_padded)
fft_ker = np.fft.fft2(kernel_padded)
fft_result = fft_img * fft_ker
conv_result = np.fft.ifft2(fft_result).real

# Crop lại kết quả (same size as original image)
start_h = (kernel.shape[0] - 1) // 2
start_w = (kernel.shape[1] - 1) // 2
final = conv_result[start_h:start_h + image.shape[0], start_w:start_w + image.shape[1]]

# Hiển thị kết quả
cv2.imwrite("output/result_fft_blur.png", np.clip(final, 0, 255).astype(np.uint8))
