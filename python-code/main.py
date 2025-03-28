import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2

import os
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'  # Disable GUI backend
import cv2
cv2.setNumThreads(0)  # Disable threading for image processing

def fft_convolve2d(image, kernel):
    """
    2D convolution using FFT
    """
    # Get dimensions
    s1 = image.shape
    s2 = kernel.shape
    
    # Calculate full output size
    size = (s1[0] + s2[0] - 1, s1[1] + s2[1] - 1)
    
    # Pad image and kernel to the output size
    fsize = (2 ** np.ceil(np.log2(size)).astype(int)).tolist()
    fslice = tuple([slice(0, s) for s in size])
    
    # Pad arrays with zeros
    new_img = np.zeros(fsize)
    new_img[0:s1[0], 0:s1[1]] = image
    
    new_kernel = np.zeros(fsize)
    new_kernel[0:s2[0], 0:s2[1]] = kernel
    
    # FFT of image and kernel
    img_fft = np.fft.fft2(new_img)
    kernel_fft = np.fft.fft2(new_kernel)
    
    # Multiplication in frequency domain
    result = np.fft.ifft2(img_fft * kernel_fft)
    
    # Extract valid part and convert to real
    result = result[fslice].real
    
    # Crop to valid convolution area
    start0 = (s2[0] - 1) // 2
    start1 = (s2[1] - 1) // 2
    end0 = start0 + s1[0]
    end1 = start1 + s1[1]
    
    result = result[start0:end0, start1:end1]
    
    return result

# Define the kernel from your provided values
kernel = np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1]
], dtype=np.float32)

# Normalize kernel (optional, but often helpful)
kernel = kernel / np.sum(kernel)

# Example: Load an image
# You can replace this with your own image loading code
from PIL import Image
import numpy as np

# Replace the cv2.imread line with:
image = np.array(Image.open('../data/test2.jpg').convert('L'))

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
# For testing, let's create a sample image
# image = np.zeros((100, 100), dtype=np.float32)
# image[40:60, 40:60] = 1.0  # Create a square

# Apply FFT-based convolution
result = fft_convolve2d(image, kernel)

# For comparison, using SciPy's implementation
scipy_result = signal.convolve2d(image, kernel, mode='same')

cv2.imwrite('result.jpg', result)

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(132)
plt.imshow(result, cmap='gray')
plt.title('FFT-based Convolution')

plt.subplot(133)
plt.imshow(scipy_result, cmap='gray')
plt.title('SciPy Convolution')

plt.tight_layout()
plt.show()

# Verify our implementation matches scipy's
print(f"Maximum difference: {np.max(np.abs(result - scipy_result))}")
print(f"Are results close? {np.allclose(result, scipy_result)}")