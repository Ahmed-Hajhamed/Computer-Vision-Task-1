from numpy.lib.stride_tricks import as_strided
import cv2




def add_uniform_noise( image, intensity):
    noise = np.random.uniform(-intensity * 255, intensity * 255, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

def add_gaussian_noise( image, sigma):
    # sigma = intensity * 255
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

def add_salt_pepper_noise( image, prob=0.05):
    noisy_image = np.copy(image)
    salt_mask = np.random.rand(*image.shape) < prob / 2
    pepper_mask = np.random.rand(*image.shape) < prob / 2
    noisy_image[salt_mask] = 255
    noisy_image[pepper_mask] = 0
    return noisy_image


def convolve1(image, kernel):
    """
    Manually applies a convolutional filter to an image.

    :param image: Input image as a 2D NumPy array (grayscale).
    :param kernel: Kernel (filter) as a 2D NumPy array.
    :return: Filtered image as a 2D NumPy array.
    """
    # Get dimensions
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Compute padding size
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    # Pad the image with zeros
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # Create output array
    output = np.zeros((image_height, image_width))

    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            # Extract region of interest
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            # Element-wise multiplication and summation
            output[i, j] = np.sum(region * kernel)

    return output


def apply_filter(image, kernel):
    """
    Applies a convolutional filter to an image.

    :param image: Input image as a NumPy array.
    :param kernel: Kernel (filter) as a 2D NumPy array.
    :return: Filtered image as a NumPy array.
    """
    # if np.sum(kernel) != 0:  # Normalize only if sum is nonzero
    #     kernel = kernel / np.sum(kernel)

    if image.ndim == 3 and image.shape[2] == 3:  # Ensure exactly 3 channels (RGB)
        print("Processing RGB image with shape:", image.shape)
        filtered_image = np.zeros_like(image)
        for i in range(3):  # Process each channel separately
            filtered_image[:, :, i] = convolve1(image[:, :, i], kernel)

    # elif  image.ndim == 3 and image.shape[2] == 3:  # Handle grayscale with extra channel
    #     print("Processing single-channel grayscale image.")
    #     filtered_image = convolve(image[:, :, 0], kernel)
    #     filtered_image = filtered_image[:, :, np.newaxis]  # Keep the same shape

    elif image.ndim == 2:
        print("Processing standard grayscale image with shape:", image.shape)
        filtered_image = convolve1(image, kernel)

    return filtered_image

def average_filter(image, kernel_size = 3):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    return apply_filter(image, kernel).astype(np.uint8)

def gaussian_filter( image, kernel_size=3, sigma=1):
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    kernel /= np.sum(kernel)
    return apply_filter(image, kernel).astype(np.uint8)


def median_filter(image, kernel_size=3):
    """
    Applies a median filter to a grayscale or RGB image.

    :param image: Input image as a NumPy array (H, W) for grayscale or (H, W, 3) for RGB.
    :param kernel_size: Size of the median filter kernel (must be odd).
    :return: Filtered image as a NumPy array.
    """
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number!")

    k_size = kernel_size // 2  # Compute padding size

    # Handle both grayscale and RGB images
    if len(image.shape) == 3:  # RGB Image
        filtered_image = np.zeros_like(image)
        for c in range(image.shape[2]):  # Apply separately for each channel
            filtered_image[:, :, c] = apply_median(image[:, :, c], kernel_size, k_size)
    else:  # Grayscale Image
        filtered_image = apply_median(image, kernel_size, k_size)

    return filtered_image.astype(np.uint8)


def apply_median(image, kernel_size, k_size):
    """
    Helper function to apply a median filter to a single-channel image.
    """
    padded_image = np.pad(image, k_size, mode='reflect')  # Reflective padding

    # Use as_strided to extract sliding windows
    window_shape = (image.shape[0], image.shape[1], kernel_size, kernel_size)
    strides = (padded_image.strides[0], padded_image.strides[1], padded_image.strides[0], padded_image.strides[1])
    windows = as_strided(padded_image, shape=window_shape, strides=strides, writeable=False)

    # Compute median over the kernel window
    return np.median(windows, axis=(2, 3))

def sobel_edge_detection(image):
    # img = image.astype('float64')
    image = convert_to_grayscale(image)[0]
    image = gaussian_filter(image, 3)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    grad_x = apply_filter(image, sobel_x)
    grad_y = apply_filter(image, sobel_y)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude * (255.0 / magnitude.max())

    # Apply thresholding
    threshold = magnitude.mean() * 0.5
    magnitude[magnitude < threshold] = 0
    return np.clip(magnitude, 0, 255).astype(np.uint8)

def roberts_edge_detection(image):
    # img = image.astype('float64')
    image = convert_to_grayscale(image)[0]
    image = gaussian_filter(image, 3)
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    padded_img = np.pad(image, ((0, 1), (0, 1)), mode='edge')
    grad_x = apply_filter(padded_img, kernel_x)
    grad_y = apply_filter(padded_img, kernel_y)
    # Calculate magnitude and normalize
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude[:-1, :-1]  # Remove padding
    magnitude = magnitude * (255.0 / magnitude.max())

    # Apply thresholding
    threshold = magnitude.mean() * 0.5
    magnitude[magnitude < threshold] = 0
    return np.clip(magnitude, 0, 255).astype(np.uint8)

def prewitt_edge_detection(image):
    # img = image.astype('float64')
    image = convert_to_grayscale(image)[0]
    image = gaussian_filter(image, 3)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    grad_x = apply_filter(image, kernel_x)
    grad_y = apply_filter(image, kernel_y)
    # Calculate magnitude and normalize
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude * (255.0 / magnitude.max())

    # Apply thresholding
    threshold = magnitude.mean() * 0.5
    magnitude[magnitude < threshold] = 0
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)


# Function for Image Normalization
def normalize_image(image, new_min=0, new_max=255):
    old_min, old_max = np.min(image), np.max(image)
    normalized = ((image - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    return normalized.astype(np.uint8)

# Function for Global Thresholding
def global_threshold(image, threshold):
    binary_image = np.where(image >= threshold, 255, 0)
    return binary_image.astype(np.uint8)

# Function for Local Thresholding (Adaptive)
def local_threshold(image, window_size):
    pad = window_size // 2
    padded_image = np.pad(image, pad, mode='reflect')
    thresholded_image = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            local_region = padded_image[i:i+window_size, j:j+window_size]
            local_thresh = np.mean(local_region)
            thresholded_image[i, j] = 255 if image[i, j] >= local_thresh else 0
    
    return thresholded_image.astype(np.uint8)

# Function for Frequency Domain Filtering (High-pass and Low-pass)
def frequency_filter(image, filter_type='Low Pass', cutoff_percent=10):
    """
    Apply frequency domain filter with cutoff as percentage of image size
    Args:
        image: Input image
        filter_type: 'Low Pass' or 'High Pass'
        cutoff_percent: Cutoff frequency as percentage of image size (1-50%)
    """
    # Convert percentage to actual radius
    rows, cols = image.shape
    min_dimension = min(rows, cols)
    cutoff = int((cutoff_percent / 100.0) * min_dimension)
    
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    
    if filter_type == 'Low Pass':
        mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 1
    elif filter_type == 'High Pass':
        mask.fill(1)
        mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 0
    
    filtered_dft = dft_shift * mask
    dft_ishift = np.fft.ifftshift(filtered_dft)
    filtered_image = np.fft.ifft2(dft_ishift)
    
    return np.clip(np.abs(filtered_image), 0, 255).astype(np.uint8)


def compute_histogram(image):
    histogram = [0] * 256
    for row in image:
        for pixel in row:
            histogram[pixel] += 1
    return np.array(histogram)

def convert_to_grayscale(image):
    """
    يحوّل الصورة إلى تدرجات الرمادي إذا كانت ملونة،
    أما إذا كانت رمادية بالفعل، يرجعها كما هي بدون تعديل.
    """
    # لو الصورة رمادي فعلًا (تحتوي على قناة واحدة فقط)، رجّعها كما هي
    if len(image.shape) == 2:
        return image, None, None, None  # لا يوجد قنوات لون لأن الصورة رمادي بالفعل

    # لو الصورة ملونة، افصل القنوات
    red_channel, green_channel, blue_channel = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # تحويل الصورة إلى رمادي باستخدام معادلة Y = 0.299R + 0.587G + 0.114B
    gray_image = (0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel).astype(np.uint8)

    return gray_image, red_channel, green_channel, blue_channel

# def gray_image(image):
#
#     h, w = image.shape[:2]
#     converted_image = convert_to_grayscale(image)[0]
#     return converted_image.reshape(h, w)
import cv2
import numpy as np

def gray_image(image):
    h, w = image.shape[:2]
    converted_image = convert_to_grayscale(image)[0]
    return converted_image.reshape(h, w)


def recover_rgb_image(equalized_red_channel, equalized_green_channel, equalized_blue_channel):
    equalized_rgb_image = np.stack((equalized_red_channel, equalized_green_channel, equalized_blue_channel), axis=2)
    return equalized_rgb_image

def compute_distribution_curve(image, histogram):
    total_pixels = image.size 
    probability_distribution_curve = histogram / total_pixels
    return probability_distribution_curve

def compute_cumulative_distribution_function(image, histogram):
    cumulative_distribution_function = [sum(histogram[:i+1]) for i in range(256)]
    cdf_min = min(cumulative_distribution_function)

    normalized_cumulative_distribution_function =\
          [(cumulative_distribution_function[i] - cdf_min) / (image.size - cdf_min) * 255 for i in range(256)]
    
    normalized_cumulative_distribution_function =\
          np.round(normalized_cumulative_distribution_function).astype(np.uint8)
    
    return normalized_cumulative_distribution_function

def equalize_image(image, normalized_cumulative_distribution_function):
    equalized_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            equalized_image[i, j] = normalized_cumulative_distribution_function[image[i, j]]
    return equalized_image