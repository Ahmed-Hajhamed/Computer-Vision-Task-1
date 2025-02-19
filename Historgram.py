import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("Images\\man.tif", cv2.IMREAD_GRAYSCALE)

def compute_histogram(image):
    histogram = [0] * 256
    for row in image:
        for pixel in row:
            histogram[pixel] += 1
    return np.array(histogram)

def convert_to_grayscale(image_path: str):
    image = cv2.imread(image_path)  # Reads in BGR format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    red_channel, green_channel, blue_channel = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    gray_image = (0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel).astype(np.uint8)
    return gray_image, red_channel, green_channel, blue_channel

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

histogram = compute_histogram(image)
pdf = compute_distribution_curve(image, histogram)
cdf = compute_cumulative_distribution_function(image, histogram)
equalized_image = equalize_image(image, cdf)
equalized_histogram = compute_histogram(equalized_image)
equalized_pdf = compute_distribution_curve(equalized_image, equalized_histogram)
equalized_cdf = compute_cumulative_distribution_function(image, equalized_histogram)

# Display the original image
plt.figure(figsize=(12, 8))
plt.subplot(2, 4, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Display the histogram
plt.subplot(2, 4, 2)
plt.title("Histogram")
plt.plot(histogram)
plt.xlim([0, 256])

# Display the PDF
plt.subplot(2, 4, 3)
plt.title("PDF")
plt.plot(pdf)
plt.xlim([0, 256])

# Display the CDF
plt.subplot(2, 4, 4)
plt.title("CDF")
plt.plot(cdf)
plt.xlim([0, 256])

# Display the equalized image
plt.subplot(2, 4, 5)
plt.title("Equalized Image")
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

# Display the histogram of the equalized image
plt.subplot(2, 4, 6)
plt.title("Equalized Histogram")
plt.plot(equalized_histogram)
plt.xlim([0, 256])

# plt.figure(figsize=(12, 6))

# Compute and display the PDF of the equalized image
plt.subplot(2, 4, 7)
plt.title("Equalized PDF")
plt.plot(equalized_pdf)
plt.xlim([0, 256])

# Compute and display the CDF of the equalized image
plt.subplot(2, 4, 8)
plt.title("Equalized CDF")
plt.plot(equalized_cdf)
plt.xlim([0, 256])

plt.tight_layout()
plt.show()
