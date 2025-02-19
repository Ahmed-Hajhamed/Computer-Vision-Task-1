import Historgram

image, red_channel, green_channel, blue_channel = Historgram.convert_to_grayscale('Images\Bill-Gates.jpg')
histogram = Historgram.compute_histogram(image)
red_histogram = Historgram.compute_histogram(red_channel)
green_histogram = Historgram.compute_histogram(green_channel)
blue_histogram = Historgram.compute_histogram(blue_channel)
cdf = Historgram.compute_cumulative_distribution_function(image, histogram)
red_cdf = Historgram.compute_cumulative_distribution_function(red_channel, red_histogram)
green_cdf = Historgram.compute_cumulative_distribution_function(green_channel, green_histogram)
blue_cdf = Historgram.compute_cumulative_distribution_function(blue_channel, blue_histogram)
equalized_grayscale_image = Historgram.equalize_image(image, cdf)
equalized_red_channel = Historgram.equalize_image(red_channel, red_cdf)
equalized_green_channel = Historgram.equalize_image(green_channel, green_cdf)
equalized_blue_channel = Historgram.equalize_image(blue_channel, blue_cdf)
equalized_histogram = Historgram.compute_histogram(equalized_grayscale_image)
equalized_red_histogram = Historgram.compute_histogram(equalized_red_channel)
equalized_green_histogram = Historgram.compute_histogram(equalized_green_channel)
equalized_blue_histogram = Historgram.compute_histogram(equalized_blue_channel)
equalized_rgb_image = Historgram.recover_rgb_image(equalized_red_channel, equalized_green_channel, equalized_blue_channel)

import matplotlib.pyplot as plt
# Plot original and equalized histograms, and CDFs in one figure
plt.figure(figsize=(20, 15))

# Original histograms
plt.subplot(3, 4, 1)
plt.title('Original Red Histogram')
plt.plot(red_histogram, color='red')

plt.subplot(3, 4, 2)
plt.title('Original Green Histogram')
plt.plot(green_histogram, color='green')

plt.subplot(3, 4, 3)
plt.title('Original Blue Histogram')
plt.plot(blue_histogram, color='blue')

plt.subplot(3, 4, 4)
plt.title('Original Image Histogram')
plt.plot(histogram, color='black')

# Equalized histograms
plt.subplot(3, 4, 5)
plt.title('Equalized Red Histogram')
plt.plot(equalized_red_histogram, color='red')

plt.subplot(3, 4, 6)
plt.title('Equalized Green Histogram')
plt.plot(equalized_green_histogram, color='green')

plt.subplot(3, 4, 7)
plt.title('Equalized Blue Histogram')
plt.plot(equalized_blue_histogram, color='blue')

plt.subplot(3, 4, 8)
plt.title('Equalized Image Histogram')
plt.plot(equalized_histogram, color='black')

# CDFs
plt.subplot(3, 4, 9)
plt.title('Red Channel CDF')
plt.plot(red_cdf, color='red')

plt.subplot(3, 4, 10)
plt.title('Green Channel CDF')
plt.plot(green_cdf, color='green')

plt.subplot(3, 4, 11)
plt.title('Blue Channel CDF')
plt.plot(blue_cdf, color='blue')

plt.subplot(3, 4, 12)
plt.title('Image CDF')
plt.plot(cdf, color='black')

plt.tight_layout()
plt.show()

# Display original and equalized images
plt.figure(figsize=(20, 8))

plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray', vmin=0, vmax=255)

plt.subplot(2, 3, 2)
plt.title('Equalized Grayscale Image')
plt.imshow(equalized_grayscale_image, cmap='gray', vmin=0, vmax=255)

plt.subplot(2, 3, 3)
plt.title('Equalized RGB Image')
plt.imshow(equalized_rgb_image, vmin=0, vmax=255)

plt.subplot(2, 3, 4)
plt.title('Red Channel')
plt.imshow(red_channel, cmap='Reds', vmin=0, vmax=255)

plt.subplot(2, 3, 5)
plt.title('Green Channel')
plt.imshow(green_channel, cmap='Greens', vmin=0, vmax=255)

plt.subplot(2, 3, 6)
plt.title('Blue Channel')
plt.imshow(blue_channel, cmap='Blues', vmin=0, vmax=255)

plt.tight_layout()
plt.show()
