import numpy as np
from scipy import ndimage
import cv2
import tkinter as tk
from tkinter import filedialog


class ImageProcessor:
    def add_uniform_noise(self, image, low=0, high=50):
        """Add uniform noise to image"""
        noise = np.random.uniform(low, high, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def add_gaussian_noise(self, image, mean=0, sigma=25):
        """Add Gaussian noise to image"""
        noise = np.random.normal(mean, sigma, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def add_salt_pepper_noise(self, image, prob=0.05):
        """Add salt and pepper noise to image"""
        noisy_image = np.copy(image)
        # Salt noise
        salt_mask = np.random.random(image.shape) < prob / 2
        noisy_image[salt_mask] = 255
        # Pepper noise
        pepper_mask = np.random.random(image.shape) < prob / 2
        noisy_image[pepper_mask] = 0
        return noisy_image

    def average_filter(self, image, kernel_size=3):
        """Apply average filter"""
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        filtered_image = ndimage.convolve(image, kernel)
        return filtered_image.astype(np.uint8)

    def gaussian_filter(self, image, kernel_size=3, sigma=1):
        """Apply Gaussian filter"""
        x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                           np.linspace(-1, 1, kernel_size))
        d = np.sqrt(x * x + y * y)
        kernel = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
        kernel = kernel / kernel.sum()
        filtered_image = ndimage.convolve(image, kernel)
        return filtered_image.astype(np.uint8)

    def median_filter(self, image, kernel_size=3):
        """Apply median filter"""
        filtered_image = ndimage.median_filter(image, size=kernel_size)
        return filtered_image.astype(np.uint8)

    def sobel_edge_detection(self, image):
        """Apply Sobel edge detection"""
        # Convert image to float
        img = image.astype('float64')

        # Define Sobel kernels
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        # Apply convolution
        grad_x = ndimage.convolve(img, sobel_x)
        grad_y = ndimage.convolve(img, sobel_y)

        # Calculate magnitude and normalize
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        magnitude = magnitude * (255.0 / magnitude.max())

        # Apply thresholding
        threshold = magnitude.mean() * 0.6
        magnitude[magnitude < threshold] = 0

        return magnitude.astype(np.uint8)

    def roberts_edge_detection(self, image):
        """Apply Roberts edge detection"""
        # Convert image to float
        img = image.astype('float64')

        # Define Roberts kernels
        roberts_x = np.array([[1, 0],
                              [0, -1]])
        roberts_y = np.array([[0, 1],
                              [-1, 0]])

        # Pad image to handle boundaries
        padded_img = np.pad(img, ((0, 1), (0, 1)), mode='edge')

        # Apply convolution
        grad_x = ndimage.convolve(padded_img, roberts_x)
        grad_y = ndimage.convolve(padded_img, roberts_y)

        # Calculate magnitude and normalize
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        magnitude = magnitude[:-1, :-1]  # Remove padding
        magnitude = magnitude * (255.0 / magnitude.max())

        # Apply thresholding
        threshold = magnitude.mean() * 0.5
        magnitude[magnitude < threshold] = 0

        return magnitude.astype(np.uint8)

    def prewitt_edge_detection(self, image):
        """Apply Prewitt edge detection"""
        # Convert image to float
        img = image.astype('float64')

        # Define Prewitt kernels
        prewitt_x = np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]])
        prewitt_y = np.array([[-1, -1, -1],
                              [0, 0, 0],
                              [1, 1, 1]])

        # Apply convolution
        grad_x = ndimage.convolve(img, prewitt_x)
        grad_y = ndimage.convolve(img, prewitt_y)

        # Calculate magnitude and normalize
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        magnitude = magnitude * (255.0 / magnitude.max())

        # Apply thresholding
        threshold = magnitude.mean() * 0.6
        magnitude[magnitude < threshold] = 0

        return magnitude.astype(np.uint8)

    def canny_edge_detection(self, image, low_threshold=50, high_threshold=150):
        """Apply Canny edge detection"""
        return cv2.Canny(image, low_threshold, high_threshold)


def test_image_processing():
    """Test function with edge detection comparison"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="اختر الصورة طال عمرك", filetypes=[("image files", "*.jpg;*.png")])
    # قراءة الصورة
    original_image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)


    if original_image is None:
        print("خطأ فادح جدا يكاد يكون كارثة في قراءة الصورة!")
        return

    processor = ImageProcessor()
    #اختبار اضافة الضوضاء التجربة الخامسة
    add_uniform_noise = processor.add_uniform_noise(gray_image)
    add_gaussian_noise = processor.add_gaussian_noise(gray_image)
    add_salt_and_pepper_noise = processor.add_salt_pepper_noise(gray_image)
    combined_noises = np.hstack((add_uniform_noise, add_gaussian_noise, add_salt_and_pepper_noise))

    #اختبار تطبيق الفلاتر التجربة الثالثة
    average_filter = processor.average_filter(combined_noises)
    gaussian_filter = processor.gaussian_filter(combined_noises)
    median_filter = processor.median_filter(combined_noises)
    combined_filtered = np.hstack((average_filter, gaussian_filter, median_filter))


    #  اختبار كشف الحواف التجربة التاسعة
    edges_sobel = processor.sobel_edge_detection(gray_image)
    edges_roberts = processor.roberts_edge_detection(gray_image)
    edges_prewitt = processor.prewitt_edge_detection(gray_image)
    edges_canny = processor.canny_edge_detection(gray_image)
    combined_edges = np.hstack((edges_sobel, edges_roberts, edges_prewitt, edges_canny))

    # عرض النتائج جنباً إلى جنب


    #cv2.imshow('Edge Detection Comparison', combined_edges)
    cv2.imwrite('edge_detection_comparison.jpg', combined_edges)
    cv2.imwrite("noise_adding_comparison.jpg", combined_noises)
    cv2.imwrite("filtered_noised_images.jpg", combined_filtered)
    cv2.imwrite("gray image.jpg", gray_image)
    cv2.imwrite("the_original_image.jpg", original_image)




    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_image_processing()


# # للصور الملونة
# image = cv2.imread('test_image.jpg', cv2.IMREAD_COLOR)
#
# # معالجة كل قناة لون على حدة
# b, g, r = cv2.split(image)
# b_processed = processor.some_function(b)
# g_processed = processor.some_function(g)
# r_processed = processor.some_function(r)
# processed_image = cv2.merge([b_processed, g_processed, r_processed])