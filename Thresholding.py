import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Function for Image Normalization
def normalize_image(image, new_min=0, new_max=255):
    old_min, old_max = np.min(image), np.max(image)
    normalized = (image - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
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
def frequency_filter(image, filter_type='low', cutoff=20):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    
    if filter_type == 'low':
        mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1
    elif filter_type == 'high':
        mask[:crow - cutoff, :] = 1
        mask[crow + cutoff:, :] = 1
        mask[:, :ccol - cutoff] = 1
        mask[:, ccol + cutoff:] = 1
    
    filtered_dft = dft_shift * mask
    dft_ishift = np.fft.ifftshift(filtered_dft)
    filtered_image = np.fft.ifft2(dft_ishift)
    
    return np.abs(filtered_image).astype(np.uint8)

# Load Image Function
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path).convert('L')
        image = np.array(image)
        process_image(image)

# Process Image and Show Results
def process_image(image):
    normalized_img = normalize_image(image)
    global_thresh_img = global_threshold(image, 127)
    local_thresh_img = local_threshold(image, 15)
    low_pass_img = frequency_filter(image, 'low', 30)
    high_pass_img = frequency_filter(image, 'high', 30)
    
    # Display results
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs[0, 0].imshow(image, cmap='gray'); axs[0, 0].set_title('Original Image')
    axs[0, 1].imshow(normalized_img, cmap='gray'); axs[0, 1].set_title('Normalized Image')
    axs[0, 2].imshow(global_thresh_img, cmap='gray'); axs[0, 2].set_title('Global Threshold')
    axs[1, 0].imshow(local_thresh_img, cmap='gray'); axs[1, 0].set_title('Local Threshold')
    axs[1, 1].imshow(low_pass_img, cmap='gray'); axs[1, 1].set_title('Low-Pass Filter')
    axs[1, 2].imshow(high_pass_img, cmap='gray'); axs[1, 2].set_title('High-Pass Filter')
    
    for ax in axs.ravel():
        ax.axis('off')
    
    plt.show()

# GUI Setup
root = tk.Tk()
root.title("Image Processing")

btn_load = tk.Button(root, text="Load Image", command=load_image)
btn_load.pack()

root.mainloop()