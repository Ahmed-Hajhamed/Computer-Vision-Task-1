import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from  gui import ImageProcessingUI
import cv2
from qt_material import apply_stylesheet

import functions as f


class ImageProcessing(ImageProcessingUI):
    def __init__(self):
        super().__init__()

        # Default image
        self.image = None
        self.processed_image = None

        # Initialize variables
        self.first_image = None
        self.second_image = None
        self.hybrid_image = None

        #connect of main page
        self.load_button.clicked.connect(self.load_image) 
        self.rest_button.clicked.connect(self.rest_image)
        
        self.add_noise_button.clicked.connect(self.add_noise)
        self.apply_filter_button.clicked.connect(self.apply_filter)
        self.edge_button.clicked.connect(self.detect_edges)
        self.normalize_button.clicked.connect(self.normalize_image)
        self.equalize_button.clicked.connect(self.equalize_image)
        self.grayscale_button.clicked.connect(self.convert_image)
        self.freq_button.clicked.connect(self.apply_frequency_filter)
        self.threshold_button.clicked.connect(self.apply_thresholding)


        # Connect of hybrid page
        self.first_image_widget.load_button.clicked.connect(lambda: self.load_image_for_hybrid(1))
        self.second_image_widget.load_button.clicked.connect(lambda: self.load_image_for_hybrid(2))
        self.create_hybrid_button.clicked.connect(self.create_hybrid_image)


    
    def load_image(self):
        """Load an image from file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp *.gif *.tif)")
        if file_path:
            self.image = cv2.imread(file_path)
            self.processed_image = None
            self.processed_label.clear()
            self.processed_label.setText("Processed Image")
            self.update_display()

    def rest_image(self):
        """Reset processed image."""
        self.processed_image = self.image
        self.update_display()

    def update_display(self):
        """Update the original and processed image displays."""
        if self.image is not None:
            self.display_image(self.image, self.original_label)
        if self.processed_image is not None:
            self.display_image(self.processed_image, self.processed_label)

    def display_image(self, img, label):
        """Display an image in a QLabel."""
        if len(img.shape) == 2:
            h, w = img.shape
            bytes_per_line = w
            qt_image = QImage(bytes(img.data), w, h, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qt_image)
            label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        else:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def add_noise(self):
        """Add noise to the image."""
        self.check_processed_image()
        noise_type = self.noise_combo.currentText()
        intensity = self.noise_slider.value() / 100.0
        noisy_image = self.processed_image.copy()

        if noise_type == "Uniform":
            noisy_image = f.add_uniform_noise(noisy_image, intensity)
        elif noise_type == "Gaussian":
            noisy_image = f.add_gaussian_noise(noisy_image, intensity)
        elif noise_type == "Salt & Pepper":
            noisy_image = f.add_salt_pepper_noise(noisy_image, intensity)
        
        self.processed_image = noisy_image
        self.update_display()

    def apply_filter(self):
        """Apply a low-pass filter to the image."""
        self.check_processed_image()
        filter_type = self.filter_combo.currentText()
        kernel_size = self.filter_slider.value()

        if filter_type == "Average":
            filtered_image = f.average_filter(self.processed_image, kernel_size)
        elif filter_type == "Gaussian":
            filtered_image = f.gaussian_filter(self.processed_image, kernel_size, 1)
        elif filter_type == "Median":
            filtered_image = f.median_filter(self.processed_image, kernel_size)
        print(filtered_image.shape)
        self.processed_image = filtered_image
        self.update_display()

    def detect_edges(self):
        """Detect edges in the image."""
        self.check_processed_image()
        edge_type = self.edge_combo.currentText()

        if edge_type == "Sobel":
            edges = f.sobel_edge_detection(self.processed_image)
        
        elif edge_type == "Roberts":
            edges = f.roberts_edge_detection(self.processed_image)

        elif edge_type == "Prewitt":
            edges = f.prewitt_edge_detection(self.processed_image)
        
        elif edge_type == "Canny":
            edges = cv2.Canny(self.processed_image, 100, 200)

        self.processed_image = edges
        self.update_display()
    

    def normalize_image(self):
        self.check_processed_image()
        normalized_image = f.normalize_image(self.processed_image)
        self.processed_image = normalized_image
        self.update_display()

    def equalize_image(self):
        self.check_processed_image()
        img = f.gray_image(self.processed_image) if len(self.processed_image.shape) == 3 else self.processed_image
        hist = f.compute_histogram(img)
        normalized_cdf = f.compute_cumulative_distribution_function(img, hist)
        equalized_image = f.equalize_image(img, normalized_cdf)
        self.processed_image = equalized_image
        self.update_display()
    
    def convert_image(self):
        self.check_processed_image()
        if len(self.processed_image.shape) == 2:
            QMessageBox.warning(self, "Error", "The image is already in grayscale.")
        else:
            self.processed_image = f.gray_image(self.processed_image)
            self.update_display()

    def apply_thresholding(self):
        self.check_processed_image()
        if self.threshold_radio_global.isChecked():
            self.processed_image = f.global_threshold(self.processed_image, self.threshold_slider.value())
        else:
            self.processed_image = f.local_threshold(self.processed_image, self.threshold_slider.value())
        self.update_display()

    def apply_frequency_filter(self):
        self.check_processed_image()
        filter_type = self.freq_combo.currentText()
        image =  f.gray_image(self.processed_image) if len(self.processed_image.shape) == 3 else self.processed_image
        self.processed_image = f.frequency_filter(image, filter_type, int(self.cutoff_slider.value()))
        self.update_display()


    def load_image_for_hybrid(self, image_index):
        """Load an image for hybrid creation."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp *.gif *.tif)")
        if file_path:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if image_index == 1:
                self.first_image = img
                self.display_image_for_hybrid(img, self.first_image_widget.image_label)
            elif image_index == 2:
                self.second_image = img
                self.display_image_for_hybrid(img, self.second_image_widget.image_label)
    
    def display_image_for_hybrid(self, img, label):
        """Display an image in a QLabel for hybrid page."""
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w , ch= rgb_image.shape
        bytes_per_line = w * ch
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))

    def create_hybrid_image(self):
        """Create and display the hybrid image."""
        if self.first_image is not None and self.second_image is not None:
            # Resize images to the same size
            first_resized = cv2.resize(self.first_image, (256, 256))
            second_resized = cv2.resize(self.second_image, (256, 256))

            # Apply low-pass and high-pass filters
            low_pass = cv2.GaussianBlur(first_resized, (51, 51), 0)
            high_pass = second_resized - cv2.GaussianBlur(second_resized, (51, 51), 0)

            # Combine images
            hybrid = low_pass + high_pass
            self.hybrid_image = hybrid
            self.display_image_for_hybrid(hybrid, self.hybrid_image_label)

    def check_processed_image(self):
        if self.processed_image is None:
            self.processed_image = self.image


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # apply_stylesheet(app,  theme='dark_teal.xml')
    window = ImageProcessing()
    window.show()
    sys.exit(app.exec_())
