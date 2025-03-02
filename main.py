import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from  gui import ImageProcessingUI, draw_2d_array
import cv2
from qt_material import apply_stylesheet
import functions as f


class ImageProcessing(ImageProcessingUI):
    def __init__(self):
        super().__init__()

        # Default image
        self.image = None
        self.processed_image = None
        self.original_histogram = None
        self.original_red_histogram = None
        self.original_green_histogram = None
        self.original_blue_histogram = None
    
        self.processed_histogram = None
        self.processed_red_histogram = None
        self.processed_green_histogram = None
        self.processed_blue_histogram = None

        self.original_pdf = None
        self.original_cdf = None
        self.original_red_pdf = None
        self.original_red_cdf = None
        self.original_green_pdf = None
        self.original_green_cdf = None
        self.original_blue_pdf = None
        self.original_blue_cdf = None

        self.processed_pdf = None
        self.processed_cdf = None
        self.processed_red_pdf = None
        self.processed_red_cdf = None
        self.processed_green_pdf = None
        self.processed_green_cdf = None
        self.processed_blue_pdf = None
        self.processed_blue_cdf = None 

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
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.processed_image = self.image.copy()
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
        self.original_histogram = f.compute_histogram(self.image)
        self.processed_histogram = f.compute_histogram(self.processed_image)
        self.original_pdf = f.compute_distribution_curve(self.image, self.original_histogram)
        self.processed_pdf = f.compute_distribution_curve(self.processed_image, self.processed_histogram)
        self.original_cdf = f.compute_cumulative_distribution_function(self.image, self.original_histogram)
        self.processed_cdf = f.compute_cumulative_distribution_function(self.processed_image, self.processed_histogram)
        draw_2d_array(self.original_histogram, self.original_curve_plots, 0, title="Original Histogram")
        draw_2d_array(self.processed_histogram, self.equalized_curve_plots, 0,title= "Processed Histogram")
        draw_2d_array(self.original_pdf, self.original_curve_plots, 1,title= "Original PDF")
        draw_2d_array(self.processed_pdf, self.equalized_curve_plots, 1, title="Processed PDF")
        draw_2d_array(self.original_cdf, self.original_curve_plots, 2, title="Original CDF")
        draw_2d_array(self.processed_cdf, self.equalized_curve_plots, 2, title="Processed CDF")
        self.original_canvas.draw()
        self.equalized_canvas.draw()
    
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
        noisy_image = self.processed_image.copy() if self.process_combo.currentText()== "Processed" else self.image.copy()

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
        image =  self.processed_image.copy() if self.process_combo.currentText()== "Processed" else self.image.copy()

        if filter_type == "Average":
            filtered_image = f.average_filter(image, kernel_size)
        elif filter_type == "Gaussian":
            filtered_image = f.gaussian_filter(image, kernel_size, 3)
        elif filter_type == "Median":
            filtered_image = f.median_filter(image, kernel_size)
        
        self.processed_image = filtered_image
        self.update_display()

    def detect_edges(self):
        """Detect edges in the image."""
        self.check_processed_image()
        edge_type = self.edge_combo.currentText()
        image  = self.processed_image.copy() if self.process_combo.currentText()== "Processed" else self.image.copy()

        if edge_type == "Sobel":
            edges = f.sobel_edge_detection(image)
        
        elif edge_type == "Roberts":
            edges = f.roberts_edge_detection(image)

        elif edge_type == "Prewitt":
            edges = f.prewitt_edge_detection(image)
        
        elif edge_type == "Canny":
            edges = cv2.Canny(image, 100, 200)

        self.processed_image = edges
        self.update_display()
    

    def normalize_image(self):
        self.check_processed_image()
        image = self.processed_image.copy() if self.process_combo.currentText()== "Processed" else self.image.copy()
        normalized_image = f.normalize_image(image)
        self.processed_image = normalized_image
        self.update_display()

    def equalize_image(self):
        self.check_processed_image()
        image  = self.processed_image.copy() if self.process_combo.currentText()== "Processed" else self.image.copy()
        img = f.gray_image(image) if len(image) == 3 else image
        hist = f.compute_histogram(img)
        normalized_cdf = f.compute_cumulative_distribution_function(img, hist)
        equalized_image = f.equalize_image(img, normalized_cdf)
        self.processed_image = equalized_image
        self.update_display()
    
    def convert_image(self):
        self.check_processed_image()
        image = self.processed_image.copy() if self.process_combo.currentText()== "Processed" else self.image.copy()
        if len(image) == 2:
            QMessageBox.warning(self, "Error", "The image is already in grayscale.")
        else:
            self.processed_image = f.gray_image(image)
            self.update_display()

    def apply_thresholding(self):
        self.check_processed_image()
        image = self.processed_image.copy() if self.process_combo.currentText()== "Processed" else self.image.copy()
        if self.threshold_radio_global.isChecked():
            self.processed_image = f.global_threshold(
                image, 
                self.threshold_slider.value()
            )
        else:
            # Convert to grayscale if needed
            img = f.gray_image(image) if len(image.shape) == 3 else image
            # Use local threshold slider value
            window_size = self.local_threshold_slider.value()
            # Ensure window size is odd
            if window_size % 2 == 0:
                window_size += 1
            self.processed_image = f.local_threshold(img, window_size)
        self.update_display()

    def apply_frequency_filter(self):
        self.check_processed_image()
        image = self.processed_image.copy() if self.process_combo.currentText()== "Processed" else self.image.copy()
        filter_type = self.freq_combo.currentText()
        img =  f.gray_image(image) if len(image.shape) == 3 else image
        self.processed_image = f.frequency_filter(img, filter_type, int(self.cutoff_slider.value()))
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
            first_resized = f.gray_image(cv2.resize(self.first_image, (256, 256)))
            second_resized = f.gray_image(cv2.resize(self.second_image, (256, 256)))

            # Apply low-pass and high-pass filters
            low_pass = f.frequency_filter(first_resized,self.first_image_widget.filters.currentText(), 
                                          int(self.first_image_widget.slider.value()))
            high_pass = f.frequency_filter(second_resized,self.second_image_widget.filters.currentText(),
                                            int(self.second_image_widget.slider.value()))

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
