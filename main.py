import sys
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from  gui import ImageProcessingUI
import cv2
from qt_material import apply_stylesheet


class ImageProcessing(ImageProcessingUI):
    def __init__(self):
        super().__init__()

        # Default image
        self.image = None
        self.processed_image = None
    
    def load_image(self):
        """Load an image from file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp *.gif)", options=options)
        if file_path:
            self.image = cv2.imread(file_path)
            self.update_display()

    def rest_image(self):
        pass

    def update_display(self):
        """Update the original and processed image displays."""
        if self.image is not None:
            self.display_image(self.image, self.original_label)
        if self.processed_image is not None:
            self.display_image(self.processed_image, self.processed_label)

    def display_image(self, img, label):
        """Display an image in a QLabel."""
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def create_hybrid_image(self):
        """Create a hybrid image from two images."""
        options = QFileDialog.Options()
        file_path1, _ = QFileDialog.getOpenFileName(self, "Open First Image", "", "Images (*.png *.jpg *.bmp)", options=options)
        file_path2, _ = QFileDialog.getOpenFileName(self, "Open Second Image", "", "Images (*.png *.jpg *.bmp)", options=options)

        if file_path1 and file_path2:
            img1 = cv2.imread(file_path1)
            img2 = cv2.imread(file_path2)

            # Resize images to the same size
            img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

            # Apply low-pass and high-pass filters
            low_pass = cv2.GaussianBlur(img1, (51, 51), 0)
            high_pass = img2 - cv2.GaussianBlur(img2, (51, 51), 0)

            # Combine images
            hybrid = low_pass + high_pass
            self.processed_image = hybrid
            self.update_display()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app,  theme='dark_teal.xml')
    window = ImageProcessing()
    window.show()
    sys.exit(app.exec_())
