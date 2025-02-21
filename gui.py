import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QComboBox, QSlider, QLabel, QGridLayout, QRadioButton, QStackedWidget, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from qt_material import apply_stylesheet
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def create_line(horizontal = False, thick = True):
        line = QFrame() 
        line.setFrameShape(QFrame.HLine) if horizontal else line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        if thick: line.setStyleSheet("border: 1px solid purple;")
        return line

def create_label(text:str):
     label = QLabel(text)
     label.setMaximumHeight(9)
     label.setStyleSheet("")
     return label

class ImageProcessingUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing Toolbox")
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)  # Main layout: Left + Right (stacked Center & Right)

        # Left Panel (Controls)
        left_panel = QVBoxLayout()
        self.init_left_panel(left_panel)

        # Central Area (Center Panel and Right Panel stacked vertically)
        central_area = QVBoxLayout()

        # Center Panel (Image Display)
        center_panel = QVBoxLayout()
        self.original_label = QLabel("Original Image")
        self.processed_label = QLabel("Processed Image")
        center_layout = QGridLayout()
        center_layout.addWidget(self.original_label, 0, 0)
        center_layout.addWidget(self.processed_label, 0, 1)
        center_panel.addLayout(center_layout)

        # Right Panel (Empty for now)
        right_panel = QVBoxLayout()
        self.init_right_panel(right_panel)

        # Add Center and Right Panels to the central area
        central_area.addLayout(center_panel)  # Center Panel on top
        central_area.addLayout(right_panel)   # Right Panel below

        # Add panels to main layout
        main_layout.addLayout(left_panel, 1)       # Left Panel
        main_layout.addLayout(central_area, 3)     # Stacked Center & Right Panels
        
        self.main_page = QWidget()
        self.main_page.setLayout(main_layout)

        self.hybrid_page = QWidget()
        self.init_hybrid_page()

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.main_page)
        self.stacked_widget.addWidget(self.hybrid_page)
        self.setCentralWidget(self.stacked_widget)

    def init_left_panel(self, layout):
        """Initialize the left panel with controls and sliders."""
        # Load Image Button
        self.load_button = QPushButton("Load Image")
        layout.addWidget(self.load_button)

        self.separator_one = create_line(horizontal= True)
        layout.addWidget(self.separator_one)

        # Noise Addition
        self.noise_combo = QComboBox()
        self.noise_combo.addItems(["Uniform", "Gaussian", "Salt & Pepper"])
        layout.addWidget(QLabel("Add Noise:"))
        layout.addWidget(self.noise_combo)

        noise_slider_layout = QHBoxLayout()
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setMinimum(0)
        self.noise_slider.setMaximum(100)
        self.noise_slider.setValue(50)
        self.noise_slider.valueChanged.connect(lambda value: self.update_slider_label(value, self.noise_value_label))
        self.noise_value_label = QLabel("50")
        noise_slider_layout.addWidget(self.noise_slider)
        noise_slider_layout.addWidget(self.noise_value_label)
        layout.addLayout(noise_slider_layout)

        self.add_noise_button = QPushButton("Add Noise")
        layout.addWidget(self.add_noise_button)

        self.separator_two = create_line(horizontal= True)
        layout.addWidget(self.separator_two)

        # Low Pass Filters
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Average", "Gaussian", "Median"])
        layout.addWidget(QLabel("Low Pass Filter:"))
        layout.addWidget(self.filter_combo)

        filter_slider_layout = QHBoxLayout()
        self.filter_slider = QSlider(Qt.Horizontal)
        self.filter_slider.setMinimum(3)
        self.filter_slider.setMaximum(15)
        self.filter_slider.setValue(5)
        self.filter_slider.setSingleStep(2)
        self.filter_slider.valueChanged.connect(lambda value: self.update_slider_label(value, self.filter_value_label))
        self.filter_value_label = QLabel("5")
        filter_slider_layout.addWidget(self.filter_slider)
        filter_slider_layout.addWidget(self.filter_value_label)
        layout.addLayout(filter_slider_layout)

        self.apply_filter_button = QPushButton("Apply Filter")
        layout.addWidget(self.apply_filter_button)

        # Edge Detection
        self.edge_combo = QComboBox()
        self.edge_combo.addItems(["Sobel", "Roberts", "Prewitt", "Canny"])
        layout.addWidget(QLabel("Edge Detection:"))
        layout.addWidget(self.edge_combo)

        self.edge_button = QPushButton("Detect Edges")
        layout.addWidget(self.edge_button)

        # Histogram
        self.hist_button = QPushButton("Draw Histogram")
        layout.addWidget(self.hist_button)

        # Equalization
        self.equalize_button = QPushButton("Equalize Image")
        layout.addWidget(self.equalize_button)

        # Normalization
        self.normalize_button = QPushButton("Normalize Image")
        layout.addWidget(self.normalize_button)

        # Thresholding
        self.threshold_radio_global = QRadioButton("Global Thresholding")
        self.threshold_radio_local = QRadioButton("Local Thresholding")
        layout.addWidget(self.threshold_radio_global)
        layout.addWidget(self.threshold_radio_local)

        threshold_slider_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(128)
        self.threshold_slider.valueChanged.connect(lambda value: self.update_slider_label(value, self.threshold_value_label))
        self.threshold_value_label = QLabel("128")
        threshold_slider_layout.addWidget(self.threshold_slider)
        threshold_slider_layout.addWidget(self.threshold_value_label)
        layout.addLayout(threshold_slider_layout)

        self.threshold_button = QPushButton("Apply Threshold")
        layout.addWidget(self.threshold_button)

        # Color to Grayscale
        self.grayscale_button = QPushButton("Convert to Grayscale")
        layout.addWidget(self.grayscale_button)

        # Frequency Domain Filters
        self.freq_combo = QComboBox()
        self.freq_combo.addItems(["High Pass", "Low Pass"])
        layout.addWidget(QLabel("Frequency Domain Filter:"))
        layout.addWidget(self.freq_combo)

        self.freq_button = QPushButton("Apply Frequency Filter")
        layout.addWidget(self.freq_button)

        # Hybrid Images
        self.hybrid_button = QPushButton("Hybrid Image")
        self.hybrid_button.clicked.connect(self.open_hybrid_page)
        layout.addWidget(self.hybrid_button)

    def init_right_panel(self,layout):
        """Initialize the right panel with histograms and CDF/PDF plots."""
        layout_it = QGridLayout()
        
        # Original Image Plot
        self.original_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        layout_it.addWidget(QLabel("Original Image Analysis"), 0, 0)
        layout_it.addWidget(self.original_canvas, 1, 0)

        # Equalized Image Plot
        self.equalized_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        layout_it.addWidget(QLabel("Equalized Image Analysis"), 0, 1)
        layout_it.addWidget(self.equalized_canvas, 1, 1)

        layout.addLayout(layout_it)
    

    def init_hybrid_page(self):
        layout = QHBoxLayout()

        # input_layout = QVBoxLayout()

        
        # Image Upload Section
        upload_layout = QVBoxLayout()

        # First Image
        first_image_layout = QVBoxLayout()
        self.first_image_label = QLabel("First Image")
        self.first_image_label.setFixedSize(300, 300)
        first_image_layout.addWidget(self.first_image_label)
        first_image_button = QPushButton("Load First Image")
        # first_image_button.clicked.connect(lambda: self.load_image_for_hybrid(1))
        first_image_button.setFixedHeight(30)
        first_image_layout.addWidget(first_image_button)
        upload_layout.addLayout(first_image_layout)

        # Second Image
        second_image_layout = QVBoxLayout()
        self.second_image_label = QLabel("Second Image")
        self.second_image_label.setFixedSize(300, 300)
        second_image_layout.addWidget(self.second_image_label)
        second_image_button = QPushButton("Load Second Image")
        # second_image_button.clicked.connect(lambda: self.load_image_for_hybrid(2))
        second_image_button.setFixedHeight(30)
        second_image_layout.addWidget(second_image_button)
        upload_layout.addLayout(second_image_layout)

        layout.addLayout(upload_layout)

        # Hybrid Image Section
        
        hybrid_layout = QVBoxLayout()
        
        create_hybrid_button = QPushButton("Create Hybrid Image")
        # create_hybrid_button.clicked.connect(self.create_hybrid_image)
        create_hybrid_button.setFixedHeight(30)
        hybrid_layout.addWidget(create_hybrid_button)
        
        
        self.hybrid_image_label = QLabel("Hybrid Image")
        self.hybrid_image_label.setFixedSize(300, 300)
        hybrid_layout.addWidget(self.hybrid_image_label)
    
    
        # Back Button
        back_button = QPushButton("Back to Main Page")
        back_button.clicked.connect(self.back_to_main_page)
        back_button.setFixedHeight(30)
        hybrid_layout.addWidget(back_button)
        
        layout.addLayout(hybrid_layout)
        
        self.hybrid_page.setLayout(layout)

        # Initialize variables
        self.first_image = None
        self.second_image = None
        self.hybrid_image = None


    def update_slider_label(self, value, label):
        """Update the label with the slider's current value."""
        label.setText(str(value))
    
    def open_hybrid_page(self):
        """Switch to the hybrid image page."""
        self.stacked_widget.setCurrentIndex(1)

    def back_to_main_page(self):
        """Switch back to the main page."""
        self.stacked_widget.setCurrentIndex(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingUI()
    # apply_stylesheet(app, theme='dark_teal.xml')
    window.show()
    sys.exit(app.exec_())