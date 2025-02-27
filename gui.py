import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QComboBox, QSlider, QLabel, QGridLayout, QRadioButton, QStackedWidget, QFrame, QGroupBox
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

def create_slider(minimum:int, maximum:int, value:int, horizontal=True):
    slider = QSlider(Qt.Horizontal) if horizontal else QSlider(Qt.Vertical)
    slider.setMinimum(minimum)
    slider.setMaximum(maximum)
    slider.setValue(value)
    return slider

class image_hybrid:
    def __init__(self):
        self.image_layout = QVBoxLayout()

        self.image_label = QLabel("Image")
        self.image_label.setFixedSize(300, 300)
        self.image_layout.addWidget(self.image_label)

        first_h_button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Image")
        self.load_button.setFixedHeight(30)
        first_h_button_layout.addWidget(self.load_button)
        
        self.filters = QComboBox()
        self.filters.addItems(["low filter", "high filter"])
        first_h_button_layout.addWidget(self.filters)

        self.image_layout.addLayout(first_h_button_layout)
        
        h_slider_layout = QHBoxLayout()
        self.slider = create_slider(0, 255, 128)
        h_slider_layout.addWidget(self.slider)
        self.slider_value = QLabel("128")
        self.slider.valueChanged.connect(lambda value: update_slider_label(value, self.slider_value))
        h_slider_layout.addWidget(self.slider_value)

        self.image_layout.addLayout(h_slider_layout)


class ImageProcessingUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing Toolbox")
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
        # main_widget = QWidget(self)
        # self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()  

        # Left Panel (Controls)
        left_panel = QVBoxLayout()
        self.init_left_panel(left_panel)

        # Central Area (Center Panel and Right Panel stacked vertically)
        central_area = QVBoxLayout()

        # Center Panel (Image Display)
        center_panel = QVBoxLayout()
        self.original_label = QLabel("Original Image")
        self.processed_label = QLabel("Processed Image")
        center_layout = QVBoxLayout()
        center_layout.addWidget(self.original_label, 0, Qt.AlignCenter)
        center_layout.addWidget(self.processed_label, 1, Qt.AlignCenter)
        center_panel.addLayout(center_layout)

        # Add Center and Right Panels to the central area
        central_area.addLayout(center_panel)  # Center Panel on top

        # Add panels to main layout
        main_layout.addLayout(left_panel, 1)       # Left Panel
        main_layout.addLayout(central_area, 3)     # Stacked Center & Right Panels
        
        self.main_page = QWidget()
        self.main_page.setLayout(main_layout)

        self.plot_page = QWidget()
        self.init_plot_page()

        self.hybrid_page = QWidget()
        self.init_hybrid_page()

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.main_page)
        self.stacked_widget.addWidget(self.plot_page)
        self.stacked_widget.addWidget(self.hybrid_page)
        self.setCentralWidget(self.stacked_widget)

    def init_left_panel(self, layout):
        """Initialize the left panel with controls and sliders."""
        # Load Image Button
        layout_load = QHBoxLayout()

        self.load_button = QPushButton("Load Image")
        layout_load.addWidget(self.load_button)

        self.rest_button = QPushButton("Reset Image")
        layout_load.addWidget(self.rest_button)

        layout.addLayout(layout_load)

        # Noise Addition
        self.noise_combo = QComboBox()
        self.noise_combo.addItems(["Uniform", "Gaussian", "Salt & Pepper"])
        layout.addWidget(QLabel("Add Noise:"))
        layout.addWidget(self.noise_combo)

        noise_slider_layout = QHBoxLayout()
        self.noise_slider = create_slider(0, 100, 50)
        self.noise_slider.valueChanged.connect(lambda value: update_slider_label(value, self.noise_value_label))
        self.noise_value_label = QLabel("50")
        noise_slider_layout.addWidget(self.noise_slider)
        noise_slider_layout.addWidget(self.noise_value_label)
        layout.addLayout(noise_slider_layout)

        self.add_noise_button = QPushButton("Add Noise")
        layout.addWidget(self.add_noise_button)

        # Low Pass Filters
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Average", "Gaussian", "Median"])
        layout.addWidget(QLabel("Low Pass Filter:"))
        layout.addWidget(self.filter_combo)

        filter_slider_layout = QHBoxLayout()
        self.filter_slider = create_slider(3, 15, 5)
        self.filter_slider.setSingleStep(2)
        self.filter_slider.valueChanged.connect(lambda value: update_slider_label(value, self.filter_value_label))
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
        self.hist_button.clicked.connect(self.open_plot_page)
        layout.addWidget(self.hist_button)

        # Equalization
        self.equalize_button = QPushButton("Equalize Image")
        layout.addWidget(self.equalize_button)

        # Normalization
        self.normalize_button = QPushButton("Normalize Image")
        layout.addWidget(self.normalize_button)

        # Thresholding
        layout.addWidget(self.init_threshold_controls())

        # Color to Grayscale
        self.grayscale_button = QPushButton("Convert to Grayscale")
        layout.addWidget(self.grayscale_button)

        # Frequency Domain Filters
        self.freq_combo = QComboBox()
        self.freq_combo.addItems(["High Pass", "Low Pass"])
        layout.addWidget(QLabel("Frequency Domain Filter:"))
        layout.addWidget(self.freq_combo)
        
        cutoff_slider_layout = QHBoxLayout()
        self.cutoff_slider = create_slider(0, 255, 128)
        self.cutoff_value_label = QLabel("128")
        self.cutoff_slider.valueChanged.connect(lambda value: update_slider_label(value, self.cutoff_value_label))
        cutoff_slider_layout.addWidget(self.cutoff_slider)
        cutoff_slider_layout.addWidget(self.cutoff_value_label)
        layout.addLayout(cutoff_slider_layout)

        self.freq_button = QPushButton("Apply Frequency Filter")
        layout.addWidget(self.freq_button)

        # Hybrid Images
        self.hybrid_button = QPushButton("Hybrid Image")
        self.hybrid_button.clicked.connect(self.open_hybrid_page)
        layout.addWidget(self.hybrid_button)

    def init_plot_page(self):
        """Initialize the right panel with histograms and CDF/PDF plots."""

        layout = QGridLayout()
        # Original Image Plot
        self.original_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        layout.addWidget(QLabel("Original Image Analysis"), 0, 0)
        layout.addWidget(self.original_canvas, 1, 0)

        # Equalized Image Plot
        self.equalized_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        layout.addWidget(QLabel("Equalized Image Analysis"), 2, 0)
        layout.addWidget(self.equalized_canvas, 3, 0)

        # Back Button
        back_button = QPushButton("Back to Main Page")
        back_button.clicked.connect(self.back_to_main_page)
        layout.addWidget(back_button, 4, 0, 1, 1)

        self.plot_page.setLayout(layout)
    

    def init_hybrid_page(self):
        layout = QHBoxLayout()

        # Image Upload Section
        upload_layout = QVBoxLayout()

        # First Image
        self.first_image_widget = image_hybrid()
        upload_layout.addLayout(self.first_image_widget.image_layout)

        # Second Image
        self.second_image_widget = image_hybrid()
        upload_layout.addLayout(self.second_image_widget.image_layout)

        layout.addLayout(upload_layout)

        # Hybrid Image Section
        
        hybrid_layout = QVBoxLayout()
        
        self.create_hybrid_button = QPushButton("Create Hybrid Image")
        self.create_hybrid_button.setFixedHeight(30)
        hybrid_layout.addWidget(self.create_hybrid_button)
        
        
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
    
    def open_hybrid_page(self):
        """Switch to the hybrid image page."""
        print("test")
        self.stacked_widget.setCurrentIndex(2)
    
    def open_plot_page(self):
        """Switch to the plot page."""
        self.stacked_widget.setCurrentIndex(1)

    def back_to_main_page(self):
        """Switch back to the main page."""
        self.stacked_widget.setCurrentIndex(0)

    def init_threshold_controls(self):
        threshold_group = QGroupBox("Thresholding")
        threshold_layout = QVBoxLayout()
        
        # Radio buttons in horizontal layout
        radio_layout = QHBoxLayout()
        self.threshold_radio_global = QRadioButton("Global")
        self.threshold_radio_local = QRadioButton("Local")
        self.threshold_radio_global.setChecked(True)
        radio_layout.addWidget(self.threshold_radio_global)
        radio_layout.addWidget(self.threshold_radio_local)
        
        # Global threshold slider with label in horizontal layout
        global_slider_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(127)
        self.threshold_value = QLabel("127")  # Simplified label
        self.threshold_value.setMinimumWidth(30)  # Ensure label has enough space
        self.threshold_slider.valueChanged.connect(
            lambda: self.threshold_value.setText(str(self.threshold_slider.value()))
        )
        global_slider_layout.addWidget(QLabel("Global:"))  # Short label
        global_slider_layout.addWidget(self.threshold_slider)
        global_slider_layout.addWidget(self.threshold_value)
        
        # Local threshold slider with label in horizontal layout
        local_slider_layout = QHBoxLayout()
        self.local_threshold_slider = QSlider(Qt.Horizontal)
        self.local_threshold_slider.setMinimum(3)
        self.local_threshold_slider.setMaximum(31)
        self.local_threshold_slider.setValue(3)
        self.local_threshold_slider.setSingleStep(2)
        self.local_threshold_value = QLabel("3")  # Simplified label
        self.local_threshold_value.setMinimumWidth(30)  # Ensure label has enough space
        self.local_threshold_slider.valueChanged.connect(
            lambda: self.local_threshold_value.setText(str(self.local_threshold_slider.value()))
        )
        local_slider_layout.addWidget(QLabel("Window:"))  # Short label
        local_slider_layout.addWidget(self.local_threshold_slider)
        local_slider_layout.addWidget(self.local_threshold_value)
        
        # Apply button
        self.threshold_button = QPushButton("Apply Threshold")
        
        # Add all components to layout with minimal spacing
        threshold_layout.setSpacing(5)  # Reduce vertical spacing
        threshold_layout.addLayout(radio_layout)
        threshold_layout.addLayout(global_slider_layout)
        threshold_layout.addLayout(local_slider_layout)
        threshold_layout.addWidget(self.threshold_button)
        
        threshold_group.setLayout(threshold_layout)
        return threshold_group

def update_slider_label(value, label):
        """Update the label with the slider's current value."""
        label.setText(str(value))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingUI()
    # apply_stylesheet(app, theme='dark_teal.xml')
    window.show()
    sys.exit(app.exec_())