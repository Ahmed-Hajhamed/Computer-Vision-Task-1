import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QComboBox, QSlider, QLabel, QGridLayout, QRadioButton, QStackedWidget, QFrame, QGroupBox
)
from PyQt5.QtCore import Qt
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
        self.image_layout = QGridLayout()

        self.image_label = QLabel("Image")
        self.image_label.setFixedSize(300, 300)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_layout.addWidget(self.image_label, 0, 0, 1, 2)

        self.load_button = QPushButton("Load Image")
        self.load_button.setFixedHeight(30)
        self.image_layout.addWidget(self.load_button, 1, 0)
        
        self.filters = QComboBox()
        self.filters.addItems(["Low Pass", "High Pass"])
        self.image_layout.addWidget(self.filters, 1, 1)

        self.slider = create_slider(1, 50, 10) 
        self.image_layout.addWidget(self.slider, 2, 0, 1, 2)
        self.slider_value = QLabel("10%")
        self.slider.valueChanged.connect(lambda value: self.slider_value.setText(f"{value}%"))
        self.image_layout.addWidget(self.slider_value, 2, 2)


class ImageProcessingUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing Toolbox")
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
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

        self.load_button = QPushButton("Load Image")
        self.rest_button = QPushButton("Save Image")
        layout.addLayout(horizontal_layout_maker([self.load_button, self.rest_button]))

        #process on 
        self.process_combo = QComboBox()
        self.process_combo.addItems(["Original", "Processed"])
        layout.addLayout(horizontal_layout_maker([QLabel("Process on:"), self.process_combo]))

        # Noise Addition
        self.noise_combo = QComboBox()
        self.noise_combo.addItems(["Uniform", "Gaussian", "Salt & Pepper"])
        self.noise_combo.currentIndexChanged.connect(lambda index: self.enable_controls([self.sigma_slider_noise, 
                                                                                         self.sigma_value_label_noise], index == 1, True))
        layout.addLayout(horizontal_layout_maker([QLabel("Add Noise :"), self.noise_combo]))

        self.noise_slider = create_slider(0, 100, 50)
        self.noise_slider.valueChanged.connect(lambda value: self.noise_value_label.setText(f"{value}%"))
        self.noise_value_label = QLabel("50%")
        layout.addLayout(horizontal_layout_maker([QLabel("Intensity :"), self.noise_slider, self.noise_value_label]))

        self.sigma_slider_noise = create_slider(0, 100, 50)
        self.sigma_slider_noise.valueChanged.connect(lambda value: self.sigma_value_label_noise.setText(str(value)))
        self.sigma_value_label_noise = QLabel("50")
        layout.addLayout(horizontal_layout_maker([QLabel("Sigma :"), self.sigma_slider_noise, self.sigma_value_label_noise]))
        self.enable_controls([self.sigma_slider_noise, self.sigma_value_label_noise], False)


        self.add_noise_button = QPushButton("Add Noise")
        layout.addWidget(self.add_noise_button)

        # Low Pass Filters
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Average", "Gaussian", "Median"])
        self.filter_combo.currentIndexChanged.connect(lambda index: self.enable_controls([self.sigma_slider_filter, 
                                                                                          self.sigma_value_label_filter], index == 1))
        layout.addLayout(horizontal_layout_maker([QLabel("Low Pass Filter:"), self.filter_combo]))

        self.filter_slider = create_slider(3, 15, 3)
        self.filter_slider.setSingleStep(2)
        self.filter_slider.valueChanged.connect(lambda value: self.filter_value_label.setText(str(value)))
        self.filter_value_label = QLabel("3")
        layout.addLayout(horizontal_layout_maker([QLabel("Kernel Size :"),self.filter_slider, self.filter_value_label]))

        self.sigma_slider_filter = create_slider(0, 100, 50)
        self.sigma_slider_filter.valueChanged.connect(lambda value: self.sigma_value_label_filter.setText(str(value)))
        self.sigma_value_label_filter = QLabel("50")
        layout.addLayout(horizontal_layout_maker([QLabel("Sigma :"), self.sigma_slider_filter, self.sigma_value_label_filter]))
        self.enable_controls([self.sigma_slider_filter, self.sigma_value_label_filter], False)

        self.apply_filter_button = QPushButton("Apply Filter")
        layout.addWidget(self.apply_filter_button)

        # Edge Detection
        self.edge_combo = QComboBox()
        self.edge_combo.addItems(["Sobel", "Roberts", "Prewitt", "Canny"])
        layout.addLayout(horizontal_layout_maker([QLabel("Edge Detection :"), self.edge_combo]))

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
        self.freq_combo.addItems(["Low Pass", "High Pass"])
        layout.addLayout(horizontal_layout_maker([QLabel("Frequency Domain Filter :"), self.freq_combo]))

        # Change slider range to percentage of image size (1-50%)
        self.cutoff_slider = create_slider(1, 50, 10)  # Default 10%
        self.cutoff_value_label = QLabel("10%")
        self.cutoff_slider.valueChanged.connect(
            lambda value: self.cutoff_value_label.setText(f"{value}%")
        )
        layout.addLayout(horizontal_layout_maker([QLabel("Cutoff Radius:"), self.cutoff_slider, self.cutoff_value_label]))

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
        # Create a canvas with 3 subfigures for the original image
        self.original_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        self.original_fig = self.original_canvas.figure
        self.original_curve_plots = []
        # Create a 1x3 grid of subfigures
        for i in range(1,4):
            self.original_curve_plots.append(self.original_fig.add_subplot(1, 3, i))
        self.original_fig.tight_layout()
        layout.addWidget(QLabel("Original Image Analysis"), 0, 0)
        layout.addWidget(self.original_canvas, 1, 0)

        # Equalized Image Plot
        self.equalized_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        self.equalized_fig = self.equalized_canvas.figure
        self.equalized_curve_plots = []
        for i in range(1,4):
            self.equalized_curve_plots.append(self.equalized_fig.add_subplot(1, 3, i))
        self.equalized_fig.tight_layout()
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
        self.save_hybrid_button = QPushButton("Save Hybrid Image")
        self.create_hybrid_button.setFixedHeight(30)
        hybrid_layout.addLayout(horizontal_layout_maker([self.create_hybrid_button, self.save_hybrid_button]))
        
        
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
        self.threshold_radio_global = QRadioButton("Global")
        self.threshold_radio_local = QRadioButton("Local")
        self.threshold_radio_global.setChecked(True)
        radio_layout = horizontal_layout_maker([self.threshold_radio_global, self.threshold_radio_local])
        
        # Connect radio buttons to enable/disable sliders
        self.threshold_radio_global.toggled.connect(self.update_threshold_controls)
        self.threshold_radio_local.toggled.connect(self.update_threshold_controls)
        
        # Global threshold slider with label in horizontal layout
        self.threshold_slider = create_slider(0, 255, 127)
        self.threshold_value = QLabel("127")  # Simplified label
        self.threshold_value.setMinimumWidth(30)  # Ensure label has enough space
        self.threshold_slider.valueChanged.connect(
            lambda value: self.threshold_value.setText(str(value))
        )
        global_slider_layout = horizontal_layout_maker([QLabel("Global:"), self.threshold_slider, self.threshold_value])
        
        # Local threshold slider with label in horizontal layout
        self.local_threshold_slider = create_slider(3, 31, 3)
        self.local_threshold_slider.setSingleStep(2)
        self.local_threshold_value = QLabel("3")  # Simplified label
        self.local_threshold_value.setMinimumWidth(30)  # Ensure label has enough space
        self.local_threshold_slider.valueChanged.connect(
            lambda value : self.local_threshold_value.setText(str(value))
        )
        local_slider_layout = horizontal_layout_maker([QLabel("Window:"), self.local_threshold_slider, self.local_threshold_value])
        
        # Apply button
        self.threshold_button = QPushButton("Apply Threshold")
        
        # Add all components to layout with minimal spacing
        threshold_layout.setSpacing(5)  # Reduce vertical spacing
        threshold_layout.addLayout(radio_layout)
        threshold_layout.addLayout(global_slider_layout)
        threshold_layout.addLayout(local_slider_layout)
        threshold_layout.addWidget(self.threshold_button)
        
        # Initial state of sliders
        self.update_threshold_controls()
        
        threshold_group.setLayout(threshold_layout)
        return threshold_group

    def update_threshold_controls(self):
        """Enable/disable threshold sliders based on radio button selection"""
        # Enable global threshold controls if global is selected
        self.threshold_slider.setEnabled(self.threshold_radio_global.isChecked())
        self.threshold_value.setEnabled(self.threshold_radio_global.isChecked())
        
        # Enable local threshold controls if local is selected
        self.local_threshold_slider.setEnabled(self.threshold_radio_local.isChecked())
        self.local_threshold_value.setEnabled(self.threshold_radio_local.isChecked())
    
    def enable_controls(self, widgets:list,enable=True, noise=False):
        """Enable or disable all controls in the UI."""
        for widget in widgets:
            if hasattr(widget, "setEnabled"):
                widget.setEnabled(enable)
        if noise:
            self.enable_controls([self.noise_slider, self.noise_value_label], not enable)
    
    def visible_controls(self, widgets:list,visible=True):
        """Enable or disable all controls in the UI."""
        for widget in widgets:
            widget.setVisible(visible)

def horizontal_layout_maker(widgets):
    layout = QHBoxLayout()
    for widget in widgets:
        layout.addWidget(widget)
    return layout


# Draw a sample 2D array in the first subfigure
# This method can be called whenever you want to update the plot
def draw_2d_array(data, plot, subplot_index=0, cmap='gray', title='2D Array'):
            """
            Draw a 2D array in one of the subfigures
            
            Parameters:
            - data: 2D numpy array to display
            - subplot_index: index of the subplot (0, 1, or 2)
            - cmap: colormap to use (e.g., 'viridis', 'gray', 'hot')
            - title: title for the subplot
            """
            ax = plot[subplot_index]
            ax.clear()
            ax.plot(data)
            ax.set_title(title)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingUI()
    # apply_stylesheet(app, theme='dark_teal.xml')
    window.show()
    sys.exit(app.exec_())