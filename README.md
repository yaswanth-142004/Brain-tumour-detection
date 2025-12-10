# 🧠 Brain Tumor Segmentation UI

A professional Streamlit web application for brain tumor segmentation using a hybrid U-Net + RBF neural network with Hebbian learning.

## Features

- **Interactive Web Interface**: Clean, professional UI built with Streamlit
- **Real-time Segmentation**: Upload MRI scans and get instant results
- **Advanced Visualization**: Multiple views including raw predictions, smoothed outputs, and overlays
- **Adjustable Parameters**: Fine-tune threshold and smoothing iterations
- **Statistical Analysis**: Detailed metrics on tumor area and confidence
- **Download Results**: Export segmentation masks and overlay images

## Model Architecture

The application uses a sophisticated hybrid approach:
- **U-Net Architecture**: Encoder-decoder structure with skip connections
- **RBF Layer**: Radial Basis Function layer for enhanced boundary detection
- **Hebbian Learning**: Post-processing smoothing based on neuroplasticity principles

### Performance Metrics
- Dice Coefficient: ~0.83
- IoU (Jaccard): ~0.71
- Sensitivity: ~0.83
- Specificity: ~0.99

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd lgg-mri-segmentation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model file exists**
   Make sure `brain_tumor_segmentation_model.keras` is in the same directory as `app.py`

## Usage

### Running the Application

1. **Start the Streamlit server**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   - The app will automatically open in your default browser
   - If not, navigate to `http://localhost:8501`

### Using the Interface

1. **Upload MRI Scan**
   - Click on the file uploader
   - Select a brain MRI image (JPG, PNG, or TIF format)

2. **Adjust Parameters** (Optional)
   - Use the sidebar sliders to adjust:
     - **Segmentation Threshold**: Controls sensitivity (0.1-0.5)
     - **Hebbian Iterations**: Controls smoothness (5-30)

3. **View Results**
   - Original MRI image
   - Raw probability map
   - Hebbian smoothed map
   - Final segmentation mask
   - Tumor overlay visualization
   - Contour detection

4. **Download Results**
   - Click download buttons to save:
     - Segmentation mask (PNG)
     - Overlay image (PNG)

## File Structure

```
lgg-mri-segmentation/
├── app.py                                  # Main Streamlit application
├── brain_tumor_segmentation_model.keras    # Trained model file
├── requirements.txt                        # Python dependencies
├── README.md                              # This file
└── psc.ipynb                              # Training notebook
```

## Technical Details

### Image Processing Pipeline

1. **Preprocessing**
   - Resize to 128x128 pixels
   - Normalize pixel values (0-1)
   - Convert to RGB if needed

2. **Prediction**
   - Forward pass through U-Net + RBF
   - Generate probability map

3. **Post-processing**
   - Apply Hebbian smoothing
   - Apply threshold for binary mask
   - Extract contours

### Customization

You can modify the following parameters in the sidebar:
- **Threshold**: Lower values = higher sensitivity, more tumor detection
- **Hebbian Iterations**: Higher values = smoother boundaries

## Troubleshooting

### Common Issues

**Model not found**
- Ensure `brain_tumor_segmentation_model.keras` is in the same directory as `app.py`
- Check file permissions

**Import errors**
- Run `pip install -r requirements.txt` again
- Ensure Python version is 3.8+

**Performance issues**
- Use smaller images
- Reduce Hebbian iterations
- Close other applications

## Requirements

- streamlit >= 1.31.0
- tensorflow >= 2.15.0
- opencv-python >= 4.9.0
- pillow >= 10.2.0
- numpy >= 1.24.3
- matplotlib >= 3.8.2
- scikit-learn >= 1.4.0

## License

This project is for educational and research purposes.

## Acknowledgments

- U-Net architecture inspired by Ronneberger et al.
- Dataset: LGG MRI Segmentation Dataset
- Built with Streamlit, TensorFlow, and OpenCV
