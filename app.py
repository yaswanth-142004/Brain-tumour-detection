import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, backend as K
import matplotlib.pyplot as plt
import io

# Custom RBF Layer (required for model loading)
class RBFLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[-1]),
                                       initializer='uniform',
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer='ones',
                                     trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, x):
        return K.exp(-self.betas * K.sum(K.square(x), axis=-1, keepdims=True))

    def get_config(self):
        config = super(RBFLayer, self).get_config()
        config.update({"output_dim": self.output_dim})
        return config

# Hebbian Smoothing Function
def hebbian_smoothing(probability_mask, iterations=5, learning_rate=0.1):
    smoothed = probability_mask.copy()
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    kernel = kernel / np.sum(kernel)
    
    for _ in range(iterations):
        neighbor_activity = cv2.filter2D(smoothed, -1, kernel)
        delta = learning_rate * (smoothed * neighbor_activity)
        decay = 0.01 * smoothed 
        smoothed = smoothed + delta - decay
        smoothed = np.clip(smoothed, 0, 1)
    return smoothed

# Load Model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            'brain_tumor_segmentation_model.keras',
            custom_objects={'RBFLayer': RBFLayer}
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess Image
def preprocess_image(image, img_size=(128, 128)):
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if grayscale
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize and normalize
    img_resized = cv2.resize(img_array, img_size)
    img_normalized = img_resized / 255.0
    
    return img_normalized, img_array

# Predict Function
def predict_segmentation(model, image, threshold=0.2, hebbian_iterations=15):
    # Add batch dimension
    img_batch = np.expand_dims(image, axis=0)
    
    # Get prediction
    prediction = model.predict(img_batch, verbose=0)[0][:, :, 0]
    
    # Apply Hebbian smoothing
    smoothed = hebbian_smoothing(prediction, iterations=hebbian_iterations)
    
    # Apply threshold
    final_mask = (smoothed > threshold).astype(float)
    
    return prediction, smoothed, final_mask

# Page Configuration
st.set_page_config(
    page_title="LGG MRI SEGMENTATION",
    page_icon="brain_image_lol.png",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E86DE;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #636E72;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F0F3F7;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"> LGG MRI Segmentation by using U-NET </div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced MRI Analysis using U-Net + RBF Neural Network with Hebbian Learning</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("Model Parameters")
    threshold = st.slider(
        "Segmentation Threshold",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Lower values detect more tumor regions (higher sensitivity)"
    )
    
    hebbian_iter = st.slider(
        "Hebbian Smoothing Iterations",
        min_value=5,
        max_value=30,
        value=15,
        step=5,
        help="More iterations produce smoother segmentation masks"
    )
    
    st.divider()
    
    st.subheader("ℹ️ About")
    st.info("""
    This application uses a hybrid deep learning approach:
    
    - **U-Net Architecture**: For feature extraction and segmentation
    - **RBF Layer**: For enhanced boundary detection
    - **Hebbian Learning**: For post-processing and smoothing
    
    **Model Performance:**
    - Dice Coefficient: ~0.88
    - IoU: ~0.84
    - Sensitivity: ~0.934
    - Specificity: ~0.99
    - Hausdorff distance: 6.84
    """)
    
    st.divider()
    
    st.markdown("___ Built this with guidance from : Dr Prasun Dutta (SRM-AP) _____")

# Main Content
model = load_model()

if model is None:
    st.error("⚠️ Model file not found. Please ensure 'brain_tumor_segmentation_model.keras' is in the same directory.")
    st.stop()

st.success("✅ Model loaded successfully!")

# File Upload
uploaded_file = st.file_uploader(
    "Upload an MRI scan (JPG, PNG, TIF)",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    help="Upload a brain MRI image for tumor segmentation"
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    
    # Preprocess
    with st.spinner("Processing image..."):
        processed_img, original_img = preprocess_image(image)
        
        # Predict
        raw_pred, smoothed_pred, final_mask = predict_segmentation(
            model, 
            processed_img, 
            threshold=threshold,
            hebbian_iterations=hebbian_iter
        )
    
    # Display Results
    st.header("📊 Segmentation Results")
    
    # Create columns for visualization
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("Original MRI")
        st.image(original_img, use_column_width=True)
    
    with col2:
        st.subheader("Raw Prediction")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(raw_pred, cmap='hot')
        ax.axis('off')
        ax.set_title("Probability Map", fontsize=10)
        st.pyplot(fig)
        plt.close()
    
    with col3:
        st.subheader("Hebbian Smoothed")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(smoothed_pred, cmap='hot')
        ax.axis('off')
        ax.set_title("Smoothed Map", fontsize=10)
        st.pyplot(fig)
        plt.close()
    
    with col4:
        st.subheader("Final Segmentation")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(final_mask, cmap='gray')
        ax.axis('off')
        ax.set_title(f"Threshold: {threshold}", fontsize=10)
        st.pyplot(fig)
        plt.close()
    
    # Overlay visualization
    st.header("🔍 Tumor Overlay")
    
    col_overlay1, col_overlay2 = st.columns(2)
    
    with col_overlay1:
        st.subheader("Segmentation Overlay")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cv2.resize(original_img, (128, 128)))
        ax.imshow(final_mask, cmap='Reds', alpha=0.5)
        ax.axis('off')
        ax.set_title("Tumor Region Highlighted", fontsize=12)
        st.pyplot(fig)
        plt.close()
    
    with col_overlay2:
        st.subheader("Tumor Contour")
        contour_img = cv2.resize(original_img, (128, 128)).copy()
        contours, _ = cv2.findContours(
            (final_mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(contour_img)
        ax.axis('off')
        ax.set_title("Tumor Boundary Detection", fontsize=12)
        st.pyplot(fig)
        plt.close()
    
    # Statistics
    st.header("📈 Analysis Statistics")
    
    tumor_pixels = np.sum(final_mask)
    total_pixels = final_mask.shape[0] * final_mask.shape[1]
    tumor_percentage = (tumor_pixels / total_pixels) * 100
    confidence = np.mean(smoothed_pred[final_mask > 0]) if tumor_pixels > 0 else 0
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Tumor Detected", "Yes" if tumor_pixels > 50 else "No")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_stat2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Tumor Area", f"{tumor_percentage:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_stat3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Pixels Affected", f"{int(tumor_pixels)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_stat4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Confidence", f"{confidence:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Download Options
    st.header("💾 Download Results")
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        # Save segmentation mask
        mask_img = Image.fromarray((final_mask * 255).astype(np.uint8))
        buf = io.BytesIO()
        mask_img.save(buf, format='PNG')
        st.download_button(
            label="📥 Download Segmentation Mask",
            data=buf.getvalue(),
            file_name="tumor_segmentation.png",
            mime="image/png"
        )
    
    with col_dl2:
        # Save overlay
        overlay_fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cv2.resize(original_img, (128, 128)))
        ax.imshow(final_mask, cmap='Reds', alpha=0.5)
        ax.axis('off')
        
        buf2 = io.BytesIO()
        overlay_fig.savefig(buf2, format='PNG', bbox_inches='tight', dpi=150)
        plt.close()
        
        st.download_button(
            label="📥 Download Overlay Image",
            data=buf2.getvalue(),
            file_name="tumor_overlay.png",
            mime="image/png"
        )

else:
    # Display instructions when no file is uploaded
    st.info("👆 Please upload an MRI scan to begin analysis")
    
    # Example instructions
    with st.expander("📖 How to use this application"):
        st.markdown("""
        ### Step-by-step Guide:
        
        1. **Upload Image**: Click on the file uploader and select an MRI scan
        2. **Adjust Settings**: Use the sidebar to fine-tune segmentation parameters
        3. **View Results**: Examine the segmentation output and overlays
        4. **Download**: Save the segmentation mask or overlay for your records
        
        ### Supported Formats:
        - JPG/JPEG
        - PNG
        - TIF/TIFF
        
        ### Tips for Best Results:
        - Use high-quality MRI scans
        - Lower threshold values increase sensitivity
        - Higher Hebbian iterations create smoother boundaries
        """)
    
    with st.expander("🔬 Model Architecture"):
        st.markdown("""
        ### Hybrid Deep Learning Architecture:
        
        **U-Net Encoder-Decoder**:
        - 3 encoder blocks with Conv2D layers
        - 2 decoder blocks with skip connections
        - MaxPooling for downsampling
        - Conv2DTranspose for upsampling
        
        **RBF Enhancement**:
        - Radial Basis Function layer for boundary refinement
        - Gaussian activation functions
        - Adaptive center and beta parameters
        
        **Hebbian Post-Processing**:
        - Iterative smoothing based on Hebbian learning
        - Neighborhood activity correlation
        - Adaptive decay mechanism
        """)
