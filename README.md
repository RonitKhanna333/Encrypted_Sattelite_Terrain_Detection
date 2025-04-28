# Satellite Image Classification

## 1. Methodology
```
EuroSAT Dataset Loading → Image Preprocessing (Resize/Normalize) → Transfer Learning with MobileNetV2 → Feature Extraction → TenSEAL Homomorphic Encryption → Model Evaluation → Streamlit Web Interface
```

## 2. Description
* Dataset = EuroSAT RGB Satellite Imagery
* Best Model = MobileNetV2 Transfer Learning
* Best Accuracy = 91.85%
* Other information:
  * Advanced satellite image classification system using TensorFlow
  * Transfer learning with MobileNetV2 and custom CNN options
  * GPU-accelerated processing with CUDA/cuDNN optimization
  * Homomorphic encryption demonstration using TenSEAL
  * Interactive web interface built with Streamlit

## 3. Input / Output
| Test Image | Actual | Predicted | Result |
|------------|--------|-----------|--------|
| ![Image](annualcrop.jpg) | AnnualCrop | AnnualCrop | ✓ |
| ![Image](pasture.jpg) | Pasture | Pasture | ✓ |
| ![Image](permacrop.jpg) | PermanentCrop | PermanentCrop | ✓ |
| ![Image](residential.jpg) | Residential | Residential | ✓ |
| ![Image](sealake.jpg) | SeaLake | Forest | ✗ |

## 4. Live Link
Link: https://ronitkhanna.streamlit.app/

## 5. Screenshot of the Interface
![Interface Screenshot](image.png)

## Installation Requirements
```bash
# Clone the repository
https://github.com/RonitKhanna333/Hackspire.git
cd Hackspire

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

The project requires the following packages:
- TensorFlow
- TensorFlow Datasets
- NumPy
- Matplotlib
- Joblib
- scikit-learn
- TenSEAL
- Seaborn
- Streamlit (for the web interface)

## Usage

### Training the Model

Run the advanced model training script:

```bash
python advanced_model.py
```

This will:
1. Load and preprocess the EuroSAT RGB dataset
2. Train using either transfer learning or a custom CNN architecture
3. Evaluate the model on a test set
4. Generate visualizations including confusion matrices and training history
5. Create a feature extraction model
6. Demonstrate homomorphic encryption with TenSEAL

### Running the Web Interface

Start the Streamlit web application:

```bash
streamlit run streamlit_app.py
```

The interactive dashboard provides:
- Model architecture summary
- Test dataset evaluation metrics
- Classification reports and confusion matrices
- Interactive prediction capabilities for user-uploaded images
- System information including GPU availability

## GPU Configuration

The system is designed to automatically use GPU acceleration when available. For optimal performance:

1. Verify your CUDA installation path in both `advanced_model.py` and `streamlit_app.py`:
   ```python
   cuda_path = '/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'  # <-- VERIFY THIS PATH
   ```

2. Make sure your CUDA Toolkit and cuDNN libraries are properly installed.

3. The system includes fallback mechanisms to handle common PTXAS-related issues with TensorFlow.

## Troubleshooting GPU Issues

If you encounter GPU-related errors:
1. Verify your CUDA toolkit and cuDNN installation
2. Ensure TensorFlow has access to `ptxas` (part of CUDA toolkit)
3. Check that your PATH includes CUDA directories
4. Try setting the `XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda` environment variable
5. Monitor GPU memory usage to avoid out-of-memory errors

## Project Files

- `advanced_model.py`: Main training script with model architecture and TenSEAL demonstration
- `streamlit_app.py`: Web interface for model visualization and testing
- `advanced_model.h5`: Saved trained model file
- `advanced_model_components.joblib`: Saved model metadata and class information
- `advanced_feature_extractor.h5`: Feature extraction model for advanced analysis
- `advanced_confusion_matrix.png`: Visualization of model prediction accuracy
- `training_history.png`: Visualization of model training metrics over time
- `encrypted_data_visualization.png`: Visualization of homomorphic encryption capabilities
- `feature_importance.png`: Feature importance analysis visualization
- `requirements.txt`: Project dependencies

## Homomorphic Encryption

This project demonstrates privacy-preserving machine learning using homomorphic encryption:

1. Feature vectors are extracted from images using the trained model
2. The TenSEAL library provides CKKS encryption for these feature vectors
3. Visualizations compare original, encrypted, and decrypted data
4. Mean squared error is calculated to demonstrate encryption accuracy

## Acknowledgements

- EuroSAT dataset for providing high-quality satellite imagery
- TensorFlow and Streamlit communities
- TenSEAL project for homomorphic encryption capabilities
