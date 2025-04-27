import streamlit as st
# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(layout="wide")

import os
import re  
import pandas as pd
from io import StringIO
from PIL import Image
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --- Attempt to fix XLA/PTXAS issues ---
# IMPORTANT: Replace with the correct path to your CUDA installation visible from WSL
cuda_path = '/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'  # <-- VERIFY THIS PATH
if os.path.exists(cuda_path):
    st.sidebar.info(f"Setting XLA_FLAGS to use CUDA path: {cuda_path}")
    # Set flags to help XLA find CUDA and potentially fallback if ptxas is missing
    os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir="{cuda_path}" --xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true'
else:
    st.sidebar.warning(f"Specified CUDA path not found: {cuda_path}. XLA might still have issues.")
    # Still try the fallback flag
    os.environ['XLA_FLAGS'] = '--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true'

# Disable TensorFlow logging noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR

# --- Configuration (should match training script) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(script_dir, 'advanced_model.h5')
COMPONENTS_PATH = os.path.join(script_dir, 'advanced_model_components.joblib')
TRAINING_HISTORY_IMG = os.path.join(script_dir, 'training_history.png')
CONFUSION_MATRIX_IMG = os.path.join(script_dir, 'advanced_confusion_matrix.png')
ENCRYPTION_VIS_IMG = os.path.join(script_dir, 'encrypted_data_visualization.png')
LOG_FILE_PATH = os.path.join(script_dir, 'model_run.log')

# Configure basic logging (can be done after set_page_config)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.title("Advanced Satellite Image Classification Model Analysis")

# --- Check for GPU Availability ---
st.sidebar.header("System Information")
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    st.sidebar.success(f"GPU detected and available: {gpu_devices}")
    st.sidebar.write("TensorFlow will use GPU for model operations")
else:
    st.sidebar.info("No GPU detected. Running on CPU.")

# --- Load Saved Components ---
st.header("Loading Model Components")
try:
    if os.path.exists(COMPONENTS_PATH):
        components = joblib.load(COMPONENTS_PATH)
        CLASS_NAMES = components.get('class_names', [])
        IMG_SIZE = components.get('img_size', 64)  # Default to 64 if not found
        NUM_CLASSES = len(CLASS_NAMES)
        st.write(f"Loaded components: Image Size={IMG_SIZE}, Classes={NUM_CLASSES}")
        st.write(f"Class Names: {', '.join(CLASS_NAMES)}")
    else:
        st.error(f"Error: Components file not found at {COMPONENTS_PATH}")
        st.write("Check that the file exists and the path is correct.")
        st.stop()
except Exception as e:
    st.error(f"Error loading components: {str(e)}")
    st.write("This could be due to incompatible joblib versions or corrupted file.")
    st.stop()

# --- Load Pre-trained Model ---
st.header("Loading Pre-trained Model & Displaying Summary")
model_summary_string = None
model = None # Initialize model
if os.path.exists(MODEL_SAVE_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        st.success(f"Model loaded successfully from {MODEL_SAVE_PATH}")

        # Capture model summary
        string_buffer = StringIO()
        model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
        model_summary_string = string_buffer.getvalue()
        string_buffer.close()

        st.subheader("Model Architecture Summary")
        st.code(model_summary_string, language='text')

    except Exception as e:
        st.error(f"Error loading model or generating summary: {e}")
        st.stop() # Stop execution if model loading fails
else:
    st.error(f"Error: Model file not found at {MODEL_SAVE_PATH}")
    st.stop()

# --- Image Preprocessing Functions ---
def preprocess_dataset_image(image, label):
    """Preprocess image coming from the TensorFlow dataset."""
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def preprocess_uploaded_image(image_pil):
    """Preprocess an uploaded image (PIL format)."""
    image = tf.convert_to_tensor(np.array(image_pil))
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    # Add batch dimension
    image = tf.expand_dims(image, axis=0)
    return image

# --- Load Test Dataset ---
st.header("Loading and Preparing Test Data")
test_ds = None # Initialize test_ds
try:
    # Use the same split as in training
    ds_test, ds_info = tfds.load(
        'eurosat/rgb',
        split='train[90%:]', # Ensure this matches the test split in advanced_model.py
        as_supervised=True,
        with_info=True,
    )
    test_ds = ds_test.map(preprocess_dataset_image).batch(32).prefetch(tf.data.AUTOTUNE)
    st.write(f"Test dataset loaded successfully. Number of test samples: {ds_info.splits['train[90%:]'].num_examples}")
except Exception as e:
    st.error(f"Error loading or preparing dataset: {e}")
    # Don't stop here, evaluation is optional if dataset fails but model loaded

# --- Evaluate Model and Generate Predictions ---
st.header("Model Evaluation on Test Set")
y_true = None
y_pred = None
if test_ds and model: # Only evaluate if both model and test_ds are loaded
    with st.spinner("Evaluating model and generating predictions (using GPU if available)..."): # Added GPU note
        try:
            # model.evaluate and model.predict will use the GPU if TensorFlow detected one earlier.
            logging.info("Starting model evaluation...")
            test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
            logging.info(f"Model evaluation completed. Loss: {test_loss}, Accuracy: {test_accuracy}")

            y_true_list = []
            y_pred_probs = []
            # Get a single batch for feature extraction example if needed later
            # Use try-except for iterator in case dataset is empty or fails
            try:
                sample_batch = next(iter(test_ds.take(1)))
            except StopIteration:
                st.warning("Test dataset appears to be empty. Cannot generate predictions or sample batch.")
                sample_batch = None # Ensure sample_batch is defined
            except Exception as e_iter:
                st.error(f"Error getting sample batch from test dataset: {e_iter}")
                sample_batch = None # Ensure sample_batch is defined

            logging.info("Starting prediction generation...")
            # Check if dataset iteration is possible before proceeding
            if sample_batch is not None: # Check if we could get at least one batch
                pred_count = 0
                for images, labels in test_ds:
                    # This predict call will use the GPU if available.
                    predictions = model.predict(images, verbose=0) # Added verbose=0 to silence predict progress
                    y_pred_probs.extend(predictions)
                    y_true_list.extend(labels.numpy())
                    pred_count += len(labels.numpy())
                logging.info(f"Generated predictions for {pred_count} samples.")

                if y_true_list: # Ensure we actually got predictions
                    y_pred = np.argmax(y_pred_probs, axis=1)
                    y_true = np.array(y_true_list)
                    # y_pred is already a numpy array

                    st.metric(label="Test Accuracy", value=f"{test_accuracy:.4f}")
                    st.metric(label="Test Loss", value=f"{test_loss:.4f}")

                    st.subheader("Classification Report")
                    # Generate report as dictionary
                    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0) # Added zero_division=0
                    # Convert dictionary to DataFrame
                    report_df = pd.DataFrame(report_dict).transpose()
                    # Display DataFrame
                    st.dataframe(report_df)
                else:
                    st.warning("No predictions were generated, possibly due to an empty or problematic test set.")
            else:
                 st.warning("Skipping prediction generation as the test dataset could not be iterated.")


        except tf.errors.InternalError as e:
            logging.error(f"TensorFlow Internal Error during evaluation/prediction: {e}", exc_info=True)
            st.error(
                f"TensorFlow Internal Error: {e}\n\n"
                "This often indicates a problem with the GPU setup (e.g., CUDA/cuDNN libraries) or resource exhaustion. "
                "Check the console logs for more details.\n\n"
                "**Troubleshooting Tips:**\n"
                "1. Verify your CUDA toolkit and cuDNN installation.\n"
                "2. Ensure TensorFlow has access to `ptxas` (part of CUDA toolkit). Check your PATH.\n"
                "3. Try setting the `XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda` environment variable.\n"
                "4. Monitor GPU memory usage.\n"
                "5. Restart the Streamlit application and potentially your system."
            )
            # Allow continuing even if evaluation fails
        except Exception as e:
            logging.error(f"Error during evaluation or prediction: {e}", exc_info=True)
            st.error(f"An unexpected error occurred during evaluation or prediction: {e}")
            # Allow continuing even if evaluation fails
else:
    st.warning("Skipping model evaluation as the model or test dataset could not be loaded.")


# --- Display Visualizations ---
st.header("Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion Matrix")
    if y_true is not None and y_pred is not None: # Check if evaluation results exist
        fig_cm = None # Initialize fig_cm
        try:
            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax_cm)
            ax_cm.set_title('Confusion Matrix')
            ax_cm.set_ylabel('True Label')
            ax_cm.set_xlabel('Predicted Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig_cm)
        except Exception as e:
            st.error(f"Could not generate confusion matrix: {e}")
            # Fallback to loading the image if generation fails or image exists
            if os.path.exists(CONFUSION_MATRIX_IMG):
                st.image(CONFUSION_MATRIX_IMG, caption="Confusion Matrix (from file)")
        finally:
            # Ensure the figure is closed to free memory
            if fig_cm:
                plt.close(fig_cm)
    else:
        st.info("Confusion matrix requires successful model evaluation on the test set.")
        if os.path.exists(CONFUSION_MATRIX_IMG):
             st.image(CONFUSION_MATRIX_IMG, caption="Confusion Matrix (from file - fallback)")


with col2:
    st.subheader("Training History")
    if os.path.exists(TRAINING_HISTORY_IMG):
        st.image(TRAINING_HISTORY_IMG, caption="Model Training History (Accuracy & Loss)")
    else:
        st.warning(f"Training history image not found at {TRAINING_HISTORY_IMG}")

# --- Interactive Prediction ---
st.header("Test with Your Own Image")
uploaded_file = st.file_uploader("Choose a satellite image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model:
    try:
        # Read the image using Pillow
        image_pil = Image.open(uploaded_file).convert('RGB') # Ensure image is RGB

        # Display the uploaded image
        st.image(image_pil, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying (using GPU if available)...") # Added GPU note

        # Preprocess the image
        processed_image = preprocess_uploaded_image(image_pil)

        # Make prediction
        # This predict call will also use the GPU if available.
        logging.info("Starting prediction for uploaded image...")
        prediction_probs = model.predict(processed_image, verbose=0)[0] # Get probabilities for the single image
        predicted_class_index = np.argmax(prediction_probs)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        prediction_confidence = prediction_probs[predicted_class_index]
        logging.info(f"Prediction complete: {predicted_class_name} ({prediction_confidence:.4f})")

        st.success(f"Prediction: **{predicted_class_name}** (Confidence: {prediction_confidence:.4f})")

        # Optionally display top N predictions
        st.write("Top 3 Predictions:")
        top_indices = np.argsort(prediction_probs)[-3:][::-1]
        for i in top_indices:
            st.write(f"- {CLASS_NAMES[i]}: {prediction_probs[i]:.4f}")

    except tf.errors.InternalError as e:
        logging.error(f"TensorFlow Internal Error during prediction for uploaded image: {e}", exc_info=True)
        st.error(
            f"TensorFlow Internal Error during prediction: {e}\n\n"
            "This often indicates a problem with the GPU setup (e.g., CUDA/cuDNN libraries) or resource exhaustion. "
            "Check the console logs for more details.\n\n"
            "**Troubleshooting Tips:**\n"
            "1. Verify your CUDA toolkit and cuDNN installation.\n"
            "2. Ensure TensorFlow has access to `ptxas` (part of CUDA toolkit). Check your PATH.\n"
            "3. Try setting the `XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda` environment variable.\n"
            "4. Monitor GPU memory usage.\n"
            "5. Restart the Streamlit application and potentially your system."
        )
    except Exception as e:
        logging.error(f"Error processing uploaded image: {e}", exc_info=True)
        st.error(f"Error processing uploaded image: {e}")
elif uploaded_file is not None and not model:
    st.error("Model not loaded. Cannot perform prediction.")

st.success("Analysis App Ready.")

