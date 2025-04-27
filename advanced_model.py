import os  # Import os first

# --- Attempt to fix XLA/PTXAS issues ---
# IMPORTANT: Replace with the correct path to your CUDA installation visible from WSL
cuda_path = '/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'  # <-- VERIFY THIS PATH
if os.path.exists(cuda_path):
    print(f"Setting XLA_FLAGS to use CUDA path: {cuda_path}")
    # Set flags to help XLA find CUDA and potentially fallback if ptxas is missing
    os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir="{cuda_path}" --xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true'
else:
    print(f"Warning: Specified CUDA path not found: {cuda_path}. XLA might still have issues.")
    # Still try the fallback flag
    os.environ['XLA_FLAGS'] = '--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true'

# Disable TensorFlow logging noise (optional, can help clean up output)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, applications
import time
from sklearn.metrics import classification_report, confusion_matrix
import tenseal as ts  # Import TenSEAL for encryption
import seaborn as sns  # For better visualization
import os  # Import os for environment variable setting if needed

# --- GPU Configuration Check ---
print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs found and configured.")
        # Optionally, force TensorFlow to use a specific GPU
        # tf.config.set_visible_devices(gpus[0], 'GPU') # Uncomment to use only the first GPU
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"Error configuring GPU memory growth: {e}")
else:
    print("No GPU found. TensorFlow will run on CPU.")
    # Optionally, hide GPUs if you want to force CPU execution for testing
    # try:
    #     tf.config.set_visible_devices([], 'GPU')
    #     print("GPU usage explicitly disabled. Forcing CPU execution.")
    # except RuntimeError as e:
    #     print(f"Error disabling GPU: {e}")

# --- Configuration ---
IMG_SIZE = 128  # Larger image size for better feature extraction
BATCH_SIZE = 32
EPOCHS = 30
MODEL_SAVE_PATH = './advanced_model.h5'
FEATURE_EXTRACTOR_PATH = './advanced_feature_extractor.h5'
COMPONENTS_PATH = './advanced_model_components.joblib'
RANDOM_STATE = 42

# --- 1. Load EuroSAT Dataset ---
print("Loading EuroSAT dataset...")
(ds_train, ds_val, ds_test), ds_info = tfds.load(
    'eurosat/rgb',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],  # Proper training/validation/test split
    as_supervised=True,
    with_info=True,
)
NUM_CLASSES = ds_info.features['label'].num_classes
CLASS_NAMES = ds_info.features['label'].names
print(f"Number of classes: {NUM_CLASSES}")
print(f"Class names: {CLASS_NAMES}")

# --- 2. Image Preprocessing ---
def preprocess_image(image, label):
    """Preprocess image for a CNN."""
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Prepare datasets for training
train_ds = ds_train.map(preprocess_image).cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = ds_val.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = ds_test.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- 3. Build Advanced CNN Model ---
# Option 1: Custom CNN with more capacity
def build_custom_cnn():
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Option 2: Transfer Learning from pre-trained model
def build_transfer_learning_model():
    # Use a pre-trained model (MobileNetV2 is a good balance of accuracy and efficiency)
    base_model = applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create new model on top
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model, base_model

# Choose which approach to use - transfer learning typically gives better accuracy
use_transfer_learning = True

if use_transfer_learning:
    model, base_model = build_transfer_learning_model()
    print("Created model using transfer learning from MobileNetV2")
else:
    model = build_custom_cnn()
    print("Created custom CNN model")

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 4. Train the Model ---
print("\nTraining the model...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

start_time = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# --- 5. Evaluate the Model ---
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {test_accuracy:.4f}")

# Generate predictions for classification report
y_true = []
y_pred = []
for images, labels in test_ds:
    predictions = model.predict(images)
    pred_labels = np.argmax(predictions, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(pred_labels)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Advanced Model Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(CLASS_NAMES))
plt.xticks(tick_marks, CLASS_NAMES, rotation=90)
plt.yticks(tick_marks, CLASS_NAMES)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('advanced_confusion_matrix.png')

# --- 6. Plot Training History ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')

# --- 7. Create a Feature Extraction Model ---
print("\nCreating feature extraction model...")

if use_transfer_learning:
    # For transfer learning model, we'll use the layer before the final Dense layer
    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-3].output  # Features before the final Dense layer
    )
else:
    # For custom model, we'll use the 512-neuron Dense layer
    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-3].output
    )

feature_extractor.summary()

# --- 8. TenSEAL Encryption and Visualization ---
def create_tenseal_context(poly_modulus_degree=8192, 
                          bit_scales=[40, 20, 20, 20],
                          security_level=128):
    """
    Create a TenSEAL context for encryption.
    
    Args:
        poly_modulus_degree: The polynomial modulus degree
        bit_scales: The bit-scales for the context
        security_level: Security level in bits
    
    Returns:
        A TenSEAL context
    """
    print("Creating TenSEAL context for encryption...")
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=bit_scales
    )
    context.global_scale = 2**bit_scales[0]
    context.generate_galois_keys()
    return context

def encrypt_vector(context, vector):
    """
    Encrypt a vector using TenSEAL.
    
    Args:
        context: TenSEAL context
        vector: NumPy array or list to encrypt
    
    Returns:
        Encrypted vector
    """
    return ts.ckks_vector(context, vector)

def decrypt_vector(encrypted_vector):
    """
    Decrypt a TenSEAL encrypted vector.
    
    Args:
        encrypted_vector: TenSEAL encrypted vector
    
    Returns:
        Decrypted vector as a list
    """
    return encrypted_vector.decrypt()

def visualize_encrypted_data(original_data, encrypted_data, decrypted_data, title="Encryption Visualization"):
    """
    Visualize original, encrypted (noise representation), and decrypted data.
    
    Args:
        original_data: Original data vector
        encrypted_data: TenSEAL encrypted vector
        decrypted_data: Decrypted data vector
        title: Plot title
    """
    plt.figure(figsize=(15, 8))
    
    # Plot original data
    plt.subplot(3, 1, 1)
    plt.plot(original_data, 'b-', label='Original Data')
    plt.title('Original Data')
    plt.grid(True)
    plt.legend()
    
    # For encrypted data, we can't directly plot the encrypted values
    # Instead, we'll show a representation of the encryption
    plt.subplot(3, 1, 2)
    # Generate a visual representation of encryption noise
    ciphertext_size = encrypted_data.size()
    encrypted_representation = np.random.normal(0, 1, len(original_data))
    plt.plot(range(len(encrypted_representation)), encrypted_representation, 'r-', marker='o', linestyle='None', label='Encrypted Data (Noise Representation)') 
    plt.title(f'Encrypted Data Representation (Actual Size: {ciphertext_size} bytes)')
    plt.grid(True)
    plt.legend()
    
    # Plot decrypted data
    plt.subplot(3, 1, 3)
    plt.plot(decrypted_data, 'g-', label='Decrypted Data')
    plt.title('Decrypted Data')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.savefig('encrypted_data_visualization.png')
    plt.close()  # Close the figure to free memory

def demo_tenseal_encryption():
    """
    Demonstrate TenSEAL encryption on feature vectors extracted from the model.
    """
    print("\nDemonstrating TenSEAL encryption with the advanced model...")
    
    # Get some test samples
    test_samples = next(iter(test_ds))[0][:5]  # Get 5 test images
    
    # Extract features using our feature extractor
    features = feature_extractor.predict(test_samples)
    
    # Create TenSEAL context
    context = create_tenseal_context()
    
    # Select one feature vector for demonstration
    feature_vector = features[0].flatten()[:20]  # Take first 20 elements for simplicity
    
    print(f"Original feature vector (first few elements): {feature_vector[:5]}...")
    
    # Encrypt the feature vector
    encrypted_vector = encrypt_vector(context, feature_vector)
    print(f"Encrypted vector size: {encrypted_vector.size()} bytes")
    
    # Decrypt the feature vector
    decrypted_vector = decrypt_vector(encrypted_vector)
    print(f"Decrypted feature vector (first few elements): {decrypted_vector[:5]}...")
    
    # Calculate mean squared error between original and decrypted
    mse = np.mean((feature_vector - decrypted_vector) ** 2)
    print(f"Mean squared error between original and decrypted: {mse:.8f}")
    
    # Visualize the process
    visualize_encrypted_data(feature_vector, encrypted_vector, decrypted_vector,
                           title="Feature Vector Encryption Visualization")
    
    return context, encrypted_vector, decrypted_vector

# --- 9. Save the models and components ---
print("\nSaving models and components...")
model.save(MODEL_SAVE_PATH)
feature_extractor.save(FEATURE_EXTRACTOR_PATH)

# Save other components like class names and scaler
components = {
    'class_names': CLASS_NAMES,
    'img_size': IMG_SIZE,
    'random_state': RANDOM_STATE,
}
joblib.dump(components, COMPONENTS_PATH)

# Demonstrate TenSEAL encryption if this file is run directly
if __name__ == "__main__":
    print("\nRunning TenSEAL encryption demonstration...")
    context, encrypted_data, decrypted_data = demo_tenseal_encryption()
    print("\nScript finished successfully.")  # Add a confirmation message