"""
ExoHunter Classification Module
Author: Sarthakk Anjariya

This module defines, trains, and evaluates two neural network architectures:
1. Model A (CNN-LSTM): Classifies folded phase curves into 4 categories:
   - transit (0)
   - eclipse (1)
   - blend (2)
   - other (3)
2. Model B (Autoencoder): Reconstructs folded phase curves to estimate an anomaly score
   (reconstruction MSE), useful for finding outliers or unclassified transits.

Astrophysical reasoning:
- Conv1D layers are highly effective at capturing local temporal shapes (e.g., U-shaped planetary
  transits vs V-shaped eclipsing binary dips).
- LSTM layers capture the sequential context and asymmetry in the folded curves.
- Phase-folded light curves are shift-invariant relative to the search epoch, but timing errors
  can shift the transit center slightly. We use phase-shifting data augmentation to make the
  model robust to epoch selection errors.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from detect import detect_candidates

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def build_cnn_lstm(input_shape=(200, 1)):
    """
    Build the CNN-LSTM classifier (Model A).
    Conv1D(64,3,relu) -> MaxPool1D(2) -> Conv1D(128,3,relu) -> MaxPool1D(2) ->
    LSTM(64) -> Dense(64,relu) -> Dropout(0.3) -> Dense(4,softmax)
    """
    model = models.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.LSTM(64),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4, activation='softmax')
    ], name="CNN_LSTM_Classifier")
    return model

def build_autoencoder(input_shape=(200, 1)):
    """
    Build the Autoencoder anomaly detector (Model B).
    Encoder: Conv1D(32) -> Conv1D(16) -> Flatten -> Dense(8)
    Decoder: Dense(800) -> Reshape((50, 16)) -> Conv1DTranspose(32) -> Conv1DTranspose(1)
    """
    # Encoder
    encoder_inputs = Input(shape=input_shape)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(encoder_inputs)
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    latent = layers.Dense(8, activation='relu', name="latent_bottleneck")(x)
    
    # Decoder
    # To get back to (200, 1) using Conv1DTranspose with stride=2:
    # Stride 2 doubles length. Starting from length 50, two Transpose layers output 200.
    # So Dense projects latent space to 50 * 16 = 800 channels.
    x = layers.Dense(50 * 16, activation='relu')(latent)
    x = layers.Reshape((50, 16))(x)
    x = layers.Conv1DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    decoder_outputs = layers.Conv1DTranspose(1, 3, strides=2, padding='same', activation='linear')(x)
    
    model = models.Model(encoder_inputs, decoder_outputs, name="Transit_Autoencoder")
    return model

def augment_data(X, y, factor=10):
    """
    Augment folded phase curves to enlarge the dataset.
    Augmentation techniques:
    1. Phase roll (simulate timing/epoch shifts)
    2. Add Gaussian noise (simulate different SNR conditions)
    3. Depth scaling (simulate different planetary sizes)
    """
    X_aug = []
    y_aug = []
    
    for i in range(len(X)):
        curve = X[i]
        label = y[i]
        
        # Add original
        X_aug.append(curve)
        y_aug.append(label)
        
        for _ in range(factor - 1):
            aug_curve = curve.copy()
            
            # 1. Phase roll (shift by up to 5 bins left/right)
            shift = np.random.randint(-5, 6)
            aug_curve = np.roll(aug_curve, shift, axis=0)
            
            # 2. Depth scaling (only if it is a transit/eclipse/blend)
            if label in [0, 1, 2]:
                scale = np.random.uniform(0.8, 1.2)
                aug_curve = aug_curve * scale
                
            # 3. Add small random Gaussian noise (scaled by 1000 to match)
            noise = np.random.normal(0, 0.5, size=aug_curve.shape)
            aug_curve = aug_curve + noise
            
            X_aug.append(aug_curve)
            y_aug.append(label)
            
    return np.array(X_aug, dtype=np.float32), np.array(y_aug, dtype=np.int32)

def extract_folded_curves(processed_dir="data/processed"):
    """
    Load preprocessed 4000-pt curves, run BLS on each to find the best candidate period,
    and return the 200-pt binned folded curves along with their labels and TIC IDs.
    """
    X_raw = np.load(os.path.join(processed_dir, "X.npy"))
    y = np.load(os.path.join(processed_dir, "y.npy"))
    times = np.load(os.path.join(processed_dir, "times.npy"))
    tics = np.load(os.path.join(processed_dir, "tics.npy"), allow_pickle=True)
    
    X_folded = []
    
    logging.info(f"Extracting folded curves for {len(X_raw)} stars...")
    for i in range(len(X_raw)):
        flux = X_raw[i, :, 0]
        time = times[i]
        
        # Find candidates using BLS (from detect.py)
        candidates = detect_candidates(time, flux)
        
        # Take the folded flux of the best candidate (highest SNR)
        # Note: detect_candidates returns candidates sorted by SNR descending
        best_cand = candidates[0]
        X_folded.append((best_cand["folded_flux"] - 1.0) * 1000.0)
        
    X_folded = np.array(X_folded, dtype=np.float32) # shape (N, 200, 1)
    return X_folded, y, tics

def train_and_evaluate(processed_dir="data/processed", models_dir="models"):
    """
    Orchestrate the full training and evaluation pipeline for both models.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Extract folded curves
    X, y, tics = extract_folded_curves(processed_dir)
    
    # 2. Split dataset into 80/20 train/test split
    # Stratified split ensures equal class representation in train and test sets
    X_train, X_val, y_train, y_val, tics_train, tics_val = train_test_split(
        X, y, tics, test_size=0.2, random_state=42, stratify=y
    )
    
    logging.info(f"Split sizes - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
    
    # 3. Augment training set to make it robust
    X_train_aug, y_train_aug = augment_data(X_train, y_train, factor=10)
    logging.info(f"Augmented Train size: {X_train_aug.shape[0]}")
    
    # --- Train Model A: CNN-LSTM ---
    logging.info("Building and compiling CNN-LSTM Classifier...")
    clf = build_cnn_lstm()
    clf.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callback to prevent overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    
    logging.info("Training CNN-LSTM Classifier...")
    clf.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate CNN-LSTM
    val_preds = clf.predict(X_val)
    val_pred_labels = np.argmax(val_preds, axis=1)
    
    label_names = ["transit", "eclipse", "blend", "other"]
    print("\n" + "="*50)
    print("CNN-LSTM CLASSIFIER REPORT ON VALIDATION SET")
    print("="*50)
    print(classification_report(y_val, val_pred_labels, target_names=label_names))
    
    # ROC-AUC per class
    # Convert y_val to one-hot encoding for multiclass ROC-AUC
    y_val_one_hot = tf.keras.utils.to_categorical(y_val, num_classes=4)
    roc_auc = roc_auc_score(y_val_one_hot, val_preds, multi_class='ovr')
    print(f"Overall Multiclass ROC-AUC (OVR): {roc_auc:.4f}")
    print("="*50 + "\n")
    
    # --- Train Model B: Autoencoder ---
    logging.info("Building and compiling Autoencoder anomaly detector...")
    ae = build_autoencoder()
    ae.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse'
    )
    
    # Train Autoencoder on all training samples to learn general curves
    logging.info("Training Autoencoder...")
    ae_early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    ae.fit(
        X_train_aug, X_train_aug, # inputs and targets are the same for autoencoder
        validation_data=(X_val, X_val),
        epochs=80,
        batch_size=32,
        callbacks=[ae_early_stop],
        verbose=1
    )
    
    # Save the models
    clf_path = os.path.join(models_dir, "clf_model.keras")
    ae_path = os.path.join(models_dir, "ae_model.keras")
    
    clf.save(clf_path)
    ae.save(ae_path)
    logging.info(f"Models successfully saved to {clf_path} and {ae_path}")
    
    # Log some example reconstruction errors (anomaly scores)
    val_reconstructed = ae.predict(X_val)
    errors = np.mean(np.square(X_val - val_reconstructed), axis=(1, 2))
    for i in range(min(5, len(y_val))):
        logging.info(f"Val target {tics_val[i]} ({label_names[y_val[i]]}) - Anomaly Score (MSE): {errors[i]:.6f}")
        
    return clf, ae

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    train_and_evaluate(processed_dir=processed_dir, models_dir=models_dir)
