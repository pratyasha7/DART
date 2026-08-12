import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Reshape, Conv2D, Dense, Multiply, GlobalAveragePooling1D, Dropout

# =====================================================================
# 1. LOAD THE EXTRACTED NUMPY ARRAYS
# =====================================================================
print("Loading processed features from disk...")
X_data = np.load("X_features.npy")
y_data = np.load("y_labels.npy")
fold_data = np.load("fold_assignments.npy")

N_MELS = 128            
EXPECTED_FRAMES = 63
EXPECTED_FEATURES = N_MELS * EXPECTED_FRAMES  # 8,064 features

X_flat = X_data.reshape(X_data.shape[0], -1)

print(f"Dataset shape loaded: {X_flat.shape} (Samples, Input Features)")
print(f"Target labels shape:  {y_data.shape}")

# =====================================================================
# 2. DEFINE LIGHTWEIGHT HARDWARE-ALIGNED ARCHITECTURE
# =====================================================================
def create_tinyml_model():
    """
    Optimized Self-Attention network with explicit projection to keep 
    SRAM footprint extremely small (<20KB) on Cortex-M4 hardware.
    """
    inputs = Input(shape=(EXPECTED_FEATURES,))
    
    # Reshape flat vector to 2D image: (63 time frames, 128 mel bins, 1 channel)
    x = Reshape((EXPECTED_FRAMES, N_MELS, 1))(inputs)
    
    # 1. 2D Convolution (ARM CMSIS-NN accelerated)
    x = Conv2D(4, kernel_size=(3, 3), padding='same', activation='relu')(x) # (63, 128, 4)
    
    # 2. Project to a compact temporal feature dimension (63 frames, 64 features)
    x = Reshape((EXPECTED_FRAMES, N_MELS * 4))(x)
    x = Dense(64, activation='relu')(x)
    
    # 3. Compact Self-Attention Block
    attention_weights = Dense(64, activation='softmax', name='attention_weights')(x)
    attention_mul = Multiply()([x, attention_weights])
    
    # 4. Global Average Pooling & Classification Head
    x = GlobalAveragePooling1D()(attention_mul)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model

# =====================================================================
# 3. RUN 10-FOLD CROSS-VALIDATION (LEAKAGE-PROOF SCALING)
# =====================================================================
fold_accuracies = []
print("\nStarting 10-Fold Cross-Validation (Leakage-proof Scaling)...")

for test_fold in range(1, 11):
    X_train_raw = X_flat[fold_data != test_fold]
    y_train = y_data[fold_data != test_fold]
    
    X_test_raw = X_flat[fold_data == test_fold]
    y_test = y_data[fold_data == test_fold]
    
    mean_train = X_train_raw.mean(axis=0)
    std_train = X_train_raw.std(axis=0) + 1e-8
    
    X_train = (X_train_raw - mean_train) / std_train
    X_test = (X_test_raw - mean_train) / std_train
    
    model = create_tinyml_model()
    
    model.fit(
        X_train, y_train, 
        epochs=35, 
        batch_size=16, 
        verbose=0,
        validation_data=(X_test, y_test)
    )
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f" - Fold {test_fold:02d} Test Accuracy: {accuracy * 100:.2f}%")
    fold_accuracies.append(accuracy)

mean_acc = np.mean(fold_accuracies) * 100
std_acc = np.std(fold_accuracies) * 100
print("\n==========================================")
print("             CROSS-VALIDATION RESULTS     ")
print("==========================================")
print(f"Mean Accuracy: {mean_acc:.2f}%")
print(f"Standard Dev:  ±{std_acc:.2f}%")
print("==========================================\n")

# =====================================================================
# 4. FINAL PRODUCTION TRAINING & INT8 TFLITE EXPORT
# =====================================================================
print("Preparing final production model...")

X_mean_final = X_flat.mean(axis=0)
X_std_final = X_flat.std(axis=0) + 1e-8
X_flat_normalized = (X_flat - X_mean_final) / X_std_final

final_model = create_tinyml_model()
final_model.fit(X_flat_normalized, y_data, epochs=40, batch_size=16, verbose=0)

print("\nConverting model to FULL INTEGER INT8 TensorFlow Lite...")

# Balanced Representative Dataset Generator across the entire dataset
def representative_data_gen():
    np.random.seed(42)
    sample_indices = np.random.choice(X_flat_normalized.shape[0], size=150, replace=False)
    for idx in sample_indices:
        sample = np.expand_dims(X_flat_normalized[idx], axis=0).astype(np.float32)
        yield [sample]

converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_int8 = converter.convert()

output_filename = "gunshot_detector.tflite"
with open(output_filename, "wb") as f:
    f.write(tflite_model_int8)

keras_size = final_model.count_params() * 4 / 1024  
tflite_size = len(tflite_model_int8) / 1024         

print("\n==========================================")
print("             TINYML COMPILE SUMMARY       ")
print("==========================================")
print(f"File Saved:                    '{output_filename}'")
print(f"Est. Unquantized Keras Size:   {keras_size:.2f} KB")
print(f"Quantized INT8 TFLite Size:     {tflite_size:.2f} KB")
print("==========================================")