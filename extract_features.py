import os
import pandas as pd
import numpy as np
import librosa

# Configuration
CSV_PATH = os.path.join("UrbanSound8K", "metadata", "UrbanSound8K.csv")
AUDIO_DIR = os.path.join("UrbanSound8K", "audio")
TARGET_SR = 16000     # Target sample rate (16kHz)
DURATION = 2.0        # Target audio duration in seconds
N_MELS = 128          # UPGRADED: 128 Mel bands (Replaces N_MFCC = 13)
EXPECTED_FRAMES = 63  # Standard time-frame count for 32,000 samples

# Validate baseline files exist
if not os.path.exists(CSV_PATH):
    print(f"Error: Metadata file not found at {CSV_PATH}")
    exit(1)

print("Loading metadata...")
metadata = pd.read_csv(CSV_PATH)

# =====================================================================
# 1. LEAKAGE-PROOF WITHIN-FOLD BALANCED SAMPLING
# =====================================================================
print("Performing within-fold balanced sampling...")

balanced_records = []

# Loop through each of the 10 official, predefined folds
for fold_id in range(1, 11):
    fold_subset = metadata[metadata['fold'] == fold_id]
    
    # Isolate gunshots in this fold (Class ID 6)
    fold_gunshots = fold_subset[fold_subset['classID'] == 6].copy()
    fold_gunshots['label'] = 1
    n_gunshots = len(fold_gunshots)
    
    # Isolate ambient city sounds in this fold (All other classes)
    fold_non_gunshots = fold_subset[fold_subset['classID'] != 6].copy()
    
    if n_gunshots == 0:
        continue
        
    # Sample matching counts of background sounds strictly from the SAME fold
    if len(fold_non_gunshots) >= n_gunshots:
        sampled_non_gunshots = fold_non_gunshots.sample(n=n_gunshots, random_state=42)
    else:
        sampled_non_gunshots = fold_non_gunshots
        
    sampled_non_gunshots['label'] = 0
    
    balanced_records.append(fold_gunshots)
    balanced_records.append(sampled_non_gunshots)

# Combine elements into a single metadata reference
balanced_metadata = pd.concat(balanced_records).reset_index(drop=True)
print(f"Dataset Balanced! Collected {len(balanced_metadata)} records across 10 folds.")

# =====================================================================
# 2. FEATURE EXTRACTION: 128-BAND LOG-MEL SPECTROGRAM
# =====================================================================
def extract_log_mel_spectrogram(file_path):
    """Loads file, normalizes samples to 2s, extracts 128-band Log-Mel Spectrogram."""
    try:
        # 1. Load audio resampled to target rate
        audio, sr = librosa.load(file_path, sr=TARGET_SR, duration=DURATION)
        
        # Enforce exact waveform size (32,000 samples for 2.0s @ 16kHz)
        total_samples = int(TARGET_SR * DURATION)
        if len(audio) < total_samples:
            audio = np.pad(audio, (0, total_samples - len(audio)), 'constant')
        else:
            audio = audio[:total_samples]
            
        # 2. Extract 128-band Mel Spectrogram
        melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
        
        # 3. Convert power spectrogram to decibel units (Log-Mel) with static ref
        log_melspec = librosa.power_to_db(melspec, ref=1.0)
        
        # 4. Enforce exact frame width (63 time columns)
        if log_melspec.shape[1] < EXPECTED_FRAMES:
            pad_width = EXPECTED_FRAMES - log_melspec.shape[1]
            log_melspec = np.pad(log_melspec, ((0, 0), (0, pad_width)), 'constant')
        elif log_melspec.shape[1] > EXPECTED_FRAMES:
            log_melspec = log_melspec[:, :EXPECTED_FRAMES]
            
        return log_melspec
    except Exception as e:
        return None

# =====================================================================
# 3. ROBUST EXTRACTION LOOP & ERROR LOGGING
# =====================================================================
X_features = []
y_labels = []
folds = []
failed_files = []

print("\nStarting 128-band Log-Mel Spectrogram feature extraction loop...")
for index, row in balanced_metadata.iterrows():
    file_path = os.path.join(AUDIO_DIR, f"fold{row['fold']}", row['slice_file_name'])
    
    log_melspec = extract_log_mel_spectrogram(file_path)
    if log_melspec is not None:
        X_features.append(log_melspec)
        y_labels.append(row['label'])
        folds.append(row['fold'])
    else:
        failed_files.append((file_path, row['slice_file_name']))
        
    if (index + 1) % 100 == 0:
        print(f"Processed {index + 1}/{len(balanced_metadata)} files...")

# Safe array conversion (Guaranteed homogeneous shapes)
X_data = np.array(X_features, dtype=np.float32)
y_data = np.array(y_labels, dtype=np.int32)
fold_data = np.array(folds, dtype=np.int32)

print("\n==========================================")
print("             EXTRACTION SUMMARY           ")
print("==========================================")
print(f"Successful extractions: {len(X_data)}")
print(f"Failed extractions:     {len(failed_files)}")
print(f"Features matrix shape:  {X_data.shape} (Expected: [Samples, {N_MELS}, {EXPECTED_FRAMES}])")
print("==========================================")

if len(failed_files) > 0:
    print("\nWarning: The following files failed to process during extraction:")
    for path, name in failed_files[:10]:
        print(f" - {name} (Path: {path})")
    if len(failed_files) > 10:
        print(f" ... and {len(failed_files) - 10} more.")

# Save updated binaries
np.save("X_features.npy", X_data)
np.save("y_labels.npy", y_data)
np.save("fold_assignments.npy", fold_data)
print("\nNumPy features (128 Log-Mel) saved successfully to workspace disk.")