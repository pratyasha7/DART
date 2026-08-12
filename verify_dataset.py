import os
import pandas as pd
import librosa

# Paths to your extracted data
CSV_PATH = os.path.join("UrbanSound8K", "metadata", "UrbanSound8K.csv")
AUDIO_DIR = os.path.join("UrbanSound8K", "audio")

print("Step 1: Checking metadata path...")
if not os.path.exists(CSV_PATH):
    print(f"Error: Metadata file not found at {CSV_PATH}")
    exit(1)

# Load metadata
metadata = pd.read_csv(CSV_PATH)
print("Metadata loaded successfully!")

# Count of our two primary target classes
gunshot_count = len(metadata[metadata['classID'] == 6])
dog_bark_count = len(metadata[metadata['classID'] == 3])

print(f"\nDataset Statistics:")
print(f" - Gun Shot (Class 6) files: {gunshot_count}")
print(f" - Dog Bark (Class 3) files: {dog_bark_count}")

print("\nStep 2: Testing Librosa audio decoder...")
# Grab the first gunshot file in the dataset to verify loading
sample_row = metadata[metadata['classID'] == 6].iloc[0]
sample_file_path = os.path.join(AUDIO_DIR, f"fold{sample_row['fold']}", sample_row['slice_file_name'])

if os.path.exists(sample_file_path):
    try:
        # Load a short 1-second clip to verify reading capabilities
        audio, sample_rate = librosa.load(sample_file_path, sr=16000, duration=1.0)
        print(f"Successfully loaded: {sample_file_path}")
        print(f"Audio Sample Rate: {sample_rate}Hz")
        print(f"Data array size: {len(audio)} samples")
        print("\nSanity check passed! Your environment is ready for the ML training loop.")
    except Exception as e:
        print(f"Decoder Error: Failed to parse audio file. Details: {e}")
else:
    print(f"Error: Sample file not found at {sample_file_path}")