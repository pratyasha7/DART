import os
import json
import time
import threading
import numpy as np
import tensorflow as tf
import librosa
import sounddevice as sd
import websocket
from collections import deque

# =====================================================================
# 1. LOCAL EDGE CONFIGURATION
# =====================================================================
MY_NODE_ID = "Sensor_A"
MY_X = 0.0
MY_Y = 0.0

HOLOCHAIN_APP_PORT = 8888 
WS_URL = f"ws://localhost:{HOLOCHAIN_APP_PORT}"

TFLITE_PATH = os.path.join("..", "gunshot_detector.tflite")
FEATURES_PATH = os.path.join("..", "X_features.npy")

SAMPLE_RATE = 16000  # Updated to 16kHz matching 128-band Mel Spectrograms
DURATION = 2.0       # 2 seconds
N_MELS = 128              
EXPECTED_FRAMES = 63
EXPECTED_FEATURES = N_MELS * EXPECTED_FRAMES  # 8,064 features
VOLUME_TRIGGER_THRESHOLD = 0.3

# =====================================================================
# 2. GLOBAL TFLITE INTERPRETER INITIALIZATION
# =====================================================================
print("[*] Allocating system memory and loading TFLite Interpreter globally...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero_point = input_details['quantization']
output_scale, output_zero_point = output_details['quantization']

# Load normalization statistics
X_train_raw = np.load(FEATURES_PATH)
X_train_flat = X_train_raw.reshape(X_train_raw.shape[0], -1)
GLOBAL_MEAN = X_train_flat.mean(axis=0)
GLOBAL_STD = X_train_flat.std(axis=0) + 1e-8
print("[SUCCESS] AI Engine initialized.\n")

# =====================================================================
# 3. HARDENED INT8 AI INFERENCE
# =====================================================================
def run_int8_inference(audio_samples):
    """Processes audio inputs and executes model evaluation safely."""
    try:
        # Extract 128-band Log-Mel Spectrogram
        melspec = librosa.feature.melspectrogram(y=audio_samples, sr=SAMPLE_RATE, n_mels=N_MELS)
        log_melspec = librosa.power_to_db(melspec, ref=1.0)

        # Padding / Truncation
        if log_melspec.shape[1] < EXPECTED_FRAMES:
            pad_width = EXPECTED_FRAMES - log_melspec.shape[1]
            log_melspec = np.pad(log_melspec, ((0, 0), (0, pad_width)), 'constant')
        else:
            log_melspec = log_melspec[:, :EXPECTED_FRAMES]

        # Flatten & Normalize
        features_float = log_melspec.flatten()
        features_normalized = (features_float - GLOBAL_MEAN) / GLOBAL_STD
        
        # INT8 Quantization
        features_int8 = np.clip(
            np.round(features_normalized / input_scale) + input_zero_point, 
            -128, 127
        ).astype(np.int8)
        
        # Reshape matching TFLite expected input shape (1D, 2D, or 4D)
        input_tensor = features_int8.reshape(input_details['shape'])
        interpreter.set_tensor(input_details['index'], input_tensor)
        interpreter.invoke()
        
        # Parse output based on model quantization spec
        quantized_output = interpreter.get_tensor(output_details['index'])[0][0]
        if output_details['dtype'] == np.int8:
            return float((quantized_output - output_zero_point) * output_scale)
        return float(quantized_output)
    except Exception as e:
        print(f"[AI ERROR] TinyML Inference Error: {e}")
        return 0.0

# =====================================================================
# 4. ASYNCHRONOUS HOLOCHAIN ZOME CALL
# =====================================================================
def make_holochain_zome_call(timestamp):
    """Establishes non-blocking connection to local Holochain Conductor."""
    try:
        ws = websocket.create_connection(WS_URL, timeout=3.0)
        
        zome_call_payload = {
            "type": "app_request",
            "data": {
                "cell_id": "acoustic_triangulation_cell_id",
                "zome_name": "acoustic_triangulation",
                "fn_name": "create_triangulation_claim",  
                "payload": {
                    "observations": [
                        {
                            "sensor_id": MY_NODE_ID,
                            "x_coord": MY_X,
                            "y_coord": MY_Y,
                            "t_arrival": timestamp
                        }
                    ]
                }
            }
        }
        
        ws.send(json.dumps(zome_call_payload))
        response = ws.recv()
        print(f"[NET] Response from Holochain Conductor: {response}")
        ws.close()
    except Exception as e:
        print(f"[NET ERROR] Failed to perform Zome Call on port {HOLOCHAIN_APP_PORT}: {e}")

# =====================================================================
# 5. AUDIO ACQUISITION & MONITORING LOOP
# =====================================================================
buffer_size = int(SAMPLE_RATE * DURATION)
audio_buffer = deque(maxlen=buffer_size)

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[Audio Status Warning]: {status}")
    audio_buffer.extend(indata[:, 0])

stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)

with stream:
    print(f"[*] Node {MY_NODE_ID} active at ({MY_X}m, {MY_Y}m). Listening...")
    
    # Warm up buffer
    while len(audio_buffer) < buffer_size:
        time.sleep(0.05)

    while True:
        # Prevent CPU spinning
        time.sleep(0.01)
        
        # Thread-safe snapshot copy
        buffer_snapshot = list(audio_buffer)
        recent_samples = buffer_snapshot[-int(SAMPLE_RATE * 0.1):]
        volume_norm = np.linalg.norm(recent_samples) / np.sqrt(len(recent_samples))
        
        if volume_norm > VOLUME_TRIGGER_THRESHOLD:
            arrival_time = time.time()
            print(f"\n[TRIGGER] Acoustic spike detected ({volume_norm:.3f}). Freezing buffer...")
            
            frozen_clip = np.array(buffer_snapshot)
            prediction = run_int8_inference(frozen_clip)
            
            if prediction > 0.5:
                print(f" -> AI: GUNSHOT CONFIRMED (Confidence: {prediction * 100:.2f}%)")
                # Threaded non-blocking network dispatch
                threading.Thread(
                    target=make_holochain_zome_call, 
                    args=(arrival_time,), 
                    daemon=True
                ).start()
            else:
                print(f" -> AI: Background ambient sound (Confidence: {prediction * 100:.2f}%). Ignored.")
            
            # Cooldown sleep
            time.sleep(1.0)