import os
import json
import time
import threading
import numpy as np
import tensorflow as tf
import librosa
import sounddevice as sd
import websocket  # pip install websocket-client
from collections import deque

MY_NODE_ID = "Sensor_A"
MY_X = 0.0
MY_Y = 0.0

HOLOCHAIN_APP_PORT = 8888 
WS_URL = f"ws://127.0.0.1:{HOLOCHAIN_APP_PORT}"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

TFLITE_PATH = os.path.join(PROJECT_ROOT, "gunshot_detector.tflite")

MEAN_PATH = os.path.join(PROJECT_ROOT, "mean.npy")
STD_PATH = os.path.join(PROJECT_ROOT, "std.npy")

SAMPLE_RATE = 16000 
DURATION = 4.0      
HOP_LENGTH = 512
N_FFT = 1024
N_MELS = 128              
EXPECTED_FRAMES = 126
EXPECTED_FEATURES = N_MELS * EXPECTED_FRAMES  # 16,128 features

VOLUME_TRIGGER_THRESHOLD = 0.015 
CONFIDENCE_THRESHOLD = 0.75       

print("[*] Allocating system memory and loading TFLite Interpreter globally...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero_point = input_details['quantization']
output_scale, output_zero_point = output_details['quantization']

GLOBAL_MEAN = np.load(MEAN_PATH)
GLOBAL_STD = np.load(STD_PATH)
print("[SUCCESS] AI Engine initialized. Operational footprint optimized.\n")

def run_int8_inference(audio_samples):
    """Processes 4.0s audio inputs and executes model evaluation safely."""
    try:
        # Scale only if signal exceeds full scale (> 1.0) to avoid boosting background noise
        peak_val = np.max(np.abs(audio_samples))
        if peak_val < 1e-4:
            return 0.0
        if peak_val > 1.0:
            audio_samples = audio_samples / peak_val

        melspec = librosa.feature.melspectrogram(
            y=audio_samples, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        log_melspec = librosa.power_to_db(melspec, ref=1.0)

        if log_melspec.shape[1] < EXPECTED_FRAMES:
            pad_width = EXPECTED_FRAMES - log_melspec.shape[1]
            log_melspec = np.pad(log_melspec, ((0, 0), (0, pad_width)), 'constant')
        else:
            log_melspec = log_melspec[:, :EXPECTED_FRAMES]

        features_float = log_melspec.flatten()
        features_normalized = (features_float - GLOBAL_MEAN) / GLOBAL_STD

        features_int8 = np.clip(
            np.round(features_normalized / input_scale) + input_zero_point, 
            -128, 127
        ).astype(np.int8)

        interpreter.set_tensor(input_details['index'], features_int8.reshape(input_details['shape']))
        interpreter.invoke()

        raw_output = interpreter.get_tensor(output_details['index'])[0][0]
        if output_details['dtype'] in (np.int8, np.uint8):
            confidence = (float(raw_output) - float(output_zero_point)) * float(output_scale)
        else:
            confidence = float(raw_output)

        return float(np.clip(confidence, 0.0, 1.0))
    except Exception as e:
        print(f"\n[AI ERROR] TinyML Inference Error: {e}")
        return 0.0

def make_holochain_zome_call(timestamp):
    """Establishes non-blocking connection to local Holochain Conductor."""
    try:
        ws = websocket.create_connection(WS_URL, timeout=3.0)
        
        zome_call_payload = {
            "type": "app_request",
            "data": {
                "cell_id": "acoustic_app",
                "zome_name": "acoustic_app",
                "fn_name": "create_triangulation_claim",  
                "payload": {
                    "observations": [
                        {"sensor_id": "Sensor_A", "x_coord": 0.0, "y_coord": 0.0, "t_arrival": timestamp},
                        {"sensor_id": "Sensor_B", "x_coord": 100.0, "y_coord": 0.0, "t_arrival": timestamp + 0.20},
                        {"sensor_id": "Sensor_C", "x_coord": 50.0, "y_coord": 86.6, "t_arrival": timestamp + 0.20}
                    ]
                }
            }
        }
        
        ws.send(json.dumps(zome_call_payload))
        response = ws.recv()
        print(f"\n[NET SUCCESS] Response from Holochain Conductor: {response}")
        ws.close()
    except Exception as e:
        print(f"\n[NET ERROR] Failed to perform Zome Call on port {HOLOCHAIN_APP_PORT}: {e}")

def start_listener():
    """Starts the real-time continuous listening and detection loop."""
    buffer_size = int(SAMPLE_RATE * DURATION)
    audio_buffer = deque(maxlen=buffer_size)
    buffer_lock = threading.Lock()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"\n[Audio Status Warning]: {status}")
        with buffer_lock:
            audio_buffer.extend(indata[:, 0])

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)

    with stream:
        print(f"[*] Node {MY_NODE_ID} active at ({MY_X}m, {MY_Y}m). Listening...\n")

        while len(audio_buffer) < buffer_size:
            time.sleep(0.05)

        while True:
            time.sleep(0.01)

            with buffer_lock:
                buffer_snapshot = list(audio_buffer)
                
            recent_samples = buffer_snapshot[-int(SAMPLE_RATE * 0.1):]
            volume_norm = np.linalg.norm(recent_samples) / np.sqrt(len(recent_samples))

            bar = '#' * int(volume_norm * 300)
            print(f"\r[Listening] Mic Level: {volume_norm:.4f} |{bar:<30}|", end="", flush=True)
            
            if volume_norm > VOLUME_TRIGGER_THRESHOLD:
                arrival_time = time.time()
                print(f"\n[TRIGGER] Acoustic spike detected ({volume_norm:.4f}). Evaluating AI model...")
                
                frozen_clip = np.array(buffer_snapshot)
                prediction = run_int8_inference(frozen_clip)
                
                if prediction >= CONFIDENCE_THRESHOLD:
                    print(f" -> AI: GUNSHOT CONFIRMED (Confidence: {prediction * 100:.2f}%)")
                    threading.Thread(
                        target=make_holochain_zome_call, 
                        args=(arrival_time,), 
                        daemon=True
                    ).start()
                    time.sleep(1.2)  
                else:
                    print(f" -> AI: Background ambient sound (Confidence: {prediction * 100:.2f}%). Ignored.")
                    time.sleep(0.05)


if __name__ == "__main__":
    start_listener() 