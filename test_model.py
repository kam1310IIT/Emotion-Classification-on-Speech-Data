import joblib
import librosa
import numpy as np
from xgboost import XGBClassifier

# Load the saved model
xgb_model = XGBClassifier()
def load_model(model_path="xgb_model.json"):
    model = XGBClassifier()
    model.load_model(model_path)
    return model

xgb_model.load_model()

# Emotion labels (change if needed)
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprise']

def extract_feature(file_path, n_mfcc=40, n_mels=128):
    y, sr = librosa.load(file_path, sr=None)
    
    # Resample to 16kHz for consistency
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Compute features
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels).T, axis=0)

    # Concatenate all
    return np.hstack([mfcc, chroma, mel])

def predict_emotion(audio_path):
    features = extract_feature(audio_path)
    features = features.reshape(1, -1)  # Reshape for model
    pred_idx = xgb_model.predict(features)[0]
    return emotion_labels[pred_idx]

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_model.py <audio_file_path>")
        exit()

    audio_file = sys.argv[1]
    emotion = predict_emotion(audio_file)
    print(f"Predicted emotion: {emotion}")
