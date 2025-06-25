# 🎧 Audio Emotion Classification using XGBoost

This project implements a complete pipeline for classifying emotions from speech using machine learning. It includes a fully functional *Streamlit web app* that allows users to upload .wav audio files and get real-time emotion predictions using a pre-trained *XGBoost* model.

---

## 🚀 Features

- Upload a .wav file through the web interface
- Extracts MFCC, Chroma, and Mel Spectrogram features
- Predicts emotion using a trained XGBoost model
- Displays the predicted emotion in a clean UI
- Lightweight and fast — suitable for real-time demos

---

## 🎯 Emotions Recognized

The model classifies speech into the following 8 emotions:

- 😠 Angry  
- 😌 Calm  
- 🤢 Disgust  
- 😨 Fearful  
- 😄 Happy  
- 😐 Neutral  
- 😢 Sad  
- 😲 Surprise  

---

## 🧠 Model Details

- Model: XGBClassifier from xgboost
- Trained on MFCC, Chroma, and Mel-spectrogram features
- Input: .wav audio sampled at 16 kHz
- Saved as: xgb_model.json

---

## 📁 Project Structure
README.md
app.py
kamranjeet_emotion_classification.ipynb
requirements.txt
xgb_model.json

---

## Coefficient matrix
![image](https://github.com/user-attachments/assets/f168cd39-6732-4acc-a3d8-2ef6269bd671)
