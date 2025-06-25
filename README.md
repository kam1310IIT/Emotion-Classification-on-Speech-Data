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

## 📊 Coefficient Matrix

![image](https://github.com/user-attachments/assets/f168cd39-6732-4acc-a3d8-2ef6269bd671)

---

## 📈 Classification Results

**XGBoost Classification Report:**

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.855     | 0.891  | 0.87     | 38.0    |
| 1     | 0.949     | 0.969  | 0.96     | 38.0    |
| 2     | 0.749     | 0.943  | 0.83     | 38.0    |
| 3     | 0.880     | 0.898  | 0.89     | 39.0    |
| 4     | 0.827     | 0.744  | 0.78     | 39.0    |
| 5     | 0.966     | 0.759  | 0.85     | 19.0    |
| 6     | 0.739     | 0.680  | 0.71     | 38.0    |
| 7     | 0.763     | 0.718  | 0.74     | 39.0    |
|       |           |        |          |         |
| **Accuracy**     |        |          | **0.83** |         |
| **Macro Avg**    | 0.841  | 0.825    | 0.83     | 288.0   |
| **Weighted Avg** | 0.833  | 0.829    | 0.83     | 288.0   |

---

## 🌐 Streamlit App

Here is the link to the live web app:  
👉 **[Click to try the app](https://emotion-classification-on-speech-data-b9tmuggblrfdxmlbub68dq.streamlit.app/)**
