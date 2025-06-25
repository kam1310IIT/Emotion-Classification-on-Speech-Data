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

---

## Classification results:
XGBoost Classification Report:

             precision    recall  f1-score support
0                0.855  0.890526      0.87    38.0
1             0.949231  0.969474      0.96    38.0
2             0.748627  0.943158      0.83    38.0
3                 0.88  0.897949      0.89    39.0
4             0.827059  0.744103      0.78    39.0
5             0.965714  0.758947      0.85    19.0
6             0.738824      0.68      0.71    38.0
7             0.763333  0.718462      0.74    39.0
accuracy                              0.83        
macro avg     0.840974  0.825327      0.83   288.0
weighted avg  0.832562  0.829306      0.83   288.0

---

## Streamlit app
Here is the link to the web app
https://emotion-classification-on-speech-data-b9tmuggblrfdxmlbub68dq.streamlit.app/
