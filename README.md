# Parkinson's Disease Detection using Handwriting and Voice Recordings ✍️🎙️

This project presents a multi-modal AI-based approach to detect **Parkinson’s Disease** through analysis of **handwritten patterns** and **voice recordings**. By leveraging features extracted from both modalities, the model aims to provide early, accurate, and non-invasive diagnosis support for medical professionals.

## 📌 Project Overview

- **Objective:** Detect Parkinson’s Disease by analyzing handwriting images and voice recordings using ML and DL techniques.
- **Data Modalities:**
  - **Handwriting** – Spiral drawings and meander patterns.
  - **Voice** – Acoustic features from sustained phonation recordings.
- **Tools & Technologies:** Python, TensorFlow, Keras, librosa, OpenCV, Scikit-learn, NumPy, Matplotlib.

## 🧠 Features

- Image preprocessing for handwritten spirals (binarization, edge detection).
- Audio preprocessing using `librosa` (MFCC, pitch, jitter, shimmer).
- Machine learning models: SVM, Random Forest, Logistic Regression.
- Deep learning CNN model for handwriting classification.
- Comparative evaluation of ML vs DL performance.
- Exploratory data analysis and visualization.

## 📂 Project Structure

```
parkinsons-detection-handwritinNotes-voiceRecordings/
│
├── dataset/
│   ├── handwriting/               # Handwriting spiral images
│   └── voice/                     # Audio recordings (WAV format)
│
├── handwriting_detection/
│   └── handwriting_cnn_model.ipynb   # CNN model for handwriting classification
│
├── voice_detection/
│   ├── extract_features.py          # Voice feature extraction using librosa
│   └── voice_ml_models.ipynb        # ML models (SVM, RF, LR) for voice classification
│
├── combined_model.ipynb             # Fusion-based prediction combining both modalities
├── visuals/                         # Graphs, charts, and confusion matrices
└── README.md                        # Project documentation
```

## 🔬 Methodology

### Handwriting Analysis
- Grayscale conversion and thresholding
- Feature extraction with CNN
- Binary classification: Parkinson vs Healthy

### Voice Analysis
- Audio normalization
- MFCC, jitter, shimmer, harmonic-to-noise ratio
- ML algorithms with cross-validation and performance evaluation

## 📊 Results

- **Handwriting CNN Accuracy:** ~92%
- **Voice ML Accuracy:** ~85% (best with SVM)
- **Fusion Accuracy:** ~94% combining handwriting + voice predictions

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aarya-04/parkinsons-detection-handwritinNotes-voiceRecordings.git
   cd parkinsons-detection-handwritinNotes-voiceRecordings
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks:**

   - For handwriting:
     ```bash
     jupyter notebook handwriting_detection/handwriting_cnn_model.ipynb
     ```

   - For voice:
     ```bash
     jupyter notebook voice_detection/voice_ml_models.ipynb
     ```

   - For combined model:
     ```bash
     jupyter notebook combined_model.ipynb
     ```

## 📚 Dependencies

- `tensorflow`
- `keras`
- `librosa`
- `opencv-python`
- `scikit-learn`
- `numpy`
- `matplotlib`
- `pandas`

## 📈 Sample Outputs

- Confusion matrices for all models
- Accuracy/loss training graphs
- Bar plots comparing individual and combined model performance

## 🧾 License

This project is open-source and available under the [MIT License](LICENSE).

## 🙌 Acknowledgements

- Data sourced from public datasets (UCI, Kaggle, or medical research archives)
- Inspired by real-world biomedical applications in neurodegenerative disease detection

## 👨‍💻 Author

**Aarya Kulkarni**  
[LinkedIn](https://www.linkedin.com/in/aaryakulkarni03) • [GitHub](https://github.com/Aarya-04)

---

> Combining handwriting and voice biomarkers, we push the boundaries of early Parkinson’s detection through AI.
