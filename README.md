# 🎙️ SHEild – Voice-Based Distress Detection

**SHEild** is a deep learning–powered voice distress detection system developed as a Master's final year project to enhance real-time safety applications. The model is capable of identifying distress in human speech using audio features extracted from publicly available emotional speech datasets.

> 🧠 Built using CNN-BiLSTM architecture  
> 🎯 Accuracy: 82.78% on test data  
> 📁 Datasets used: CREMA-D, RAVDESS, TESS  

---

## 📌 Objective

To develop a machine learning model that accurately detects distress in voice recordings, especially for use in emergency and women’s safety contexts, even when the speaker cannot trigger a manual alert.

---

## 🔍 Features

- Binary classification: **Distress** vs **Not Distress**
- MFCC-based audio feature extraction
- CNN-BiLSTM model for capturing spatial and temporal features
- Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
- Trained on ~7000 curated samples

---

## 🧪 Datasets

- [CREMA-D](https://zenodo.org/record/3816440)
- [RAVDESS](https://zenodo.org/record/1188976)
- [TESS](https://tspace.library.utoronto.ca/handle/1807/24487)

These datasets cover a wide range of emotions and speakers, with a focus on distress-related emotional patterns.

---

## ⚙️ Technologies Used

- **Python**
- **TensorFlow/Keras**
- **Librosa** for audio preprocessing
- **Scikit-learn** for evaluation
- **Matplotlib/Seaborn** for visualization

---

## 🧩 Model Architecture

A hybrid **CNN-BiLSTM** model:
- CNN layers: Extract frequency-domain features from MFCC spectrograms.
- BiLSTM layers: Capture time-dependent emotional variations.
- Dropout and regularization to reduce overfitting.

---

## 📈 Results

| Metric     | Distress | Not Distress |
|------------|----------|--------------|
| Precision  | 0.82     | 0.83         |
| Recall     | 0.59     | 0.94         |
| F1-Score   | 0.69     | 0.88         |
| Accuracy   | **82.78%** overall     |

- **AUC-ROC**: High area under curve indicating good class separation.
- **Confusion Matrix**: Visualized and analyzed for false positives/negatives.

---

## 🧪 How to Run

```bash
# Clone the repository
git clone https://github.com/nbr9097/SHEild-app.git
cd SHEild-app

# Install required libraries
pip install -r requirements.txt

# Run the Jupyter Notebook
jupyter notebook FinalSHEild.ipynb
```

---

## 📂 Project Structure

```plaintext
SHEild-app/
├── FinalSHEild.ipynb        # Main notebook with model, preprocessing, and evaluation
├── data/                    # Placeholder for dataset
├── models/                  # Saved model checkpoints
├── README.md
└── requirements.txt
```

---

## 📌 Future Enhancements

- Expand dataset with real-world audio
- Integrate Transformer or attention mechanisms
- Deploy as a mobile app or web API
- Add real-time noise filtering and multilingual support

---

## 📜 License

This project is for academic and research purposes. For commercial use, please contact the author.

---

## 👩‍💻 Author

**Noof Abdul Raheem A P**  
Master of Computer Science, St. Joseph’s College (Autonomous), Devagiri  
📧 [GitHub Profile](https://github.com/nbr9097)
