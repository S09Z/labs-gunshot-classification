# 🔫 Gunshot Sound Classification

This project uses **deep learning** to classify gunshot sounds into categories like `handgun`, `rifle`, and `shotgun`. It is built with **TensorFlow**, **librosa**, **MLflow**, and follows best practices using the **Poetry** package manager and `src/` layout.

It includes support for:

- ✅ Preprocessing `.wav` audio with log-mel spectrograms
- ✅ CNN-based model training and evaluation
- ✅ MLflow logging and experiment tracking
- ✅ Support for multiple datasets (e.g. UrbanSound8K, FSD50K)
- ✅ Easy extensibility for adding more classes or data

---

## 📁 Project Structure
```
labs-gunshot-classification/
├── audio_data/ 
├── notebooks/ 
│ └── download_datasets.ipynb
├── src/
│ └── labs_gunshot_classification/
│ ├── init.py
│ ├── config.py 
│ ├── preprocess.py 
│ ├── model.py
│ ├── train.py 
│ ├── evaluate.py
│ └── utils.py 
├── README.md
├── pyproject.toml
└── poetry.lock
```
---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/yourname/labs-gunshot-classification.git
cd labs-gunshot-classification