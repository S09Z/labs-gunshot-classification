import librosa
import numpy as np
import os
import tensorflow as tf
import keras
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


SAMPLE_RATE = 22050
DURATION = 4  # seconds
N_MELS = 64

def extract_mel(file_path, sr=SAMPLE_RATE, duration=DURATION, n_mels=N_MELS):
    y, _ = librosa.load(file_path, sr=sr, duration=duration, mono=True)
    if len(y) < sr * duration:
        y = np.pad(y, (0, sr * duration - len(y)))
    mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel)
    return log_mel

def load_dataset(data_dir: Path, class_map: dict, test_size=0.2):
    X = []
    y = []
    for label, index in class_map.items():
        folder = data_dir / label
        for file in folder.glob("*.wav"):
            mel = extract_mel(file)
            X.append(mel)
            y.append(index)
    X = np.array(X)[..., np.newaxis]
    y = to_categorical(y, num_classes=len(class_map))
    
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
