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

def load_dataset(split_dir, width=128):
    files = list(Path(split_dir).glob("*.npy"))
    X, y = [], []
    for f in files:
        data = np.load(f, allow_pickle=True).item()
        feat = data["features"]
        if feat.shape[1] < width:
            pad = width - feat.shape[1]
            feat = np.pad(feat, ((0,0), (0,pad)))
        else:
            feat = feat[:, :width]
        X.append(feat[..., np.newaxis])
        y.append(data["label"])
    return np.array(X), np.array(y)

def load_feature_dir(dir_path: Path):
    features = []
    labels = []

    npy_files = list(dir_path.glob("*.npy"))
    print(f"🔍 Found {len(npy_files)} .npy files in {dir_path}")

    for npy_file in npy_files:
        x = np.load(npy_file, allow_pickle=True)
        data = x.item()  # ดึง dict ออกมา
        feature = data['features']
        label = data['label']

        features.append(feature)
        labels.append(label)

    if not features:
        raise ValueError(f"No .npy feature files found in {dir_path}")

    X = np.stack(features)

    if len(X.shape) == 3:
        X = X[..., np.newaxis]

    return X, labels