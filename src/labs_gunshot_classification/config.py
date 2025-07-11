import os
from pathlib import Path

SAMPLE_RATE = 22050
N_MELS = 64
DURATION = 2.0  # seconds
AUDIO_DIR = os.path.abspath("audio_data/")

# CLASSES = ["handgun", "rifle", "shotgun", "not_gunshot"]
# CLASS_MAP = {label: idx for idx, label in enumerate(CLASSES)}
# DATA_DIR = Path("audio_data/processed")

CLASSES = ["gun_shot"]
CLASS_MAP = {label: idx for idx, label in enumerate(CLASSES)}
DATA_DIR = Path("audio_data/esc50_gunshot")
CLASS_TO_INDEX = {label: idx for idx, label in enumerate(CLASSES)}
INDEX_TO_CLASS = {v: k for k, v in CLASS_TO_INDEX.items()}