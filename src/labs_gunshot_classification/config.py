import os
from pathlib import Path

SAMPLE_RATE = 22050
N_MELS = 64
DURATION = 2.0
AUDIO_DIR = os.path.abspath("audio_data/")

CLASSES = ["gun_shot"]
CLASS_MAP = {label: idx for idx, label in enumerate(CLASSES)}
DATA_DIR = Path("audio_data/esc50_gunshot")
CLASS_TO_INDEX = {label: idx for idx, label in enumerate(CLASSES)}
INDEX_TO_CLASS = {v: k for k, v in CLASS_TO_INDEX.items()}
DATASET_DIR = Path("../../dataset")
FEATURES_DIR = Path("../../features")
LABEL_MAP = {
    'ak': 0, 'aug': 1, 'awm': 2, 'dbs': 3, 'deagle': 4, 'dp': 5, 'g36c': 6, 'gro': 7,
    'k2': 8, 'kar': 9, 'm16': 10, 'm24': 11, 'm249': 12, 'm4': 13, 'mini': 14, 'mk': 15,
    'nogun': 16, 'p18c': 17, 'p1911': 18, 'p90': 19, 'p92': 20, 'pp': 21, 'pump': 22,
    'qbu': 23, 'qbz': 24, 'r1895': 25, 'r45': 26, 's12k': 27, 'scar': 28, 'sks': 29,
    'slr': 30, 'tomy': 31, 'ump': 32, 'uzi': 33, 'vec': 34, 'verl': 35, 'vss': 36, 'win': 37
}