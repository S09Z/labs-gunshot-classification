import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import os

import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

from pathlib import Path
from labs_gunshot_classification.preprocess import load_feature_dir
from labs_gunshot_classification.config import LABEL_MAP

x = np.load(Path("../../features/test/ak_100m_front_0914.npy"), allow_pickle=True)
print(type(x))
print(x)
print(x.shape if hasattr(x, 'shape') else "No shape")


# 🎯 โหลด test set
X_test, y_test_str = load_feature_dir(Path("../../features/test"))
print("✅ X_test shape:", X_test.shape)

# ✅ โหลด LabelEncoder ที่ฝึกไว้ (คุณควรเซฟไว้ตอน train หรือใช้ใหม่)
encoder = LabelEncoder()
encoder.fit(list(LABEL_MAP.keys()))
y_test = encoder.transform(y_test_str)

# ✅ โหลดโมเดล
model = load_model("artifacts/best_model.keras")

# ✅ Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc:.4f} | Loss: {test_loss:.4f}")

# ✅ เริ่ม MLflow
mlflow.set_experiment("gunshot-classification")
with mlflow.start_run(run_name="evaluate-model"):
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_loss", test_loss)

    # 🎯 Predict
    y_probs = model.predict(X_test)
    y_preds = np.argmax(y_probs, axis=1)

    # ✅ Confusion matrix
    cm = confusion_matrix(y_test, y_preds)

    # 📊 วาด confusion matrix
    os.makedirs("artifacts", exist_ok=True)
    cm_path = "artifacts/confusion_matrix.png"

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # 🧾 รายงาน text
    report = classification_report(y_test, y_preds, target_names=encoder.classes_)
    print(report)

    report_path = "artifacts/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)
