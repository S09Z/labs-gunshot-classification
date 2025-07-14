import os
import mlflow
import mlflow.tensorflow
from pathlib import Path
import tensorflow as tf
import keras
import json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from labs_gunshot_classification.model import create_model
from labs_gunshot_classification.preprocess import load_dataset
from labs_gunshot_classification.config import CLASSES, DATA_DIR, CLASS_MAP

if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)
    base_dir = Path("../../features")
    with open(base_dir / "labels.json") as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    X_train, y_train = load_dataset(base_dir / "train")
    X_val, y_val = load_dataset(base_dir / "val")
    X_test, y_test = load_dataset(base_dir / "test")

    # 📦 MLflow setup
    mlflow.set_experiment("Gunshot Classification")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", "SimpleCNN")
        mlflow.log_param("epochs", 30)
        mlflow.log_param("batch_size", 64)

        # Build + train model
        model = create_model(input_shape=(64, 128, 1), num_classes=num_classes)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        
        checkpoint_cb = ModelCheckpoint(
            filepath="artifacts/best_model.keras",  # ใช้นามสกุล .keras ตามคำแนะนำใหม่
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
        
        early_stopping_cb = EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30, batch_size=64,
            # callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
            callbacks=[checkpoint_cb, early_stopping_cb],
        )

        # Log metrics
        final_val_acc = history.history["val_accuracy"][-1]
        mlflow.log_metric("val_accuracy", final_val_acc)

        test_loss, test_acc = model.evaluate(X_test, y_test)
        mlflow.log_metric("test_accuracy", test_acc)

        # Log model artifact
        model.save("model.keras")  # ✅ ใช้ฟอร์แมตใหม่
        mlflow.log_artifact("model.keras")