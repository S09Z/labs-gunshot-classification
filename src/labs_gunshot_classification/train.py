

import mlflow
import mlflow.tensorflow
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping
from labs_gunshot_classification.model import create_model
from labs_gunshot_classification.preprocess import load_dataset
from labs_gunshot_classification.config import CLASSES, DATA_DIR, CLASS_MAP

def train():
    X_train, X_val, y_train, y_val = load_dataset(DATA_DIR, CLASS_MAP)
    
    input_shape = X_train.shape[1:]  # e.g., (64, 87, 1)
    num_classes = len(CLASSES)

    model = create_model(input_shape=input_shape, num_classes=num_classes)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 🔥 MLflow start
    with mlflow.start_run():
        mlflow.tensorflow.autolog()  # <- This tracks training automatically

        model.fit(X_train, y_train,
                  
                  validation_data=(X_val, y_val),
                  epochs=15,
                  batch_size=32,
                  callbacks=[EarlyStopping(patience=3)]
                  )

        # Optional: log manually
        mlflow.log_param("model_type", "CNN")
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_metric("final_val_accuracy", model.evaluate(X_val, y_val)[1])

        # Save model
        mlflow.keras.log_model(model, artifact_path="model")

if __name__ == "__main__":
    train()
