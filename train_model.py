import os
import numpy as np
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

def train_model():
    print("Addestramento di un nuovo modello MNIST...")

    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data / 255.0, mnist.target.astype(int)

    X = X.values if hasattr(X, 'values') else X
    y = y.values if hasattr(y, 'values') else y

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=500,
        learning_rate="adaptive",
        early_stopping=True,
        alpha=0.0001,
        random_state=42,
        tol=1e-5
    )

    print("Addestramento modello...")
    mlp.fit(X_train, y_train)

    print("Calcolo accuratezza media...")
    scores = cross_val_score(mlp, X, y, cv=3)
    mean_accuracy = scores.mean()
    print(f"Accuratezza media: {mean_accuracy:.4f}")

    model_filename = os.path.join(DATASET_DIR, f"mlp_mnist_acc_{mean_accuracy:.4f}.pkl")
    joblib.dump(mlp, model_filename)
    print(f"Modello salvato in: {model_filename}")

if __name__ == "__main__":
    train_model()

