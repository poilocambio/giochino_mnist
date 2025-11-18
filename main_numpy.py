import os
import numpy as np
import joblib
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier

# Creiamo la cartella "dataset" se non esiste
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# Nome di default del modello (temporaneo, poi lo cambieremo dopo il training)
MODEL_FILENAME = os.path.join(DATASET_DIR, "mlp_mnist_model_numpy_0.2.pkl")

def load_or_train_model():
    # Se il modello esiste già, lo carichiamo
    if os.path.exists(MODEL_FILENAME):
        print(f"Caricamento modello da {MODEL_FILENAME}...")
        return joblib.load(MODEL_FILENAME)
    
    print("Caricamento dataset MNIST...")
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    
    # Convertiamo in NumPy array se fossero DataFrame
    X = X.values if hasattr(X, 'values') else X
    y = y.values if hasattr(y, 'values') else y
    
    X_train, X_test, y_train, y_test = train_test_split(
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
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),  # più grande
    activation="relu",
    solver="adam",
    max_iter=500,  # più iterazioni
    learning_rate="adaptive",
    early_stopping=True,
    alpha=0.0001,
    random_state=42
)

    print("Addestramento modello...")
    mlp.fit(X_train, y_train)

    # Accuratezza media
    scores = cross_val_score(mlp, X, y, cv=5)
    mean_accuracy = scores.mean()
    print(f"Accuratezza media: {mean_accuracy:.4f}")

    # Salvataggio modello con accuratezza nel nome
    model_filename = os.path.join(DATASET_DIR, f"mlp_mnist_acc_{mean_accuracy:.4f}.pkl")
    joblib.dump(mlp, model_filename)
    print(f"Modello salvato in {model_filename}")

    return mlp

class DigitRecognizerApp(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title("Digit Recognizer")
        self.geometry("350x400")
        self.configure(bg="#2e2e2e")
        self.model = model
        
        self.canvas_size = 280
        self.canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack(pady=10)
        
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        
        btn_frame = tk.Frame(self, bg="#2e2e2e")
        btn_frame.pack()
        
        self.predict_btn = ttk.Button(btn_frame, text="Predict", command=self.predict_digit)
        self.predict_btn.grid(row=0, column=0, padx=10)
        
        self.clear_btn = ttk.Button(btn_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.grid(row=0, column=1, padx=10)
        
        self.label = tk.Label(self, text="Disegna un numero", fg="white", bg="#2e2e2e", font=("Arial", 14))
        self.label.pack(pady=10)
    
    def paint(self, event):
        x, y = event.x, event.y
        r = 12
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="white")
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
    
    def predict_digit(self):
        img = np.array(self.image.resize((28, 28)).convert("L")) / 255.0
        img = img.flatten().reshape(1, -1)
        prediction = self.model.predict(img)[0]
        self.label.config(text=f"Predizione: {prediction}")

if __name__ == "__main__":
    model = load_or_train_model()
    app = DigitRecognizerApp(model)
    app.mainloop()
