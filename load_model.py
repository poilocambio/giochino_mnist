import os
import numpy as np
import joblib
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

MODEL_FILENAME = os.path.join(DATASET_DIR, "mlp_mnist_model_acc_0.9818.pkl")

def load_model():
    if os.path.exists(MODEL_FILENAME):
        print(f"Caricamento modello da {MODEL_FILENAME}...")
        return joblib.load(MODEL_FILENAME)
    print("Errore - Modello non trovato\nSpecificare alla riga 16 il nome del modello che si vuole utilizzare")
    
class DigitRecognizerApp(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title("Digit Recognizer")
        self.geometry("450x420")
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

        style = ttk.Style()
        style.configure("Big.TButton",
            font=("Helvetica", 20),   
            padding=(10, 10)) 
        
        self.predict_btn = ttk.Button(btn_frame, text="Predict", style="Big.TButton", command=self.predict_digit)
        self.predict_btn.grid(row=0, column=0, padx=10)

        self.clear_btn = ttk.Button(btn_frame, text="Clear", style="Big.TButton", command=self.clear_canvas)
        self.clear_btn.grid(row=0, column=1, padx=10)

        
        self.label = tk.Label(self, text="Disegna un numero", fg="white", bg="#2e2e2e", font=("Helvetica", 20))
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
        self.label.config(text=f"Predizione: {prediction}", font=("Helvetica", 20))

if __name__ == "__main__":
    model = load_model()
    app = DigitRecognizerApp(model)
    app.mainloop()
