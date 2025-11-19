import os
import numpy as np
import joblib
import tkinter as tk
from PIL import Image, ImageDraw

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

MODEL_FILENAME = os.path.join(DATASET_DIR, "mlp_mnist_model_acc_0.9818.pkl")

def load_model():
    if os.path.exists(MODEL_FILENAME):
        print(f"Caricamento modello da {MODEL_FILENAME}...")
        return joblib.load(MODEL_FILENAME)
    print("Errore - Modello non trovato\nSpecificare alla riga 16 il nome del modello che si vuole utilizzare")
    return None

class DigitRecognizerApp(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title("Digit Recognizer")
        self.geometry("650x500")
        self.configure(bg="#f0f0f0")  # sfondo chiaro minimalista
        
        self.model = model
        self.canvas_size = 280

        # Titolo minimalista
        self.title_label = tk.Label(self, text="Digit Recognizer", fg="#333333", bg="#f0f0f0",
                                    font=("Helvetica", 26, "bold"))
        self.title_label.pack(pady=10)

        # Frame per le due canvas
        canvas_frame = tk.Frame(self, bg="#f0f0f0")
        canvas_frame.pack(pady=10)

        # Canvas 1 (scuro)
        self.canvas1 = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size, bg="#1c1c1c", bd=0, highlightthickness=0)
        self.canvas1.grid(row=0, column=0, padx=15)
        self.image1 = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw1 = ImageDraw.Draw(self.image1)
        self.canvas1.bind("<B1-Motion>", lambda e: self.paint(e, self.canvas1, self.draw1))

        # Canvas 2 (scuro)
        self.canvas2 = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size, bg="#1c1c1c", bd=0, highlightthickness=0)
        self.canvas2.grid(row=0, column=1, padx=15)
        self.image2 = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw2 = ImageDraw.Draw(self.image2)
        self.canvas2.bind("<B1-Motion>", lambda e: self.paint(e, self.canvas2, self.draw2))

        # Frame bottoni
        btn_frame = tk.Frame(self, bg="#f0f0f0")
        btn_frame.pack(pady=15)

        # Bottoni Tkinter classici per leggibilità
        self.predict_btn = tk.Button(btn_frame, text="Predict", font=("Helvetica", 16, "bold"),
                                     bg="#4a90e2", fg="white", activebackground="#357ABD",
                                     width=10, command=self.predict_digits)
        self.predict_btn.grid(row=0, column=0, padx=20)

        self.clear_btn = tk.Button(btn_frame, text="Clear", font=("Helvetica", 16, "bold"),
                                   bg="#4a90e2", fg="white", activebackground="#357ABD",
                                   width=10, command=self.clear_canvas)
        self.clear_btn.grid(row=0, column=1, padx=20)

        # Label predizione
        self.label = tk.Label(self, text="Disegna due numeri", fg="#333333", bg="#f0f0f0", font=("Helvetica", 20))
        self.label.pack(pady=10)

    def paint(self, event, canvas, draw):
        x, y = event.x, event.y
        r = 12
        canvas.create_oval(x - r, y - r, x + r, y + r, fill="#ffffff", outline="#ffffff")
        draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear_canvas(self):
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.image1 = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw1 = ImageDraw.Draw(self.image1)
        self.image2 = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw2 = ImageDraw.Draw(self.image2)
        self.label.config(text="Disegna due numeri", fg="#333333")

    def predict_digits(self):
        img1 = np.array(self.image1.resize((28, 28)).convert("L")) / 255.0
        img2 = np.array(self.image2.resize((28, 28)).convert("L")) / 255.0
        imgs = np.vstack([img1.flatten(), img2.flatten()])
        predictions = self.model.predict(imgs)
        self.label.config(text=f"Predizioni: {predictions[0]}{predictions[1]}", fg="#2a9d8f")

if __name__ == "__main__":
    model = load_model()
    if model:
        app = DigitRecognizerApp(model)
        app.mainloop()
