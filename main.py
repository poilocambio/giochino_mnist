import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Modello pre-addestrato


# INFO IMPORTANTI SULLE VERSIONI DEI MODELLI

#L'algoritmo scritto sotto è quello che ha prodotto il modello 0.4, il 0.3 è uscito meglio, le differenze se non sbaglio sono:
# - la standardizzazione dei dati che nello 0.3 non esisteva;
# - early_stopping=True non c'era in 0.3
# - alpha=0.001 non c'era in 0.3
# HO COMMMENTATO TUTTO QUESTE DIFFERENZE PER CERCARE DI CONTINUARE SULLA STRADA DEL 0.3 
#
# || 0.3.1 ||
# il 0.3.1 (accuratezza media: 0.9770) è semplicemente il 0.3 con una piccola differenza: il fattore tol è 1e-4, non più 1e-5, magari era una questione di overflow e così dovrebbe evitarlo leggermente di più
# il 0.3.1 fa confusione con la stanghetta dell'1, il 2 poco, il 3 con l'8, il 4 lo vuole scritto bene, il 9 pure, ma comunque non è troppo diverso dal 0.3 e 0.4
# però il 0.3 rimane il miglire di nuovo, quindi continuerò con il tol del 0.3 ma con altre differenze: 
#
#|| 0.3.2 ||
# - i nueroni del 0.3.2 saranno (100, 100, 100), non (150, 100, 50) come il 0.3;
# - l'activation non è più tanh;
# - ho aggiunto di nuovo aplha ma stavolta con 0.0001 anzichè 0.001 che era per 0.4;
# per ora provo on questi cambiamenti perché il tempo per l'addestramento sta aumentando
# il 0.3.2 è stato un fallimento credo, perché tutto l'addestramento del modello va a cazzo di cane, sscende poi risalve, poi riscende e risale, non credo sia buono
# E' peggiore: 
# - Accuratezza media: 0.9792; 
# - l'1 non lo prende con la stanghetta, il 3 li prende, il 4 lo confonde tantissimo con il 6, 5 e 6 vanno bene, 7 confuso per 2, 8 con il 3, il 9 lo prende 
#
# || 0.3.3 ||
# Chatgpt consiglia queste modifiche per il 0.3.3:
# - mex_iter da 1000 a 2000 
# - learning_rate="adaptive";
# - early_stopping=True;
# - solver="sgd" anzichè "adam";
# quasi tutti i numeri li prende bene ma ha un accuratezza media di 0.9702 e si confonde molto con 1, 7 e poco con 6
MODEL_FILENAME = "mlp_mnist_model_0.5.pkl"
def load_or_train_model():
    if os.path.exists(MODEL_FILENAME):
        print("Caricamento modello...")
        return joblib.load(MODEL_FILENAME)
    
    print("Caricamento dataset MNIST...")
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardizzazione dei dati
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)  # Applichiamo la standardizzazione ai dati di addestramento
    #X_test = scaler.transform(X_test)  # Applichiamo la stessa trasformazione ai dati di test

    # Definiamo una rete neurale con 3 hidden layers e regularizzazione L2 (alpha)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), 
                        activation="relu", 
                        tol=1e-5, 
                        solver="sgd", 
                        max_iter=2000,#da 1000 a 2000 per 0.3.3 
                        verbose=True, 
                        random_state=42,
                        learning_rate="adaptive", #aggiunto per 0.3.3
                        early_stopping=True,
                        alpha=0.0001)  # Aggiungiamo la regularizzazione L2
                    
    print("Addestramento modello...")
    mlp.fit(X_train, y_train)

    scores = cross_val_score(mlp, X, y, cv=5)
    print(f"Accuratezza media: {scores.mean():.4f}")
    joblib.dump(mlp, MODEL_FILENAME)
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
        r = 12  # Tratto più spesso
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="white")
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
    
    def predict_digit(self):
        img = self.image.resize((28, 28)).convert("L")
        img = np.array(img) / 255.0
        img = img.flatten().reshape(1, -1)
        

        img_df = pd.DataFrame(img, columns=[f"pixel{i+1}" for i in range(img.shape[1])])
        prediction = self.model.predict(img_df)[0]
        #prediction = self.model.predict(img)[0]
        self.label.config(text=f"Predizione: {prediction}")

if __name__ == "__main__":
    model = load_or_train_model()
    app = DigitRecognizerApp(model)
    app.mainloop()
