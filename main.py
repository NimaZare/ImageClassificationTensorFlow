import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('cifar10_model.h5')

# Define class names corresponding to your model's output
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# App Theme Set & Main UI Codes
root = tk.Tk()
root.geometry("750x550")
root.title("CIFAR_10 Dataset Image Classification with TensorFlow")
root.config(bg="#333333")

# Function to load and preprocess an image
def load_and_preprocess_image(file_path):
    image = Image.open(file_path)
    image = image.resize((32, 32))  # Resize the image to match the input size of your model
    image = np.array(image) / 255.0  # Normalize pixel values
    return image

# Function to make predictions using the model
def predict_image():
    result_label.config(text="")
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            image = load_and_preprocess_image(file_path)
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            prediction = model.predict(image)
            class_idx = np.argmax(prediction)
            class_label = class_names[class_idx]
            result_label.config(text=f"Predicted Class: {class_label}")
            
            # Display the selected image in the image view
            img = Image.open(file_path)
            img = img.resize((300, 300))  # Resize for display
            img = ImageTk.PhotoImage(img)
            image_view.config(image=img)
            image_view.image = img
        except Exception as e:
            result_label.config(text=f"Error: {str(e)}")


title_lable = tk.Label(
    master = root,
    text = "Image Classification with TensorFlow ©2023",
    bg = "#333333",
    fg = "white",
    font = ("Helvetica", "20", "bold"))

title_lable.pack(pady=(40, 20))  # pad y is 40 top and 20 bottem

# Button to open an image
open_button = tk.Button(
    master = root,
    text = "Open Image",
    bg = "#555555",
    fg = "white",
    command = predict_image)

open_button.pack(pady=10)

# Label to display the prediction result
result_label = tk.Label(
    master = root,
    text = "",
    bg = "#333333",
    fg = "white",
    font = ("Helvetica", 16))

result_label.pack(pady=10)

# Image view to display the selected image
image_view = tk.Label(master = root, bg="#333333")
image_view.pack(pady=10)

# Start the main application loop
root.mainloop()
