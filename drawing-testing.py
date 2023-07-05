import torch
import torch.nn as nn
import tkinter as tk
from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
import torch.nn.functional as F

# You will need to import SimpleCNN, or define it within this script exactly as it was defined during training
# from <module_where_SimpleCNN_is_defined> import SimpleCNN

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('MNIST_pytorch\simple_cnn_model.pth')) # Update path if needed
model.eval()

# Create TKinter window
window = tk.Tk()
window.title("Digit Recognizer")
window.geometry("300x300")

# Create a drawing canvas
canvas = Canvas(window, width=280, height=280)
canvas.pack()

# Create PIL Image
image1 = Image.new("RGB", (280, 280), (255, 255, 255))
draw = ImageDraw.Draw(image1)

# Drawing function
def paint(event):
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
    draw.line([x1, y1, x2, y2], fill="black", width=5)

# Attach the painting function to canvas
canvas.bind("<B1-Motion>", paint)

# Clear canvas function
def clear_canvas():
    canvas.delete("all")
    global image1, draw
    image1 = Image.new("RGB", (280, 280), (255, 255, 255))
    draw = ImageDraw.Draw(image1)

# Prediction function
def predict_digit():
    # Resize image to 28x28
    img = image1.resize((28, 28))
    # Convert to grayscale
    img = img.convert('L')
    # Convert to numpy array
    img = np.array(img)
    # Rescale values to be between 0 and 1
    img = (255.0 - img) / 255.0
    # Convert to tensor
    img = torch.tensor(img, dtype=torch.float32).to(device)
    # Add batch and channel dimensions
    img = img.unsqueeze(0).unsqueeze(0)
    # Predict
    output = model(img)
    _, predicted = torch.max(output.data, 1)
    print(f'Predicted digit: {predicted.item()}')
    # Clear canvas for next drawing
    clear_canvas()

# Button to predict digit
btn_predict = tk.Button(window, text="Predict", command=predict_digit)
btn_predict.pack()

window.mainloop()
