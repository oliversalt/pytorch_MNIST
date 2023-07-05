import torch
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

data = pd.read_csv(r'MNIST_pytorch\Github\train.csv')

data = np.array(data)

m,n = data.shape

np.random.shuffle(data)

# Splitting the dataset into training and testing sets
# Assume 80% training and 20% testing
train_ratio = 0.8
num_train_examples = int(m * train_ratio)

# Split the data
train_data = data[:num_train_examples]
test_data = data[num_train_examples:]

# Split into features and labels for training data
X_train = train_data[:, 1:]  # Everything except the first column
Y_train = train_data[:, 0]   # Only the first column

# Split into features and labels for testing data
X_test = test_data[:, 1:]    # Everything except the first column
Y_test = test_data[:, 0]     # Only the first column

# Normalize the features to be between -1 and 1
X_train = (X_train / 255.) - 0.5
X_test = (X_test / 255.) - 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.long)

X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)

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

# Reshape input data to have shape (batch_size, channels, height, width)
X_train = X_train.view(-1, 1, 28, 28)
X_test = X_test.view(-1, 1, 28, 28)

# Instantiate the CNN
net = SimpleCNN().to(device)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Define batch size
batch_size = 100

# Number of epochs
num_epochs = 6

# Training the Network
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        # Get mini-batch
        inputs = X_train[i: i + batch_size]
        labels = Y_train[i: i + batch_size]

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


# Testing the Network
with torch.no_grad():
    outputs = net(X_test)
    _, predicted = torch.max(outputs.data, 1)
    total = Y_test.size(0)
    correct = (predicted == Y_test).sum().item()
    print(f'Accuracy of the network on test images: {100 * correct / total}%')

# Save the trained model's state_dict
model_path = 'simple_cnn_model.pth'
torch.save(net.state_dict(), model_path)
print(f'Model saved to {model_path}')

# Optionally, to load the model later for inference:
loaded_net = SimpleCNN().to(device)
loaded_net.load_state_dict(torch.load(model_path))
loaded_net.eval() # Set the model to evaluation mode

#using convolutional neural net, i was immediately able to get an accuracy of 98.79% on the test data