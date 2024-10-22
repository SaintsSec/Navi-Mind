import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import sys

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Preparation
class ChatDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def load_data_from_directory(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                json_data = json.load(f)
                for section in json_data:
                    for input_text in json_data[section]['inputs']:
                        for response_text in json_data[section]['responses']:
                            data.append({"input": input_text, "target": response_text})
    return data

def preprocess_data(data):
    inputs = [item['input'] for item in data]
    targets = [item['target'] for item in data]
    
    # Tokenization and encoding
    input_encoder = LabelEncoder()
    target_encoder = LabelEncoder()
    
    inputs = input_encoder.fit_transform(inputs)
    targets = target_encoder.fit_transform(targets)
    
    inputs = torch.tensor(inputs, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    
    return inputs, targets, input_encoder, target_encoder

# Load and preprocess data
data_directory = './training'
raw_data = load_data_from_directory(data_directory)
inputs, targets, input_encoder, target_encoder = preprocess_data(raw_data)

# Split data into training and validation sets
inputs_train, inputs_val, targets_train, targets_val = train_test_split(inputs, targets, test_size=0.2)

# Create DataLoader
train_dataset = ChatDataset(inputs_train, targets_train)
val_dataset = ChatDataset(inputs_val, targets_val)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Define the neural network
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x.unsqueeze(1))  # Add an extra dimension for batch size
        x = x[:, -1, :]  # Ensure x has the correct dimensions
        x = self.fc(x)
        return x

# Initialize model, loss function, and optimizer
input_size = len(set(inputs.numpy()))  # Vocabulary size
hidden_size = 128
output_size = len(set(targets.numpy()))  # Vocabulary size
model = ChatbotModel(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = input('Enter the number of epochs: ')
num_epochs = int(epochs)
train_losses = []

def print_progress_bar(iteration, total, length=50):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f"\rProgress: |{bar}| {percent}% Complete", end='\r')

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    start_time = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Print progress bar
        print_progress_bar(i + 1, len(train_loader))

    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)
    end_time = time.time()
    epoch_duration = end_time - start_time

    # Enhanced print statement
    print(f"\033[1;34mEpoch [{epoch+1}/{num_epochs}]\033[0m | \033[1;32mLoss: {avg_epoch_loss:.4f}\033[0m | \033[1;33mDuration: {epoch_duration:.2f}s\033[0m", end='\r')
    sys.stdout.flush()
    time.sleep(1)  # Pause for a moment to see the final epoch output

# Save the model
torch.save(model.state_dict(), 'navi-mind.pth')

# Save encoders
with open('input_encoder.json', 'w') as f:
    json.dump(input_encoder.classes_.tolist(), f)

with open('target_encoder.json', 'w') as f:
    json.dump(target_encoder.classes_.tolist(), f)

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.savefig('training_loss.png')  # Save the plot as a file
plt.close()  # Close the plot to avoid display issues