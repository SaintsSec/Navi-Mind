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
from tqdm import tqdm  # Import tqdm for progress bar
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
import subprocess  # Import subprocess for starting TensorBoard
import pylangacq  # Import pylangacq for parsing .cha files

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
                    inputs = json_data[section]['inputs']
                    responses = json_data[section]['responses']
                    data.extend({"input": input_text, "target": response_text} for input_text in inputs for response_text in responses)
        elif filename.endswith('.cha'):
            cha_data = pylangacq.read_chat(os.path.join(directory, filename))
            for utterance in cha_data.utterances():
                speaker = utterance.speaker
                if speaker == ['LENO', 'LYNN', 'DORI']:  # Assuming 'CHI' is the child and 'MOT' is the mother
                    input_text = utterance.utterance
                elif speaker == ['LENO', 'LYNN', 'DORI']:
                    response_text = utterance.utterance
                    data.append({"input": input_text, "target": response_text})
    return data

def preprocess_data(data):
    inputs = [item['input'] for item in data]
    targets = [item['target'] for item in data]
    
    input_encoder = LabelEncoder()
    target_encoder = LabelEncoder()
    
    inputs = torch.tensor(input_encoder.fit_transform(inputs), dtype=torch.long)
    targets = torch.tensor(target_encoder.fit_transform(targets), dtype=torch.long)
    
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

# Initialize TensorBoard writer
writer = SummaryWriter('runs/chatbot_experiment')

# Training loop
epochs = int(input('Enter the number of epochs: '))
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    start_time = time.time()
    
    # Use tqdm for the progress bar
    with tqdm(total=len(train_loader), desc=f"Epoch [{epoch+1}/{epochs}]", unit="batch") as pbar:
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.update(1)

    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)
    
    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    end_time = time.time()
    epoch_duration = end_time - start_time

    # Print epoch statistics
    print(f"\033[1;34mEpoch [{epoch+1}/{epochs}]\033[0m | \033[1;32mTrain Loss: {avg_epoch_loss:.4f}\033[0m | \033[1;31mVal Loss: {avg_val_loss:.4f}\033[0m | \033[1;33mDuration: {epoch_duration:.2f}s\033[0m")

    # Log metrics to TensorBoard
    writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

# Save the model
torch.save(model.state_dict(), 'navi-mind.pth')

# Save encoders
with open('input_encoder.json', 'w') as f:
    json.dump(input_encoder.classes_.tolist(), f)

with open('target_encoder.json', 'w') as f:
    json.dump(target_encoder.classes_.tolist(), f)

# Close the TensorBoard writer
writer.close()

# Start TensorBoard
subprocess.Popen(['tensorboard', '--logdir=runs'])

print("TensorBoard is running. You can view it at http://localhost:6006")
 # Close the plot to avoid display issues