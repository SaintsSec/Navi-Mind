import torch
import torch.nn as nn
import json
from sklearn.preprocessing import LabelEncoder

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Load the trained model
input_size = 50  # Replace with actual input size used during training
hidden_size = 128
output_size = 100  # Replace with actual output size used during training
model = ChatbotModel(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load('./navi-mind.pth'))
model.eval()

# Function to preprocess input text
def preprocess_input(text, input_encoder):
    # Tokenization and other preprocessing steps
    tokens = input_encoder.transform([text])
    return torch.tensor(tokens).unsqueeze(0).to(device)

# Function to generate response
def generate_response(model, input_text, input_encoder, target_encoder):
    input_tensor = preprocess_input(input_text, input_encoder)
    with torch.no_grad():
        output = model(input_tensor)
    # Convert output tensor to text
    response = target_encoder.inverse_transform(output.argmax(dim=1).cpu().numpy())
    return response[0]

# Load encoders
with open('input_encoder.json', 'r') as f:
    input_encoder = LabelEncoder()
    input_encoder.classes_ = json.load(f)

with open('target_encoder.json', 'r') as f:
    target_encoder = LabelEncoder()
    target_encoder.classes_ = json.load(f)

# Example usage
while True:
    input_text = input("Ask me your questions: ")
    if input_text.lower() in ['exit', 'quit']:
        break
    response = generate_response(model, input_text, input_encoder, target_encoder)
    print("Navi >", response)