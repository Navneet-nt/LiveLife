from flask import Flask, request, render_template
import torch
import numpy as np
import torch.nn as nn 

app = Flask(__name__)

# Load trained model
class DepressionClassifierCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DepressionClassifierCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(input_dim // 2 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

model = DepressionClassifierCNN(input_dim=600, num_classes=3)
model.load_state_dict(torch.load("C:\LiveLife\depression_classifier_full_model.pth"))
model.eval()

# Define suggestions
suggestions = {
    "neutral": "You seem fine. Keep doing what you love!",
    "anxious": "Consider mindfulness practices like yoga or meditation.",
    "depressed": "Engage in physical activities like sports, or talk to a trusted friend or professional."
}

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form["user_input"]
        
        # Process input (dummy processing here)
        input_vector = np.random.rand(1, 600)  # Replace with actual vectorization logic
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)
        
        # Predict
        with torch.no_grad():
            prediction = model(input_tensor)
            category = torch.argmax(prediction, dim=1).item()
        
        # Map prediction to label
        label_map = {0: "neutral", 1: "anxious", 2: "depressed"}
        result = label_map[category]
        advice = suggestions[result]
        
        return render_template("result.html", result=result, advice=advice)

    return render_template("index.html")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
