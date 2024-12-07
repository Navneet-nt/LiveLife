import joblib
from flask import Flask, request, render_template
import torch
import torch.nn as nn
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the fitted TF-IDF vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

class DepressionClassifierCNN(nn.Module):
    def __init__(self, input_channels, height, width, num_classes):
        super(DepressionClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)  # This will downsample by a factor of 2
        
        # Adjust the fully connected layer as per the saved model
        self.fc1 = nn.Linear(192, 128)  # Adjusted to match saved model weights
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten the tensor after pooling
        x = x.view(x.size(0), -1)  # Flatten

        # Pass through fc1 and fc2
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# Initialize the model
model = DepressionClassifierCNN(input_channels=1, height=20, width=30, num_classes=3)

# Load the trained model weights
model.load_state_dict(torch.load("depression_classifier_cnn.pth"))
model.eval()  # Set the model to evaluation mode

# Define suggestions based on the predicted category
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
        
        # Ensure the vectorizer is fitted before transforming
        if hasattr(vectorizer, 'idf_'):
            # Vectorize the text input using TF-IDF
            input_vector = vectorizer.transform([user_input]).toarray()

            # Convert the input vector to a torch tensor
            input_tensor = torch.tensor(input_vector, dtype=torch.float32)
            
            # Make the prediction
            with torch.no_grad():
                prediction = model(input_tensor)
                category = torch.argmax(prediction, dim=1).item()
            
            # Map prediction to label
            label_map = {0: "neutral", 1: "anxious", 2: "depressed"}
            result = label_map[category]
            advice = suggestions[result]
            
            return render_template("result.html", result=result, advice=advice)
        else:
            return "Error: TF-IDF vectorizer not properly fitted."

    return render_template("index.html")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
