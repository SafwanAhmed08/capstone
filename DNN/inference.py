import pandas as pd
import numpy as np
import torch
import joblib
import torch.nn as nn

class ImprovedDNN(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super(ImprovedDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.act(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.act(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.act(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.act(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

# Load feature order
feature_order = joblib.load("/Users/safwanahmed/Desktop/capstone/AE/feature_order.pkl")

# Load and prepare data
df = pd.read_csv("syn_flood_features.csv")
for col in feature_order:
    if col not in df.columns:
        df[col] = 0
df = df[feature_order]
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

# Load scaler and PCA if used
scaler = joblib.load("/Users/safwanahmed/Desktop/capstone/AE/standard_scaler.pkl")
X_scaled = scaler.transform(df)
try:
    pca = joblib.load("/Users/safwanahmed/Desktop/capstone/DNN/pca.pkl")
    X_final = pca.transform(X_scaled)
except FileNotFoundError:
    X_final = df

# Load model (full .pth)
model = torch.load("/Users/safwanahmed/Desktop/capstone/DNN/best_model.pth", map_location="cpu", weights_only=False)
model.eval()


# Inference
with torch.no_grad():
    X_tensor = torch.tensor(X_final, dtype=torch.float32)
    outputs = model(X_tensor)
    preds = torch.argmax(outputs, dim=1).numpy()

# Load label encoder and decode predictions
label_encoder = joblib.load("DNN/label_encoder.pkl")
predicted_labels = label_encoder.inverse_transform(preds)

df['predicted_label'] = predicted_labels
df.to_csv("syn_flood_inference_results.csv", index=False)
print("Inference complete. Results saved to syn_flood_inference_results.csv")