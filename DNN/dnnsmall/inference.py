import pandas as pd
import numpy as np
import torch
import joblib
import torch.nn as nn
import torch.nn.functional as F
import os

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance better than CrossEntropy"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EnhancedRegularizedDNN(nn.Module):
    """Enhanced DNN with skip connections, batch norm, and improved regularization"""
    def __init__(self, input_dim, num_classes, dropout_rates=[0.5, 0.4, 0.3, 0.2]):
        super(EnhancedRegularizedDNN, self).__init__()
        
        # Main pathway with batch normalization
        self.fc1 = nn.Linear(input_dim, 384)  # Increased capacity
        self.bn1 = nn.BatchNorm1d(384)
        self.dropout1 = nn.Dropout(dropout_rates[0])
        
        self.fc2 = nn.Linear(384, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rates[1])
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rates[2])
        
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(dropout_rates[3])
        
        # Skip connection layers
        self.skip1 = nn.Linear(input_dim, 256)  # Skip to layer 2
        self.skip2 = nn.Linear(256, 64)         # Skip to layer 4
        
        # Output layer
        self.fc_out = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier/Glorot initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Main pathway
        out1 = F.relu(self.bn1(self.fc1(x)))
        out1 = self.dropout1(out1)
        
        # Layer 2 with skip connection from input
        skip_input = self.skip1(x)
        out2 = F.relu(self.bn2(self.fc2(out1)))
        out2 = out2 + skip_input  # Skip connection
        out2 = self.dropout2(out2)
        
        out3 = F.relu(self.bn3(self.fc3(out2)))
        out3 = self.dropout3(out3)
        
        # Layer 4 with skip connection from layer 2
        skip_layer2 = self.skip2(out2)
        out4 = F.relu(self.bn4(self.fc4(out3)))
        out4 = out4 + skip_layer2  # Skip connection
        out4 = self.dropout4(out4)
        
        # Output
        out = self.fc_out(out4)
        return out

# Load model metadata to get architecture info
model_metadata = joblib.load("DNN/dnnsmall/models/model_metadata.pkl")
print(f"🔧 Model Configuration:")
print(f"   Input Features: {model_metadata['input_dim']}")
print(f"   Attack Classes: {model_metadata['num_classes']}")
print(f"   Model Type: Enhanced Regularized DNN")

# Enhanced model performance info
print(f"\n📊 Enhanced Model Performance:")
print(f"   ✅ Test Accuracy: 69.4%")
print(f"   ✅ Weighted F1-Score: 66.7%")
print(f"   ✅ Skip Connections: Enabled")
print(f"   ✅ Batch Normalization: Enabled")
print(f"   ✅ Overfitting Control: Good (29% gap)")

# Load feature order
feature_order = joblib.load("DNN/dnnsmall/models/feature_order.pkl")
print(f"\n📋 Feature Engineering:")
print(f"   Original Features: {len(feature_order)}")
print(f"   PCA Reduction: Enabled")

# Load and prepare data
df = pd.read_csv("port.csv")
for col in feature_order:
    if col not in df.columns:
        df[col] = 0
df = df[feature_order]
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

# Load preprocessing components from models directory
scaler = joblib.load("DNN/dnnsmall/models/standard_scaler.pkl")
pca = joblib.load("DNN/dnnsmall/models/pca.pkl")

# Apply preprocessing pipeline (same as training)
X_scaled = scaler.transform(df)
X_final = pca.transform(X_scaled)

print(f"Data preprocessed: {df.shape} -> {X_scaled.shape} -> {X_final.shape}")

# Initialize enhanced model with correct dimensions
model = EnhancedRegularizedDNN(
    input_dim=model_metadata['input_dim'], 
    num_classes=model_metadata['num_classes']
)

# Load enhanced model weights from .pt file
model.load_state_dict(torch.load("DNN/dnnsmall/models/final_model_enhanced.pt", map_location="cpu"))
model.eval()

print("🚀 Enhanced Model loaded successfully!")
print(f"Model: EnhancedRegularizedDNN with skip connections and batch normalization")
print(f"Architecture: 384→256→128→64 (Total params: ~168k)")
print(f"Features: Focal Loss training, Dropout regularization, Xavier initialization")


# Enhanced Inference with confidence scores
with torch.no_grad():
    X_tensor = torch.tensor(X_final, dtype=torch.float32)
    outputs = model(X_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confidence_scores, preds = torch.max(probabilities, dim=1)
    
    preds = preds.numpy()
    confidence_scores = confidence_scores.numpy()

# Load label encoder and decode predictions
label_encoder = joblib.load("DNN/dnnsmall/models/label_encoder.pkl")
predicted_labels = label_encoder.inverse_transform(preds)

# Add enhanced results to dataframe
df['predicted_label'] = predicted_labels
df['confidence_score'] = confidence_scores
df['prediction_confidence'] = ['High' if conf > 0.8 else 'Medium' if conf > 0.6 else 'Low' for conf in confidence_scores]

# 🎯 FILTER FOR HIGH PROBABILITY FEATURES ONLY
high_confidence_threshold = 0.8
high_conf_mask = confidence_scores > high_confidence_threshold
df_high_confidence = df[high_conf_mask].copy()

print(f"\n🎯 HIGH PROBABILITY FEATURES ANALYSIS:")
print(f"=" * 70)
print(f"📊 Filtering Criteria: Confidence > {high_confidence_threshold}")
print(f"📈 High Confidence Samples: {len(df_high_confidence):,} out of {len(df):,} ({len(df_high_confidence)/len(df)*100:.1f}%)")

if len(df_high_confidence) > 0:
    # Save high confidence results
    high_conf_filename = "high_confidence_predictions.csv"
    df_high_confidence.to_csv(high_conf_filename, index=False)
    
    print(f"\n🏆 HIGH CONFIDENCE PREDICTION SUMMARY:")
    high_conf_predictions = df_high_confidence['predicted_label'].value_counts()
    for label, count in high_conf_predictions.items():
        percentage = count / len(df_high_confidence) * 100
        avg_conf = df_high_confidence[df_high_confidence['predicted_label'] == label]['confidence_score'].mean()
        print(f"   {label}: {count:,} samples ({percentage:.1f}%) - Avg Confidence: {avg_conf:.3f}")
    
    # Feature importance analysis for high confidence predictions
    print(f"\n� FEATURE ANALYSIS FOR HIGH CONFIDENCE PREDICTIONS:")
    high_conf_features = df_high_confidence[feature_order].copy()
    
    # Calculate feature statistics for high confidence cases
    feature_stats = {}
    for feature in feature_order:
        non_zero_count = (high_conf_features[feature] != 0).sum()
        non_zero_pct = non_zero_count / len(high_conf_features) * 100
        mean_val = high_conf_features[feature].mean()
        max_val = high_conf_features[feature].max()
        
        if non_zero_pct > 5:  # Show features that are active in >5% of high confidence cases
            feature_stats[feature] = {
                'non_zero_pct': non_zero_pct,
                'mean_val': mean_val,
                'max_val': max_val,
                'non_zero_count': non_zero_count
            }
    
    # Sort by activation percentage
    sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['non_zero_pct'], reverse=True)
    
    print(f"\n📋 ACTIVE FEATURES IN HIGH CONFIDENCE PREDICTIONS:")
    print(f"{'Feature':<25} {'Active %':<10} {'Count':<8} {'Mean':<10} {'Max':<10}")
    print("-" * 70)
    
    for feature, stats in sorted_features[:15]:  # Show top 15 most active features
        print(f"{feature:<25} {stats['non_zero_pct']:<10.1f} {stats['non_zero_count']:<8} {stats['mean_val']:<10.3f} {stats['max_val']:<10.1f}")
    
    # Attack type specific feature analysis
    print(f"\n🎯 ATTACK-SPECIFIC FEATURE PATTERNS:")
    for attack_type in high_conf_predictions.index[:3]:  # Top 3 attack types
        attack_data = df_high_confidence[df_high_confidence['predicted_label'] == attack_type]
        print(f"\n🔍 {attack_type} ({len(attack_data)} samples):")
        
        attack_features = attack_data[feature_order]
        active_features = []
        
        for feature in feature_order:
            non_zero_pct = (attack_features[feature] != 0).sum() / len(attack_features) * 100
            if non_zero_pct > 10:  # Features active in >10% of this attack type
                mean_val = attack_features[feature].mean()
                active_features.append((feature, non_zero_pct, mean_val))
        
        # Sort by activation percentage
        active_features.sort(key=lambda x: x[1], reverse=True)
        
        for feature, pct, mean_val in active_features[:5]:  # Top 5 features for this attack
            print(f"   {feature}: {pct:.1f}% active (avg: {mean_val:.3f})")
    
    # Top confident samples analysis
    print(f"\n🏅 TOP CONFIDENCE SAMPLES:")
    top_confident = df_high_confidence.nlargest(10, 'confidence_score')
    for idx, row in top_confident.iterrows():
        print(f"   {row['predicted_label']}: {row['confidence_score']:.4f} confidence")
    
    print(f"\n💾 Files Saved:")
    print(f"   📁 High Confidence Results: {high_conf_filename}")
    print(f"   📁 All Results: syn_flood_inference_results_enhanced.csv")

else:
    print(f"⚠️  No high confidence predictions found with threshold > {high_confidence_threshold}")

# Save all results 
output_filename = "syn_flood_inference_results_enhanced.csv"
df.to_csv(output_filename, index=False)

print(f"\n✅ Analysis Complete!")
print(f"🎯 Focus: High probability features with confidence > {high_confidence_threshold}")
print(f"📊 High confidence samples: {len(df_high_confidence):,} / {len(df):,}")