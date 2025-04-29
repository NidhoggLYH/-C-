import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb

# Load the data
file_path = "D:\\2024数学建模\\解压后\\C题\\训练集特征.xlsx"
data = pd.read_excel(file_path)

# Select features and target
features = data[['偏度', '峰值因子', '峰度']]
target = data['励磁波形']

# Adjust the target to start from 0
target_adjusted = target - 1

# Split data into training and validation sets
train_features, val_features, train_target, val_target = train_test_split(
    features, target_adjusted, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
val_features_scaled = scaler.transform(val_features)

# Create an XGBoost classifier model with customized settings
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    verbosity=1,
    max_depth=3,
    n_estimators=100,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0
)

# Train the model
xgb_model.fit(train_features_scaled, train_target)

# Predict on the validation set
val_predictions = xgb_model.predict(val_features_scaled)

# Validation metrics
validation_accuracy = accuracy_score(val_target, val_predictions)
precision = precision_score(val_target, val_predictions, average='macro')
recall = recall_score(val_target, val_predictions, average='macro')
f1 = f1_score(val_target, val_predictions, average='macro')
conf_matrix = confusion_matrix(val_target, val_predictions)

# Print results
print(f"Validation Accuracy: {validation_accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)

# Print actual and predicted labels
val_target_original = val_target + 1  # Convert adjusted labels back to original
val_predictions_original = val_predictions + 1
print("\nActual Labels:", val_target_original.tolist())
print("Predicted Labels:", val_predictions_original.tolist())
