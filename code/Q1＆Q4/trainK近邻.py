import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 数据加载和预处理（如上所示）
train_file_path = "D:\\2024数学建模\\解压后\\C题\\训练集特征.xlsx"
test_file_path = "D:\\2024数学建模\\解压后\\C题\\测试集特征.xlsx"
train_data = pd.read_excel(train_file_path)
test_data = pd.read_excel(test_file_path)
train_features = train_data[['偏度', '峰值因子', '峰度']]
train_target = train_data['励磁波形'] - 1
test_features = test_data[['偏度', '峰值因子', '峰度']]
X_train, X_val, y_train, y_val = train_test_split(
    train_features, train_target, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_features_scaled = scaler.transform(test_features)

# 创建KNN模型
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    algorithm='auto',
    metric='euclidean',
    n_jobs=-1
)

# 训练模型
knn.fit(X_train_scaled, y_train)

# 预测验证集
val_predictions = knn.predict(X_val_scaled)

# 评估指标
val_accuracy = accuracy_score(y_val, val_predictions)
val_precision = precision_score(y_val, val_predictions, average='macro')
val_recall = recall_score(y_val, val_predictions, average='macro')
val_f1 = f1_score(y_val, val_predictions, average='macro')
val_conf_matrix = confusion_matrix(y_val, val_predictions)

# 打印验证结果
print(f"Validation Accuracy: {val_accuracy}")
print(f"Precision: {val_precision}")
print(f"Recall: {val_recall}")
print(f"F1 Score: {val_f1}")
print("Confusion Matrix:")
print(val_conf_matrix)

# 预测测试集
test_predictions = knn.predict(test_features_scaled)
test_predictions_original = test_predictions + 1  # 转换回原始标签尺度

# 输出测试集的预测结果
print("\nTest Predictions:", test_predictions_original.tolist())
