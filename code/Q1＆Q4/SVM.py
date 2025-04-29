import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # 导入SVM模型
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 加载训练数据
train_file_path = "D:\\2024数学建模\\解压后\\C题\\训练集特征.xlsx"
train_data = pd.read_excel(train_file_path)

# 加载测试数据
test_file_path = "D:\\2024数学建模\\解压后\\C题\\测试集特征.xlsx"
test_data = pd.read_excel(test_file_path)

# 从训练数据中选择特征和目标
train_features = train_data[['偏度', '峰值因子', '峰度']]
train_target = train_data['励磁波形'] - 1  # 调整目标使其从0开始

# 从测试数据中准备特征
test_features = test_data[['偏度', '峰值因子', '峰度']]

# 将训练数据分割为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    train_features, train_target, test_size=0.3, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_features_scaled = scaler.transform(test_features)

# 创建SVM分类器模型
svm_model = SVC(
    kernel='rbf',           # 核函数类型，可选 'linear', 'poly', 'rbf', 'sigmoid'
    C=1.0,                   # 惩罚参数
    gamma='scale',           # 核函数的系数
    decision_function_shape='ovr',  # 一对多策略
    probability=False,       # 是否启用概率估计
    random_state=42
)

# 在训练数据上训练模型
svm_model.fit(X_train_scaled, y_train)

# 在验证集上进行预测
val_predictions = svm_model.predict(X_val_scaled)
val_predictions_original = val_predictions + 1  # 将预测结果转换回原始标签尺度

# 计算验证集的评估指标
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

# 在测试数据上进行预测
test_predictions = svm_model.predict(test_features_scaled)
test_predictions_original = test_predictions + 1  # 将预测结果转换回原始标签尺度

# 输出测试集的预测结果
print("\nTest Predictions:", test_predictions_original.tolist())
