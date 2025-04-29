import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib

# 读取数据
df = pd.read_excel('data/问题提取的特征.xlsx')

# 选择特征和目标
features = df[['温度，oC', '频率，Hz', '励磁波形', '标准差', '最大值', '峰峰值', '均方根值', '最大频率','能量']]
target = df['磁芯损耗，w/m3']

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 转换为 NumPy 数组，再转换为 PyTorch 张量
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 定义 MLP 网络
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # 输入层到隐藏层
        self.fc2 = nn.Linear(64, 1)  # 隐藏层到隐藏层
        #self.fc3 = nn.Linear(64, 1)  # 隐藏层到输出层
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        #x = self.relu(x)
        #x = self.fc3(x)
        return x

# 初始化模型、损失函数和优化器
input_dim = x_train_tensor.shape[1]  # 特征数
model = MLP(input_dim)
criterion = nn.MSELoss()  # 损失函数使用均方误差 (MSE)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    optimizer.zero_grad()  # 梯度清零
    outputs = model(x_train_tensor)  # 前向传播
    loss = criterion(outputs, y_train_tensor)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 优化更新参数

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 评估模型
model.eval()  # 设置为评估模式
with torch.no_grad():
    train_pred = model(x_train_tensor)
    test_pred = model(x_test_tensor)

# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# 计算训练集和测试集上的均方误差 (MSE)
train_mse = mean_squared_error(y_train_tensor.numpy(), train_pred.numpy())
test_mse = mean_squared_error(y_test_tensor.numpy(), test_pred.numpy())

print(f"训练集 MSE: {train_mse:.4f}")
print(f"测试集 MSE: {test_mse:.4f}")

# 可视化预测结果
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, label='实际值', color='b')
plt.scatter(range(len(test_pred)), test_pred.numpy(), label='预测值', color='r')
plt.title('MLP 测试集预测 vs 实际值')
plt.xlabel('样本')
plt.ylabel('磁芯损耗')
plt.legend()
plt.grid(True)
plt.show()
