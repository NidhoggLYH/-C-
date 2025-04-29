import pandas as pd
'''
# 读取 Excel 文件
df = pd.read_excel('data/问题四_处理后_聚合后的材料数据.xlsx')

# 筛选“波形”列中值为“三角波”的所有行
#filtered_df = df['励磁波形']

# 提取第五列到最后一列的数据
subset_df = df.iloc[:, 5:]  # 第5列在 pandas 中的索引是4

# 计算每一行的最大值
max_values = subset_df.max(axis=1)

# 在第五列插入最大值，列名为“磁通密度峰值”
df.insert(5, '磁通密度峰值', max_values)  # 在索引5的位置插入（第六列）

# 保存结果到新的 Excel 文件中
df.to_excel('data/问题五_处理后_聚合后的材料数据.xlsx', index=False)

# 打印更新后的数据框
print(df)
'''

# 读取数据
df = pd.read_excel('data/四种材料整合_提取特征_材料波形编码.xlsx')

# 按照材料类型进行分组采样，每个类别采样20%
df_val = df.groupby('材料类型', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=42))

# 剩余的数据作为训练集
df_train = df.drop(df_val.index)

# 输出验证集和训练集的大小
print(f"验证集样本数: {len(df_val)}")
print(f"训练集样本数: {len(df_train)}")

# 可以保存验证集和训练集到新的文件中
df_val.to_excel('data/问题五验证集_按材料随机采样.xlsx', index=False)
df_train.to_excel('data/问题五训练集_按材料随机采样.xlsx', index=False)

'''
# 读取数据
df = pd.read_excel('data/四种材料整合_提取特征_材料波形编码.xlsx')
# 提取第五列到最后一列的数据
subset_df = df.iloc[:, 5:]  # 第5列在 pandas 中的索引是4

# 计算每一行的最大值
max_values = subset_df.max(axis=1)

# 在第五列插入最大值，列名为“磁通密度峰值”
df.insert(5, '磁通密度峰值', max_values)  # 在索引5的位置插入（第六列）

# 保存结果到新的 Excel 文件中
df.to_excel('data/镜像测试集.xlsx', index=False)
'''