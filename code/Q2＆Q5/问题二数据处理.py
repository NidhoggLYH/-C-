import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('ori_data/附件一（训练集）.xlsx',sheet_name=0)

# 筛选“波形”列中值为“三角波”的所有行
filtered_df = df[df['励磁波形'] == '正弦波']

# 提取第五列到最后一列的数据
subset_df = filtered_df.iloc[:, 4:]  # 第5列在 pandas 中的索引是4

# 计算每一行的最大值
max_values = subset_df.max(axis=1)

# 在第五列插入最大值，列名为“磁通密度峰值”
filtered_df.insert(4, '磁通密度峰值', max_values)  # 在索引5的位置插入（第六列）

# 保存结果到新的 Excel 文件中
filtered_df.to_excel('data/材料1_正弦波_带磁通密度峰值.xlsx', index=False)

# 打印更新后的数据框
print(filtered_df)
