import pandas as pd

# 假设你已经读取了合并后的 DataFrame
combined_df = pd.read_excel('data/镜像测试集.xlsx')

# 替换材料类型
combined_df['励磁波形'] = combined_df['励磁波形'].replace({'正弦波': 1, '三角波': 2, '梯形波': 3})
combined_df['磁芯材料'] = combined_df['磁芯材料'].replace({'材料1': 1, '材料2': 2, '材料3': 3, '材料4': 4})

# 保存替换后的数据到新的 Excel 文件
combined_df.to_excel('data/镜像测试集.xlsx', index=False)