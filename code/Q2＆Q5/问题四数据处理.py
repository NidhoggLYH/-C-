import pandas as pd

# 读取四个sheet并分别添加一个“材料”列
sheet1 = pd.read_excel('ori_data/附件一（训练集）.xlsx', sheet_name=0)
sheet1['材料类型'] = '材料1'

sheet2 = pd.read_excel('ori_data/附件一（训练集）.xlsx', sheet_name=1)
sheet2['材料类型'] = '材料2'

sheet3 = pd.read_excel('ori_data/附件一（训练集）.xlsx', sheet_name=2)
sheet3['材料类型'] = '材料3'

sheet4 = pd.read_excel('ori_data/附件一（训练集）.xlsx', sheet_name=3)
sheet4['材料类型'] = '材料4'


# 将四个sheet合并为一个DataFrame
combined_df = pd.concat([sheet1, sheet2, sheet3, sheet4], ignore_index=True)
#combined_df = pd.concat([sheet1], ignore_index=True)
# 将"材料"列插入到第五列的位置
combined_df.insert(4, '材料类型', combined_df.pop('材料类型'))
# 保存合并后的数据到新的Excel文件
combined_df.to_excel('data/四种材料整合.xlsx', index=False)
