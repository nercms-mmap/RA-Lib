import pandas as pd
import random

# 读取现有的 simulation_train.csv 文件
train_data = pd.read_csv('data\\simulation_test.csv', header=None)

train_data.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']
# 获取唯一的 Query 和 Item Code
queries = train_data['Query'].unique()
item_codes = train_data['Item Code'].unique()

# 创建相关性数据集
rel_data = []

for query in queries:
    for item_code in item_codes:
        # 随机生成 Relevance 值为 0 或 1
        relevance = random.choice([0, 1])
        rel_data.append([query, 0, item_code, relevance])

# 创建 DataFrame
rel_df = pd.DataFrame(rel_data, columns=['Query', '0', 'Item Code', 'Relevance'])

# 保存为 CSV 文件
rel_df.to_csv('data\\simulation_test_rel.csv', index=False, header=False)

print("模拟相关性文件 'simulation_test_rel.csv' 生成成功！")
