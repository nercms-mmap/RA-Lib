# 本代码用于生成模拟测试数据

import pandas as pd
import random

# train 42
# test 150
# 设置随机种子以保证可重复性
random.seed(150)

# 参数设置
num_queries = 20
num_voters = 4
num_items = 200

# 创建投票者名称
voter_names = [f'Voter_{i + 1}' for i in range(num_voters)]

# 创建查询
queries = [f'Query_{i + 1}' for i in range(num_queries)]

# 创建数据集
data = []

for query in queries:
    for voter in voter_names:
        # 随机选择项目并生成排名
        item_codes = list(range(1, num_items + 1))
        random.shuffle(item_codes)  # 随机打乱项目顺序
        for rank, item_code in enumerate(item_codes, start=1):
            data.append([query, voter, item_code, rank])

# 创建 DataFrame
df = pd.DataFrame(data, columns=['Query', 'Voter Name', 'Item Code', 'Item Rank'])

# 保存为 CSV 文件
df.to_csv('data\\simulation_test.csv', index=False, header=False)

print("模拟测试数据集 'simulation_test.csv' 生成成功！")
