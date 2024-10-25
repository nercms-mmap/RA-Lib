"""
主要是为了转化成mat格式的数据文件
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.io import savemat

data = pd.read_csv('data\\simulation_test.csv', header=None)
rel_data = pd.read_csv('data\\simulation_test_rel.csv', header=None)
data.columns = ['User', 'Algorithm', 'Item', 'Rank']
rel_data.columns = ['User', '0', 'Item', 'Relevance']

# 按照 'User' 分组，并计算每组中不同 'item' 的数量
grouped = data.groupby('User')['Item'].nunique()

# 找出数量的最大值
max_unique_items = grouped.max()
print(max_unique_items)

unique_query = data['User'].unique()
unique_voters = data['Algorithm'].unique()

shape = (len(unique_voters), len(unique_query), max_unique_items)
result = np.full(shape, 0)
rel_result = np.zeros((len(unique_query), max_unique_items))
query_mapping = {name: i for i, name in enumerate(unique_query)}
voter_mapping = {name: i for i, name in enumerate(unique_voters)}

for query in tqdm(unique_query):
    query_data = data[data['User'] == query]
    query_rel = rel_data[rel_data['User'] == query]

    unique_items = query_data['Item'].unique()
    item_mapping = {name: i for i, name in enumerate(unique_items)}

    for _, row in query_rel.iterrows():
        user = row['User']
        item = row['Item']
        rel = row['Relevance']
        if rel <= 0 or item not in item_mapping:
            continue

        user_idx = query_mapping[user]
        item_idx = item_mapping[item]
        rel_result[user_idx, item_idx] = rel

    for _, row in query_data.iterrows():
        voter = row['Algorithm']
        item = row['Item']
        Rank = row['Rank']
        query_idx = query_mapping[query]
        voter_idx = voter_mapping[voter]
        item_idx = item_mapping[item]
        result[voter_idx, query_idx, item_idx] = Rank

for k in range(result.shape[1]):
    for i in range(result.shape[0]):
        max_rank = np.max(result[i, k, :])
        for j in range(result.shape[2]):
            if result[i, k, j] == 0:  # partial lists
                continue
            # result[i, k, j] = max_rank - result[i, k, j] + 1
            result[i, k, j] = result.shape[2] - result[i, k, j] + 1

savemat('data\\simulation_test.mat', {'test': result.astype(np.float64)})
savemat('data\\simulation_test_rel.mat', {'testrel': rel_result})
