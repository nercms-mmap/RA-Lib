"""
存储各种评估指标的代码实现
Tancilon: 20231219
"""
import csv

import numpy as np
import pandas as pd
import math
import scipy.io
import h5py
import os
from functools import reduce
from tqdm import tqdm


class Evaluation():
    def __init__(self) -> None:
        pass

    """
    Data Process
    """

    def covert_pd_to_csv(self, query_test_data, query_rel_data):
        unique_items = query_test_data['Item Code'].unique()

        item_num = len(unique_items)
        item_mapping = {name: i for i, name in enumerate(unique_items)}
        ra_list = np.zeros(item_num)
        rel_list = np.zeros(item_num)

        for _, row in query_test_data.iterrows():
            item_code = row['Item Code']
            item_rank = row['Item Rank']

            item_id = item_mapping[item_code]
            ra_list[item_id] = item_rank

        for _, row in query_rel_data.iterrows():
            item_code = row['Item Code']
            item_rel = row['Relevance']

            if item_code not in item_mapping:
                continue

            item_id = item_mapping[item_code]
            rel_list[item_id] = item_rel

        return ra_list, rel_list

    """
    Compute precision
    Precision is the proportion of the retrieved documents that are relevant.
        Precision = r / n
    where,
        r is the number of retrieved relevant documents;
        n is the number of retrieved documents.

    compute_P_s:
            score_list: 1 * item, 一维Numpy数组, 数组内存放分数
            rel_list: 1 * item, 一维Numpy数组, 数组内存放相关性分数

    compute_P_r:
            list: 1 * item, 一维Numpy数组, 数组内存放排名
            rel_list: 1 * item, 一维Numpy数组, 数组内存放相关性分数

    compute_AP_r:
            list: 1 * item, 一维Numpy数组, 数组内存放排名
            rel_list: 1 * item, 一维Numpy数组, 数组内存放相关性分数

    compute_AP_s:
            score_list: 1 * item, 一维Numpy数组, 数组内存放分数
            rel_list: 1 * item, 一维Numpy数组, 数组内存放相关性分数

    eval_map:
            用于对算法输出的结果进行评测
            test_data: 
                - csv文件格式
                - Query | Item Code | Item Rank
            rel_data:
                - csv文件格式
                - Query | 0 | Item | Relevance
    """

    def compute_P_s(self, score_list, rel_list, topk):
        if (topk <= 0 or topk > len(rel_list)):
            topk = len(rel_list)
        # 将分数转化为排名，排名从0开始
        rank_list = np.argsort(score_list)[::-1]
        r = 0

        for k in range(topk):
            item_idx = rank_list[k]
            if (rel_list[item_idx] > 0):
                r += 1

        return r / topk

    def compute_P_r(self, list, rel_list, topk):
        rank_list = np.argsort(list)
        r = 0
        if (topk > len(rel_list)):
            # print("Warning: Calculate precision metrics where topk is greater than the number of items")
            topk = len(rel_list)

        for k in range(topk):
            item_idx = rank_list[k]
            if (rel_list[item_idx] > 0):
                r += 1
        return r / topk

    def compute_R_r(self, list, rel_list, topk):
        rank_list = np.argsort(list)
        if (topk > len(rel_list)):
            # print("Warning: Calculate precision metrics where topk is greater than the number of items")
            topk = len(rel_list)

        for k in range(topk):
            item_idx = rank_list[k]
            if (rel_list[item_idx] > 0):
                return 1

        return 0

    def compute_R_s(self, score_list, rel_list, topk):
        if (topk <= 0 or topk > len(rel_list)):
            topk = len(rel_list)
        # 将分数转化为排名，排名从0开始
        rank_list = np.argsort(score_list)[::-1]
        for k in range(topk):
            item_idx = rank_list[k]
            if (rel_list[item_idx] > 0):
                return 1

        return 0

    def compute_recall_r(self, list, rel_list, topk):
        rank_list = np.argsort(list)
        if (topk > len(rel_list)):
            # print("Warning: Calculate precision metrics where topk is greater than the number of items")
            topk = len(rel_list)

        total_r = np.sum(rel_list > 0)

        if (total_r == 0):
            return 0

        retrieved_r = 0.0
        for k in range(topk):
            item_idx = rank_list[k]
            if (rel_list[item_idx] > 0):
                retrieved_r += 1

        return retrieved_r / total_r

    def compute_recall_s(self, score_list, rel_list, topk):
        if (topk <= 0 or topk > len(rel_list)):
            topk = len(rel_list)
        # 将分数转化为排名，排名从0开始
        rank_list = np.argsort(score_list)[::-1]
        total_r = np.sum(rel_list > 0)

        if (total_r == 0):
            return 0

        retrieved_r = 0.0
        for k in range(topk):
            item_idx = rank_list[k]
            if (rel_list[item_idx] > 0):
                retrieved_r += 1

        return retrieved_r / total_r

    def compute_AP_r(self, list, rel_list, topk):
        total_r = np.sum(rel_list > 0)
        total_r = min(total_r, len(list))

        ap = 0.0
        for k in range(1, topk + 1):
            item_id = np.argmax(list == k)
            if (rel_list[item_id] > 0):
                p_k = self.compute_P_r(list, rel_list, k)
                ap += p_k / total_r
        return ap

    def compute_AP_s(self, score_list, rel_list, topk):
        sorted_indices = np.argsort(score_list)[::-1]
        para_list = np.argsort(sorted_indices) + 1
        ap = self.compute_AP_r(para_list, rel_list, topk)
        return ap

    def eval_map(self, test_data, rel_data, topk=None):
        test_data.columns = ['Query', 'Item Code', 'Item Rank']
        rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']

        unique_queries = test_data['Query'].unique()
        sum_ap = 0.0
        ap_query_results = []
        for query in unique_queries:
            query_test_data = test_data[test_data['Query'] == query]
            query_rel_data = rel_data[rel_data['Query'] == query]
            ra_list, rel_list = self.covert_pd_to_csv(query_test_data, query_rel_data)
            if (topk is None):
                topk = len(ra_list)
            ap_value = self.compute_AP_r(ra_list, rel_list, topk)
            sum_ap += ap_value
            ap_query_results.append([query - 1, ap_value])  # 推荐系统中：为了保证索引从0开始与mat格式的评测中Query编号一致，但此处会让代码很脆弱 后续可考虑优化

        return sum_ap / len(unique_queries), ap_query_results

    def eval_map_matlab(self, test_path, rel_path, test_data_name, test_rel_name, type=None, topk=10):
        test_mat = scipy.io.loadmat(test_path)
        rel_mat = scipy.io.loadmat(rel_path)
        # with h5py.File(rel_path, 'r') as file:
        #     test_rel = file['testRelevanceMatrix'][()]
        # rel_data = np.transpose(test_rel, axes=[1, 0])

        test_data = test_mat[test_data_name]
        rel_data = rel_mat[test_rel_name]

        ap_results = []
        if (type == 'rank'):
            sum_ap = 0.0
            for query in range(test_data.shape[0]):
                ap_value = self.compute_AP_r(test_data[query, :], rel_data[query, :], topk)
                sum_ap += ap_value
                ap_results.append([query, ap_value])

            return sum_ap / test_data.shape[0], ap_results

        elif (type == 'score'):
            sum_ap = 0.0
            for query in range(test_data.shape[0]):
                ap_value = self.compute_AP_s(test_data[query, :], rel_data[query, :], topk)
                sum_ap += ap_value
                ap_results.append([query, ap_value])

            return sum_ap / test_data.shape[0], ap_results

    def eval_ndcg_matlab(self, test_path, rel_path, test_data_name, test_rel_name, type=None, topk=10):
        test_mat = scipy.io.loadmat(test_path)
        rel_mat = scipy.io.loadmat(rel_path)
        # with h5py.File(rel_path, 'r') as file:
        #     test_rel = file['testRelevanceMatrix'][()]
        # rel_data = np.transpose(test_rel, axes=[1, 0])

        test_data = test_mat[test_data_name]
        rel_data = rel_mat[test_rel_name]

        sum_ndcg = 0.0
        query_results = []

        for query in range(test_data.shape[0]):
            if (type == 'score'):
                score_list = test_data[query, :]
                sorted_indices = np.argsort(score_list)[::-1]
                para_list = np.argsort(sorted_indices) + 1
            elif (type == 'rank'):
                para_list = test_data[query, :]

            ndcg_value = self.compute_ndcg_r(para_list, rel_data[query, :], topk)
            sum_ndcg += ndcg_value
            query_results.append([query, ndcg_value])

        return sum_ndcg / test_data.shape[0], query_results

    def eval_r1_matlab(self, test_path, rel_path, test_data_name, test_rel_name, type=None):
        test_mat = scipy.io.loadmat(test_path)
        rel_mat = scipy.io.loadmat(rel_path)

        # with h5py.File(rel_path, 'r') as file:
        #     test_rel = file['testRelevanceMatrix'][()]
        # rel_data = np.transpose(test_rel, axes=[1, 0])

        test_data = test_mat[test_data_name]
        rel_data = rel_mat[test_rel_name]

        # print(test_data.shape[0])
        # print(test_data.shape[1])
        # print(rel_data.shape[0])
        # print(rel_data.shape[1])

        if (type == 'rank'):
            sum_r = 0.0
            for query in range(test_data.shape[0]):
                sum_r += self.compute_P_r(test_data[query, :], rel_data[query, :], 1)

            return sum_r / test_data.shape[0]

        elif (type == 'score'):
            sum_r = 0.0
            for query in range(test_data.shape[0]):
                sum_r += self.compute_P_s(test_data[query, :], rel_data[query, :], 1)

            return sum_r / test_data.shape[0]

    def eval_r_matlab(self, test_path, rel_path, test_data_name, test_rel_name, type=None, topk=1):
        test_mat = scipy.io.loadmat(test_path)
        rel_mat = scipy.io.loadmat(rel_path)

        # with h5py.File(rel_path, 'r') as file:
        #     test_rel = file['testRelevanceMatrix'][()]
        # rel_data = np.transpose(test_rel, axes=[1, 0])

        test_data = test_mat[test_data_name]
        rel_data = rel_mat[test_rel_name]

        # print(test_data.shape[0])
        # print(test_data.shape[1])
        # print(rel_data.shape[0])
        # print(rel_data.shape[1])

        if (type == 'rank'):
            sum_r = 0.0
            for query in range(test_data.shape[0]):
                sum_r += self.compute_R_r(test_data[query, :], rel_data[query, :], topk)

            return sum_r / test_data.shape[0]

        elif (type == 'score'):
            sum_r = 0.0
            for query in range(test_data.shape[0]):
                sum_r += self.compute_R_s(test_data[query, :], rel_data[query, :], topk)

            return sum_r / test_data.shape[0]

    def eval_recall_matlab(self, test_path, rel_path, test_data_name, test_rel_name, type=None, topk=1):
        test_mat = scipy.io.loadmat(test_path)
        rel_mat = scipy.io.loadmat(rel_path)

        test_data = test_mat[test_data_name]
        rel_data = rel_mat[test_rel_name]

        if (type == 'rank'):
            sum_r = 0.0
            for query in range(test_data.shape[0]):
                sum_r += self.compute_recall_r(test_data[query, :], rel_data[query, :], topk)

            return sum_r / test_data.shape[0]

        elif (type == 'score'):
            sum_r = 0.0
            for query in range(test_data.shape[0]):
                sum_r += self.compute_recall_s(test_data[query, :], rel_data[query, :], topk)

            return sum_r / test_data.shape[0]

    def eval_r1(self, test_data, rel_data):
        test_data.columns = ['Query', 'Item Code', 'Item Rank']
        rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']
        unique_queries = test_data['Query'].unique()
        sum_r = 0.0
        for query in unique_queries:
            query_test_data = test_data[test_data['Query'] == query]
            query_rel_data = rel_data[rel_data['Query'] == query]
            ra_list, rel_list = self.covert_pd_to_csv(query_test_data, query_rel_data)
            sum_r += self.compute_P_r(ra_list, rel_list, 1)

        return sum_r / len(unique_queries)

    def eval_r(self, test_data, rel_data, topk=1):
        test_data.columns = ['Query', 'Item Code', 'Item Rank']
        rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']
        unique_queries = test_data['Query'].unique()
        sum_r = 0.0
        for query in unique_queries:
            query_test_data = test_data[test_data['Query'] == query]
            query_rel_data = rel_data[rel_data['Query'] == query]
            ra_list, rel_list = self.covert_pd_to_csv(query_test_data, query_rel_data)
            sum_r += self.compute_R_r(ra_list, rel_list, topk)

        return sum_r / len(unique_queries)

    def save_rank_by_query(self, test_data, rel_data, topk=None):
        """
        Save the rank of each query to a csv file
        """
        test_data.columns = ['Query', 'Item Code', 'Item Rank']
        rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']
        unique_queries = test_data['Query'].unique()
        sum_r = 0.0
        data = []
        for query in unique_queries:
            query_test_data = test_data[test_data['Query'] == query]
            query_rel_data = rel_data[rel_data['Query'] == query]
            ra_list, rel_list = self.covert_pd_to_csv(query_test_data, query_rel_data)
            r_value = self.compute_R_r(ra_list, rel_list, topk)
            sum_r += r_value
            data.append([query, r_value])
        with open(f'rank@{topk}.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Query', f'Rank@{topk}'])
            writer.writerows(data)

    def eval_recall(self, test_data, rel_data, topk=1):
        test_data.columns = ['Query', 'Item Code', 'Item Rank']
        rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']
        unique_queries = test_data['Query'].unique()
        sum_r = 0.0
        for query in unique_queries:
            query_test_data = test_data[test_data['Query'] == query]
            query_rel_data = rel_data[rel_data['Query'] == query]
            ra_list, rel_list = self.covert_pd_to_csv(query_test_data, query_rel_data)
            sum_r += self.compute_recall_r(ra_list, rel_list, topk)

        return sum_r / len(unique_queries)

    def eval_p(self, test_data, rel_data, topk=None):
        test_data.columns = ['Query', 'Item Code', 'Item Rank']
        rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']
        unique_queries = test_data['Query'].unique()
        sum_p = 0.0
        for query in unique_queries:
            query_test_data = test_data[test_data['Query'] == query]
            query_rel_data = rel_data[rel_data['Query'] == query]
            ra_list, rel_list = self.covert_pd_to_csv(query_test_data, query_rel_data)
            if (topk is None):
                topk = np.sum(rel_list > 0)
            sum_p += self.compute_P_r(ra_list, rel_list, topk)

        return sum_p / len(unique_queries)

    """
    Compute ndcg

    compute_ndcg_r:
        list: 1 * item, 一维Numpy数组, 数组内存放排名rank(排名)
        rel_list: 1 * item, 一维Numpy数组, 数组内存放相关性分数
    """

    def compute_dcg(self, rank_list, rel_list, topk):
        dcg = 0.0

        if (topk > len(rel_list)):
            # print("Warning: Calculate NDCG metrics where topk is greater than the number of items")
            topk = len(rel_list)

        for i in range(topk):
            item_id = rank_list[i]
            rel = rel_list[item_id]
            dcg += (2 ** rel - 1) / math.log(i + 2, 2)

        return dcg

    def compute_ndcg_r(self, list, rel_list, topk):
        # 数组内存放item编号
        rank_list = np.argsort(list)
        rank_ideal_list = np.argsort(rel_list)[::-1]
        dcg = self.compute_dcg(rank_list, rel_list, topk)
        idcg = self.compute_dcg(rank_ideal_list, rel_list, topk)

        if (idcg == 0):
            # print("Warning: There is a situation where idcg is 0 in the evaluation")
            return 0
        else:
            return dcg / idcg

    """
    eval_ndcg:
        用于对算法输出的结果进行评测
        test_data: 
            - csv文件格式
            - Query | Item Code | Item Rank
        rel_data:
            - Query | 0 | Item | Relevance
    """

    def eval_ndcg(self, test_data, rel_data, topk=None):
        test_data.columns = ['Query', 'Item Code', 'Item Rank']
        rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']

        unique_queries = test_data['Query'].unique()
        sum_ndcg = 0.0
        query_results = []
        for query in unique_queries:
            query_test_data = test_data[test_data['Query'] == query]

            query_rel_data = rel_data[rel_data['Query'] == query]
            ra_list, rel_list = self.covert_pd_to_csv(query_test_data, query_rel_data)
            if (topk is None):
                topk = np.sum(rel_list > 0)
            ndcg_value = self.compute_ndcg_r(ra_list, rel_list, topk);
            sum_ndcg += ndcg_value
            query_results.append([query - 1, ndcg_value])

        return sum_ndcg / len(unique_queries), query_results


if __name__ == '__main__':
    # RA_test_loc = r'C:\Users\2021\Desktop\Validate_SRA\Test_data\R_FLAGR_testdata_RRAexact.csv'
    # RA_rel_loc = r'C:\Users\2021\Desktop\Validate_SRA\Test_data\FLAGR_testdata_qrels.csv'
    # RA_test_data = pd.read_csv(RA_test_loc, header=None)
    # RA_rel_data = pd.read_csv(RA_rel_loc, header=None)
    # evalution = Evaluation()
    # for i in range (20, 100, 20):
    #     p = evalution.eval_p(RA_test_data, RA_rel_data, i)
    #     print("p@{0}:{1}".format(i, p))
    """
    Validate Weighted Borda
    """

    # sum_ndcg = 0.0
    # sum_p = 0.0
    # sum_map = 0.0
    # for i in range(1, 5 + 1):
    #     RA_test_loc = r'C:\Users\2021\Desktop\Validate_SRA\Validate_Weighted_Borda\result_MQ2008-agg\version_np\result_supBorda_MQ2008-agg-Fold' + str(i) + '.csv'
    #     RA_rel_loc = r'C:\Users\2021\Desktop\Validate_SRA\Dataset\Tac_MQ2008-agg\Fold' + str(i) + r'\rel_test.csv'
    #     evalution = Evaluation()
    #     RA_test_data = pd.read_csv(RA_test_loc, header=None)
    #     RA_rel_data = pd.read_csv(RA_rel_loc, header=None)
    #     sum_ndcg += evalution.eval_ndcg(RA_test_data, RA_rel_data, 10)
    #     sum_p += evalution.eval_p(RA_test_data, RA_rel_data, 10)
    #     sum_map += evalution.eval_map(RA_test_data, RA_rel_data,10)
    # print("ndcg@{0}:{1}".format(10, sum_ndcg / 5))
    # print("precision@{0}:{1}".format(10, sum_p / 5))
    # print("map:{0}".format(sum_map / 5))

    """
    Validate SupCondorcet
    """
    # RA_test_loc = r'C:\Users\2021\Desktop\Validate_SRA\Validate_Supervised_Condorcet\result_MQ2008-agg\result_SupCondorcet_MQ2008-agg-Fold1.csv'
    # RA_rel_loc = r'C:\Users\2021\Desktop\Validate_SRA\Dataset\Tac_MQ2008-agg\Fold1\rel_test.csv'
    # evalution = Evaluation()
    # RA_test_data = pd.read_csv(RA_test_loc, header=None)
    # RA_rel_data = pd.read_csv(RA_rel_loc, header=None)

    # ndcg = evalution.eval_ndcg(RA_test_data, RA_rel_data, 10)
    # print("ndcg@{0}:{1}".format(10, ndcg))

    # p = evalution.eval_p(RA_test_data, RA_rel_data, 10)
    # print("precision@{0}:{1}".format(10, p))

    # map = evalution.eval_map(RA_test_data, RA_rel_data, 10)
    # print("map:{0}".format(map))

    """
    Evolution
    """
    methods_list = []
    ap_list = []
    ndcg_list = []

    # rel_path = r'D:\RA_ReID\Evaluation\mlm-test-rel.mat'
    rel_path = 'data\\simulation_test_rel.mat'
    R_topk = 1
    NDCG_topk = 10
    MAP_topk = 10

    test_path = r'ans\rank-result-simulation-dataset-CG.mat'
    evaluation = Evaluation()
    r1 = evaluation.eval_r_matlab(test_path, rel_path, 'result', 'testrel', 'rank', topk=R_topk)
    map, ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'result', 'testrel', 'rank', MAP_topk)
    ndcg, ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'result', 'testrel', 'rank', NDCG_topk)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('CG')
    print("CG: R@{2}: {0}, Map@{5}: {1}, NDCG@{3}: {4}".format(r1, map, R_topk, NDCG_topk, ndcg, MAP_topk))

    test_path = r'ans\rank-result-simulation-dataset-CombMAX.mat'
    evaluation = Evaluation()
    r1 = evaluation.eval_r_matlab(test_path, rel_path, 'result', 'testrel', 'rank', topk=R_topk)
    map, ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'result', 'testrel', 'rank', MAP_topk)
    ndcg, ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'result', 'testrel', 'rank', NDCG_topk)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('CombMAX')
    print("CombMAX: R@{2}: {0}, Map@{5}: {1}, NDCG@{3}: {4}".format(r1, map, R_topk, NDCG_topk, ndcg, MAP_topk))

    test_path = r'ans\rank-result-simulation-dataset-CombMIN.mat'
    evaluation = Evaluation()
    r1 = evaluation.eval_r_matlab(test_path, rel_path, 'result', 'testrel', 'rank', topk=R_topk)
    map, ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'result', 'testrel', 'rank', 10)
    ndcg, ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'result', 'testrel', 'rank', NDCG_topk)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('CombMIN')
    print("CombMIN: R@{2}: {0}, Map@{5}: {1}, NDCG@{3}: {4}".format(r1, map, R_topk, NDCG_topk, ndcg, MAP_topk))

    print('-----------------------------------------------------------------------------')

    test_path = r'ans\rank-result-simulation-dataset-Dowdall.mat'
    evaluation = Evaluation()
    r1 = evaluation.eval_r_matlab(test_path, rel_path, 'score', 'testrel', 'rank', topk=R_topk)
    map, ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'score', 'testrel', 'rank', MAP_topk)
    ndcg, ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'score', 'testrel', 'rank', NDCG_topk)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('Dowdall')
    print("Dowdall: R@{2}: {0}, Map@{5}: {1}, NDCG@{3}: {4}".format(r1, map, R_topk, NDCG_topk, ndcg, MAP_topk))

    test_path = r'ans\rank-result-simulation-dataset-ER.mat'
    evaluation = Evaluation()
    r1 = evaluation.eval_r_matlab(test_path, rel_path, 'result', 'testrel', 'rank', topk=R_topk)
    map, ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'result', 'testrel', 'rank', MAP_topk)
    ndcg, ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'result', 'testrel', 'rank', NDCG_topk)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('ER')
    print("ER: R@{2}: {0}, Map@{5}: {1}, NDCG@{3}: {4}".format(r1, map, R_topk, NDCG_topk, ndcg, MAP_topk))

    test_path = r'ans\rank-result-simulation-dataset-hparunme.mat'
    evaluation = Evaluation()
    r1 = evaluation.eval_r_matlab(test_path, rel_path, 'result', 'testrel', 'rank', topk=R_topk)
    map, ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'result', 'testrel', 'rank', MAP_topk)
    ndcg, ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'result', 'testrel', 'rank', NDCG_topk)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('HPA')
    print("HPA : R@{2}: {0}, Map@{5}: {1}, NDCG@{3}: {4}".format(r1, map, R_topk, NDCG_topk, ndcg, MAP_topk))

    print('-----------------------------------------------------------------------------')

    test_path = r'ans\rank-result-simulation-dataset-iRANK.mat'
    evaluation = Evaluation()
    r1 = evaluation.eval_r_matlab(test_path, rel_path, 'result', 'testrel', 'rank', topk=R_topk)
    map, ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'result', 'testrel', 'rank', MAP_topk)
    ndcg, ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'result', 'testrel', 'rank', NDCG_topk)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('iRANK')
    print("iRANK : R@{2}: {0}, Map@{5}: {1}, NDCG@{3}: {4}".format(r1, map, R_topk, NDCG_topk, ndcg, MAP_topk))

    test_path = r'ans\rank-result-simulation-dataset-PostNDCG.mat'
    evaluation = Evaluation()
    r1 = evaluation.eval_r_matlab(test_path, rel_path, 'result', 'testrel', 'rank', topk=R_topk)
    map, ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'result', 'testrel', 'rank', MAP_topk)
    ndcg, ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'result', 'testrel', 'rank', NDCG_topk)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('PostNDCG')
    print("PostNDCG : R@{2}: {0}, Map@{5}: {1}, NDCG@{3}: {4}".format(r1, map, R_topk, NDCG_topk, ndcg, MAP_topk))

    test_path = r'ans\result-simulation-dataset-BordaCount.mat'
    evaluation = Evaluation()
    r1 = evaluation.eval_r_matlab(test_path, rel_path, 'score', 'testrel', 'score', topk=R_topk)
    map, ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'score', 'testrel', 'score', MAP_topk)
    ndcg, ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'score', 'testrel', 'score', NDCG_topk)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('Borda')
    print("Borda : R@{2}: {0}, Map@{5}: {1}, NDCG@{3}: {4}".format(r1, map, R_topk, NDCG_topk, ndcg, MAP_topk))

    print('-----------------------------------------------------------------------------')

    test_path = r'ans\result-simulation-dataset-CombANZ.mat'
    evaluation = Evaluation()
    r1 = evaluation.eval_r_matlab(test_path, rel_path, 'res', 'testrel', 'score', topk=R_topk)
    map, ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'res', 'testrel', 'score', MAP_topk)
    ndcg, ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'res', 'testrel', 'score', NDCG_topk)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('CombANZ')
    print("CombANZ : R@{2}: {0}, Map@{5}: {1}, NDCG@{3}: {4}".format(r1, map, R_topk, NDCG_topk, ndcg, MAP_topk))

    test_path = r'ans\result-simulation-dataset-CombMNZ.mat'
    evaluation = Evaluation()
    r1 = evaluation.eval_r_matlab(test_path, rel_path, 'res', 'testrel', 'score', topk=R_topk)
    map, ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'res', 'testrel', 'score', MAP_topk)
    ndcg, ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'res', 'testrel', 'score', NDCG_topk)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('CombMNZ')
    print("CombMNZ : R@{2}: {0}, Map@{5}: {1}, NDCG@{3}: {4}".format(r1, map, R_topk, NDCG_topk, ndcg, MAP_topk))

    test_path = r'ans\result-simulation-dataset-CombSUM.mat'
    evaluation = Evaluation()
    r1 = evaluation.eval_r_matlab(test_path, rel_path, 'res', 'testrel', 'score', topk=R_topk)
    map, ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'res', 'testrel', 'score', MAP_topk)
    ndcg, ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'res', 'testrel', 'score', NDCG_topk)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('CombSUM')
    print("CombSUM : R@{2}: {0}, Map@{5}: {1}, NDCG@{3}: {4}".format(r1, map, R_topk, NDCG_topk, ndcg, MAP_topk))

    print('-----------------------------------------------------------------------------')

    test_path = r'ans\result-simulation-dataset-Mean.mat'
    evaluation = Evaluation()
    r1 = evaluation.eval_r_matlab(test_path, rel_path, 'res', 'testrel', 'score', topk=R_topk)
    map, ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'res', 'testrel', 'score', MAP_topk)
    ndcg, ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'res', 'testrel', 'score', NDCG_topk)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('Mean')
    print("Mean : R@{2}: {0}, Map@{5}: {1}, NDCG@{3}: {4}".format(r1, map, R_topk, NDCG_topk, ndcg, MAP_topk))

    test_path = r'ans\result-simulation-dataset-Medium.mat'
    evaluation = Evaluation()
    r1 = evaluation.eval_r_matlab(test_path, rel_path, 'res', 'testrel', 'score', topk=R_topk)
    map, ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'res', 'testrel', 'score', MAP_topk)
    ndcg, ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'res', 'testrel', 'score', NDCG_topk)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('Medium')
    print("Medium : R@{2}: {0}, Map@{5}: {1}, NDCG@{3}: {4}".format(r1, map, R_topk, NDCG_topk, ndcg, MAP_topk))

    test_path = r'ans\result-simulation-dataset-rrf.mat'
    evaluation = Evaluation()
    r1 = evaluation.eval_r_matlab(test_path, rel_path, 'res', 'testrel', 'score', topk=R_topk)
    map, ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'res', 'testrel', 'score', MAP_topk)
    ndcg, ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'res', 'testrel', 'score', NDCG_topk)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('rrf')
    print("rrf : R@{2}: {0}, Map@{5}: {1}, NDCG@{3}: {4}".format(r1, map, R_topk, NDCG_topk, ndcg, MAP_topk))

    print('-----------------------------------------------------------------------------')

    test_path = r'ans\result-simulation-dataset-CombMED.mat'
    evaluation = Evaluation()
    r1 = evaluation.eval_r_matlab(test_path, rel_path, 'final_score', 'testrel', 'score', topk=R_topk)
    map, ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'final_score', 'testrel', 'score', MAP_topk)
    ndcg, ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'final_score', 'testrel', 'score',
                                                           NDCG_topk)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('CombMED')
    print("CombMED : R@{2}: {0}, Map@{5}: {1}, NDCG@{3}: {4}".format(r1, map, R_topk, NDCG_topk, ndcg, MAP_topk))

    test_path = r'ans\result-simulation-dataset-DIBRA.mat'
    evaluation = Evaluation()
    r1 = evaluation.eval_r_matlab(test_path, rel_path, 'new_L', 'testrel', 'score', topk=R_topk)
    map, ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'new_L', 'testrel', 'score', MAP_topk)
    ndcg, ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'new_L', 'testrel', 'score', NDCG_topk)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('DIBRA')
    print("DIBRA : R@{2}: {0}, Map@{5}: {1}, NDCG@{3}: {4}".format(r1, map, R_topk, NDCG_topk, ndcg, MAP_topk))

    # for k in range(1,50):
    #     for T in range(1,101):
    #         test_path = r'D:\RA_ReID\MovieLens1M\quantile0.95\IRAr_0.95\result-simulation-dataset-IRA-R-k' + str(k) + '-T' + str(T) + '.mat'
    #          # 检查文件是否存在
    #         if os.path.exists(test_path):
    #             evaluation = Evaluation()
    #             r1 = evaluation.eval_r1_matlab(test_path, rel_path, 'new_sim', 'testrel','score')
    #             map,ap_query_results = evaluation.eval_map_matlab(test_path,rel_path, 'new_sim','testrel' ,'score', MAP_topk)
    #             ndcg,ndcg_query_results = evaluation.eval_ndcg_matlab(test_path,rel_path, 'new_sim','testrel' ,'score', NDCG_topk)
    #             ap_list.append(ap_query_results)
    #             ndcg_list.append(ndcg_query_results)
    #             methods_list.append('IRA-R-k' + str(k) + '-T' + str(T))
    #             print("IRA-R-k{2}-T{3} : R@1: {0}, Map@10: {1}".format(r1, map, k, T))

    print('-----------------------------------------------------------------------------')

    # for k in range(1,50):
    #     for T in range(1,101):
    #         test_path = r'D:\RA_ReID\MovieLens1M\quantile0.95\IRAs_0.95\result-simulation-dataset-IRA-S-k' + str(k) + '-T' + str(T) + '.mat'
    #          # 检查文件是否存在
    #         if os.path.exists(test_path):
    #             evaluation = Evaluation()
    #             r1 = evaluation.eval_r1_matlab(test_path, rel_path, 'new_sim', 'testrel','score')
    #             map,ap_query_results = evaluation.eval_map_matlab(test_path,rel_path, 'new_sim','testrel' ,'score', MAP_topk)
    #             ndcg,ndcg_query_results = evaluation.eval_ndcg_matlab(test_path,rel_path, 'new_sim','testrel' ,'score', NDCG_topk)
    #             ap_list.append(ap_query_results)
    #             ndcg_list.append(ndcg_query_results)
    #             methods_list.append('IRA-S-k' + str(k) + '-T' + str(T))
    #             print("IRA-S-k{2}-T{3} : R@1: {0}, Map@10: {1}".format(r1, map, k, T))

    print('-----------------------------------------------------------------------------')

    # for k in range(1, 10):
    #     for T in range(1, 10):
    #         test_path = r'D:\RA_ReID\MovieLens1M\quantile0.95\QI-IRA_0.95\result-simulation-dataset-QI-IRA-k' + str(k) + '-T' + str(
    #             T) + '.mat'
    #         # 检查文件是否存在
    #         if os.path.exists(test_path):
    #             evaluation = Evaluation()
    #             r1 = evaluation.eval_r1_matlab(test_path, rel_path, 'new_sim', 'testrel', 'score')
    #             map,ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'new_sim', 'testrel', 'score', MAP_topk)
    #             ndcg,ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'new_sim', 'testrel', 'score', NDCG_topk)
    #             ap_list.append(ap_query_results)
    #             ndcg_list.append(ndcg_query_results)
    #             methods_list.append('QT-IRA-k' + str(k) + '-T' + str(T))
    #             print("QT-IRA-k{0}-T{1}: Map@{2}: {3}, NDCG@{4}: {5}".format(k, T, MAP_topk, map, NDCG_topk, ndcg))

    print('-----------------------------------------------------------------------------')

    # for k in range(1, 10):
    #     for T in range(1, 10):
    #         test_path = r"D:\RA_ReID\MovieLens1M\quantile0.95\popularity_QUARK_0.95\popularity-simulation-dataset-QUARK-k" + str(k) + "-T" + str(T) + ".mat"
    #         if os.path.exists(test_path):
    #             evaluation = Evaluation()
    #             r1 = evaluation.eval_r1_matlab(test_path, rel_path, 'new_sim', 'testrel', 'score')
    #             map,ap_query_results = evaluation.eval_map_matlab(test_path, rel_path, 'new_sim', 'testrel', 'score', MAP_topk)
    #             ndcg,ndcg_query_results = evaluation.eval_ndcg_matlab(test_path, rel_path, 'new_sim', 'testrel', 'score', NDCG_topk)
    #             ap_list.append(ap_query_results)
    #             ndcg_list.append(ndcg_query_results)
    #             methods_list.append('QUARK-k' + str(k) + '-T' + str(T))
    #             print("QUARK-k{0}-T{1}: Map@{2}: {3}, NDCG@{4}: {5}".format(k, T, MAP_topk, map, NDCG_topk, ndcg))

    # 将结果存储到Excel文件中
    # ap_dfs = [pd.DataFrame(lst, columns=['Query', methods_list[i]]) for i, lst in enumerate(ap_list)]
    # ndcg_dfs = [pd.DataFrame(lst, columns=['Query', methods_list[i]]) for i, lst in enumerate(ndcg_list)]
    # merged_ap_df = reduce(lambda left, right: pd.merge(left, right, on='Query', how='outer'), ap_dfs)
    # merged_ndcg_df = reduce(lambda left, right: pd.merge(left, right, on='Query', how='outer'), ndcg_dfs)
    # merged_ap_df.to_excel('ans\\ap_results.xlsx', index=False, sheet_name='sheet1')
    # merged_ndcg_df.to_excel('ans\\ndcg_results.xlsx', index=False, sheet_name='sheet1')
    """
    Supervised RA
    """

"""
    RA_rel_loc = r"D:\RA_ReID\MovieLens1M\quantile0.95\m1m-top20-test-rel-0.95.csv"

    print("Running ...")
    RA_test_loc = r"D:\RA_ReID\Evaluation\ans\result-Mork-m1m-top20-test.csv"
    evalution = Evaluation()
    RA_test_data = pd.read_csv(RA_test_loc, header=None)
    RA_rel_data = pd.read_csv(RA_rel_loc, header=None)


    map,ap_query_results = evalution.eval_map(RA_test_data, RA_rel_data, 10)
    ndcg,ndcg_query_results = evalution.eval_ndcg(RA_test_data, RA_rel_data, 10)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('Mork')
    print("Mork over!")

    RA_test_loc = r"D:\RA_ReID\MovieLens1M\quantile0.95\result_crf_MovieLens1M_0.95.csv"
    evalution = Evaluation()
    RA_test_data = pd.read_csv(RA_test_loc, header=None)
    RA_rel_data = pd.read_csv(RA_rel_loc, header=None)


    map,ap_query_results = evalution.eval_map(RA_test_data, RA_rel_data, 10)
    ndcg,ndcg_query_results = evalution.eval_ndcg(RA_test_data, RA_rel_data, 10)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('CRF')
    print("CRF over!")

    print("Running ...")
    RA_test_loc = r"D:\RA_ReID\MovieLens1M\quantile0.95\result_AggRankDE_MovieLens1M_0.95.csv"
    evalution = Evaluation()
    RA_test_data = pd.read_csv(RA_test_loc, header=None)
    RA_rel_data = pd.read_csv(RA_rel_loc, header=None)


    map,ap_query_results = evalution.eval_map(RA_test_data, RA_rel_data, 10)
    ndcg,ndcg_query_results = evalution.eval_ndcg(RA_test_data, RA_rel_data, 10)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('AggRankDE')
    print("AggRankDE over!")

    print("Running ...")
    RA_test_loc = r"D:\RA_ReID\MovieLens1M\quantile0.95\result_wBorda_MovieLens1M_0.95.csv"
    evalution = Evaluation()
    RA_test_data = pd.read_csv(RA_test_loc, header=None)
    RA_rel_data = pd.read_csv(RA_rel_loc, header=None)


    map,ap_query_results = evalution.eval_map(RA_test_data, RA_rel_data, 10)
    ndcg,ndcg_query_results = evalution.eval_ndcg(RA_test_data, RA_rel_data, 10)
    ap_list.append(ap_query_results)
    ndcg_list.append(ndcg_query_results)
    methods_list.append('wBorda')
    print("wBorda over!")


"""
