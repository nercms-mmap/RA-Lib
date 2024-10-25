"""
确保csv和mat两种版本的方法评测结果相同
"""

import unittest

import pandas as pd

from common.constant import InputType
from evaluation import Evaluation
from scipy.io import loadmat


class MetricsTestCase(unittest.TestCase):
    def setUp(self):
        self.mat_data_file = "data\\mat_results_rankbased.mat"
        self.mat_rel_file = "data\\mat_rel.mat"
        self.csv_data_file = "data\\trans_mat_results_rankbased.csv"
        self.csv_rel_file = "data\\trans_mat_rel.csv"
        self.csv_data = pd.read_csv(self.csv_data_file, header=None)
        self.csv_rel = pd.read_csv(self.csv_rel_file, header=None)
        self.eval = Evaluation()

    def test_map(self):
        mat_map = self.eval.eval_mean_average_precision(self.mat_data_file, self.mat_rel_file,
                                                        get_first_key(self.mat_data_file),
                                                        get_first_key(self.mat_rel_file), InputType.RANK, 10)
        csv_map = self.eval.eval_mean_average_precision(self.csv_data, self.csv_rel, 10)
        self.assertEqual(csv_map, mat_map)  # add assertion here

    def test_ndcg(self):
        mat_ndcg = self.eval.eval_ndcg(self.mat_data_file, self.mat_rel_file,
                                       get_first_key(self.mat_data_file),
                                       get_first_key(self.mat_rel_file), InputType.RANK, 10)
        csv_ndcg = self.eval.eval_ndcg(self.csv_data, self.csv_rel, 10)
        self.assertEqual(csv_ndcg, mat_ndcg)

    def test_rank(self):
        mat_rank = self.eval.eval_rank(self.mat_data_file, self.mat_rel_file,
                                       get_first_key(self.mat_data_file),
                                       get_first_key(self.mat_rel_file), InputType.RANK, 1)
        csv_rank = self.eval.eval_rank(self.csv_data, self.csv_rel, 1)
        self.assertEqual(csv_rank, mat_rank)


def get_first_key(file_path):
    mat_data = loadmat(file_path)
    keys = list(mat_data.keys())
    filtered_keys = [key for key in keys if not key.startswith('__')]
    return filtered_keys[0]


if __name__ == '__main__':
    unittest.main()
