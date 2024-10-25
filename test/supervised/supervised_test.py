import unittest

import pandas as pd

from src.rapython.common.constant import InputType
from src.rapython.evaluation import Evaluation
from scipy.io import loadmat

from src.rapython.supervised import MethodType


class SupervisedTestCase(unittest.TestCase):
    def setUp(self):
        self.mat_rel_test_path = "..\\full_lists\\data\\simulation_test_rel.mat"
        self.csv_rel_test_path = "..\\full_lists\\data\\simulation_test_rel.csv"
        self.input_file_path = "..\\full_lists\\data\\simulation_test.csv"
        self.rel_data = pd.read_csv(self.csv_rel_test_path, header=None)
        self.eval = Evaluation()
        self.map_topk = 10
        self.ndcg_topk = 10
        self.rank_topk = 1

    def processtest(self, ans_file_path, test_data, m_input_type):
        data_key = get_first_key(ans_file_path)
        rel_key = get_first_key(self.mat_rel_test_path)
        my_map = self.eval.eval_mean_average_precision(test_data, self.rel_data, self.map_topk)
        ans_map = self.eval.eval_mean_average_precision(ans_file_path, self.mat_rel_test_path, data_key, rel_key,
                                                        m_input_type, self.map_topk)
        self.assertEqual(ans_map, my_map)

        my_ndcg = self.eval.eval_ndcg(test_data, self.rel_data, self.ndcg_topk)
        ans_ndcg = self.eval.eval_ndcg(ans_file_path, self.mat_rel_test_path, data_key, rel_key,
                                       m_input_type, self.ndcg_topk)
        self.assertEqual(ans_ndcg, my_ndcg)

        my_rank = self.eval.eval_rank(test_data, self.rel_data, self.rank_topk)
        ans_rank = self.eval.eval_rank(ans_file_path, self.mat_rel_test_path, data_key, rel_key,
                                       m_input_type, self.rank_topk)
        self.assertEqual(ans_rank, my_rank)

    def test_ira_r(self):
        from src.rapython.supervised import ira
        output_file_path = "my_results\\my_ira_r.csv"
        ira(self.input_file_path, output_file_path, self.csv_rel_test_path, 3, 2, 0.02, MethodType.IRA_RANK,
            InputType.RANK)
        ans_file_path = "..\\full_lists\\ans\\result-simulation-dataset-IRA-R-k3-T2.mat"
        test_data = pd.read_csv(output_file_path, header=None)
        self.processtest(ans_file_path, test_data, InputType.SCORE)

    def test_ira_s(self):
        from src.rapython.supervised import ira
        output_file_path = "my_results\\my_ira_s.csv"
        ira(self.input_file_path, output_file_path, self.csv_rel_test_path, 3, 2, 0.02, MethodType.IRA_SCORE,
            InputType.RANK)
        ans_file_path = "..\\full_lists\\ans\\result-simulation-dataset-IRA-S-k3-T2.mat"
        test_data = pd.read_csv(output_file_path, header=None)
        self.processtest(ans_file_path, test_data, InputType.SCORE)

    def test_qi_ira(self):
        from src.rapython.supervised import qi_ira
        output_file_path = "my_results\\my_ira_s.csv"
        qi_ira(self.input_file_path, output_file_path, self.csv_rel_test_path, 3, 2, 0.02,
               InputType.RANK)
        ans_file_path = "..\\full_lists\\ans\\result-simulation-dataset-QI-IRA-k3-T2.mat"
        test_data = pd.read_csv(output_file_path, header=None)
        self.processtest(ans_file_path, test_data, InputType.SCORE)


def get_first_key(file_path):
    mat_data = loadmat(file_path)
    keys = list(mat_data.keys())
    filtered_keys = [key for key in keys if not key.startswith('__')]
    return filtered_keys[0]


if __name__ == '__main__':
    unittest.main()
