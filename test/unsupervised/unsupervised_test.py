import os.path
import unittest

import pandas as pd

from src.rapython.common.constant import InputType
from src.rapython.evaluation import Evaluation
from scipy.io import loadmat

from src.rapython.unsupervised import McType


class UnsupervisedTestVsPythonCase(unittest.TestCase):
    def setUp(self):
        self.ans_base_path = "ans_results"
        self.test_base_path = "my_results"
        self.input_file_path = "..\\full_lists\\data\\simulation_test.csv"
        self.input_rel_path = "..\\full_lists\\data\\simulation_test_rel.csv"
        self.rel_data = pd.read_csv(self.input_rel_path, header=None)
        self.map_topk = 10
        self.ndcg_topk = 10
        self.rank_topk = 1

    def processtest(self, my_data, ans_data):
        my_ndcg = Evaluation().eval_ndcg(my_data, self.rel_data, self.ndcg_topk)
        ans_ndcg = Evaluation().eval_ndcg(ans_data, self.rel_data, self.ndcg_topk)
        self.assertEqual(ans_ndcg, my_ndcg)
        my_map = Evaluation().eval_mean_average_precision(my_data, self.rel_data, self.map_topk)
        ans_map = Evaluation().eval_mean_average_precision(ans_data, self.rel_data, self.map_topk)
        self.assertEqual(ans_map, my_map)
        my_rank = Evaluation().eval_rank(my_data, self.rel_data, self.rank_topk)
        ans_rank = Evaluation().eval_rank(ans_data, self.rel_data, self.rank_topk)
        self.assertEqual(ans_rank, my_rank)

    def test_cg(self):
        from src.rapython.unsupervised import cg
        cg(self.input_file_path, os.path.join(self.test_base_path, f"my_{cg.__name__}.csv"))
        from CG import CG
        CG(self.input_file_path, os.path.join(self.ans_base_path, f"ans_{CG.__name__}.csv"))

        my_data = pd.read_csv(os.path.join(self.test_base_path, f"my_{cg.__name__}.csv"), header=None)
        ans_data = pd.read_csv(os.path.join(self.ans_base_path, f"ans_{CG.__name__}.csv"), header=None)
        self.processtest(my_data, ans_data)

    def test_borda_score(self):
        from src.rapython.unsupervised import borda_score
        borda_score(self.input_file_path, os.path.join(self.test_base_path, f"my_{borda_score.__name__}.csv"))
        from Borda_Score import Borda_Score
        Borda_Score(self.input_file_path, os.path.join(self.ans_base_path, f"ans_{Borda_Score.__name__}.csv"))
        my_data = pd.read_csv(os.path.join(self.test_base_path, f"my_{borda_score.__name__}.csv"), header=None)
        ans_data = pd.read_csv(os.path.join(self.ans_base_path, f"ans_{Borda_Score.__name__}.csv"), header=None)
        self.processtest(my_data, ans_data)

    def test_bordacount(self):
        from src.rapython.unsupervised import bordacount
        bordacount(self.input_file_path, os.path.join(self.test_base_path, f"my_{bordacount.__name__}.csv"))
        from BordaCount import BordaCount
        BordaCount(self.input_file_path, os.path.join(self.ans_base_path, f"ans_{BordaCount.__name__}.csv"))
        my_data = pd.read_csv(os.path.join(self.test_base_path, f"my_{bordacount.__name__}.csv"), header=None)
        ans_data = pd.read_csv(os.path.join(self.ans_base_path, f"ans_{BordaCount.__name__}.csv"), header=None)
        self.processtest(my_data, ans_data)

    def test_combanz(self):
        from src.rapython.unsupervised import combanz
        combanz(self.input_file_path, os.path.join(self.test_base_path, f"my_{combanz.__name__}.csv"))
        from CombANZ import CombANZ
        CombANZ(self.input_file_path, os.path.join(self.ans_base_path, f"ans_{CombANZ.__name__}.csv"))
        my_data = pd.read_csv(os.path.join(self.test_base_path, f"my_{combanz.__name__}.csv"), header=None)
        ans_data = pd.read_csv(os.path.join(self.ans_base_path, f"ans_{CombANZ.__name__}.csv"), header=None)
        self.processtest(my_data, ans_data)

    def test_combmax(self):
        from src.rapython.unsupervised import combmax
        combmax(self.input_file_path, os.path.join(self.test_base_path, f"my_{combmax.__name__}.csv"))
        from CombMAX import CombMAX
        CombMAX(self.input_file_path, os.path.join(self.ans_base_path, f"ans_{CombMAX.__name__}.csv"))
        my_data = pd.read_csv(os.path.join(self.test_base_path, f"my_{combmax.__name__}.csv"), header=None)
        ans_data = pd.read_csv(os.path.join(self.ans_base_path, f"ans_{CombMAX.__name__}.csv"), header=None)
        self.processtest(my_data, ans_data)

    def test_combmed(self):
        from src.rapython.unsupervised import combmed
        combmed(self.input_file_path, os.path.join(self.test_base_path, f"my_{combmed.__name__}.csv"))
        from CombMED import CombMED
        CombMED(self.input_file_path, os.path.join(self.ans_base_path, f"ans_{CombMED.__name__}.csv"))
        my_data = pd.read_csv(os.path.join(self.test_base_path, f"my_{combmed.__name__}.csv"), header=None)
        ans_data = pd.read_csv(os.path.join(self.ans_base_path, f"ans_{CombMED.__name__}.csv"), header=None)
        self.processtest(my_data, ans_data)

    def test_combmin(self):
        from src.rapython.unsupervised import combmin
        combmin(self.input_file_path, os.path.join(self.test_base_path, f"my_{combmin.__name__}.csv"))
        from CombMIN import CombMIN
        CombMIN(self.input_file_path, os.path.join(self.ans_base_path, f"ans_{CombMIN.__name__}.csv"))
        my_data = pd.read_csv(os.path.join(self.test_base_path, f"my_{combmin.__name__}.csv"), header=None)
        ans_data = pd.read_csv(os.path.join(self.ans_base_path, f"ans_{CombMIN.__name__}.csv"), header=None)
        self.processtest(my_data, ans_data)

    def test_combmnz(self):
        from src.rapython.unsupervised import combmnz
        combmnz(self.input_file_path, os.path.join(self.test_base_path, f"my_{combmnz.__name__}.csv"))
        from CombMNZ import CombMNZ
        CombMNZ(self.input_file_path, os.path.join(self.ans_base_path, f"ans_{CombMNZ.__name__}.csv"))
        my_data = pd.read_csv(os.path.join(self.test_base_path, f"my_{combmnz.__name__}.csv"), header=None)
        ans_data = pd.read_csv(os.path.join(self.ans_base_path, f"ans_{CombMNZ.__name__}.csv"), header=None)
        self.processtest(my_data, ans_data)

    def test_combsum(self):
        from src.rapython.unsupervised import combsum
        combsum(self.input_file_path, os.path.join(self.test_base_path, f"my_{combsum.__name__}.csv"))
        from CombSUM import CombSUM
        CombSUM(self.input_file_path, os.path.join(self.ans_base_path, f"ans_{CombSUM.__name__}.csv"))
        my_data = pd.read_csv(os.path.join(self.test_base_path, f"my_{combsum.__name__}.csv"), header=None)
        ans_data = pd.read_csv(os.path.join(self.ans_base_path, f"ans_{CombSUM.__name__}.csv"), header=None)
        self.processtest(my_data, ans_data)

    def test_dowdall(self):
        from src.rapython.unsupervised import dowdall
        dowdall(self.input_file_path, os.path.join(self.test_base_path, f"my_{dowdall.__name__}.csv"))
        from Dowdall import Dowdall
        Dowdall(self.input_file_path, os.path.join(self.ans_base_path, f"ans_{Dowdall.__name__}.csv"))
        my_data = pd.read_csv(os.path.join(self.test_base_path, f"my_{dowdall.__name__}.csv"), header=None)
        ans_data = pd.read_csv(os.path.join(self.ans_base_path, f"ans_{Dowdall.__name__}.csv"), header=None)
        self.processtest(my_data, ans_data)

    def test_markovchain(self):
        from src.rapython.unsupervised import markovchainmethod
        markovchainmethod(self.input_file_path,
                          os.path.join(self.test_base_path, f"my_{markovchainmethod.__name__}.csv"), McType.MC1)
        from MarkovChain import MarKovChainMethod
        MarKovChainMethod(self.input_file_path,
                          os.path.join(self.ans_base_path, f"ans_{MarKovChainMethod.__name__}.csv"), 'MC1')
        my_data = pd.read_csv(os.path.join(self.test_base_path, f"my_{markovchainmethod.__name__}.csv"), header=None)
        ans_data = pd.read_csv(os.path.join(self.ans_base_path, f"ans_{MarKovChainMethod.__name__}.csv"), header=None)
        self.processtest(my_data, ans_data)

    def test_mean(self):
        from src.rapython.unsupervised import mean
        mean(self.input_file_path, os.path.join(self.test_base_path, f"my_{mean.__name__}.csv"))
        from Mean import Mean
        Mean(self.input_file_path, os.path.join(self.ans_base_path, f"ans_{Mean.__name__}.csv"))
        my_data = pd.read_csv(os.path.join(self.test_base_path, f"my_{mean.__name__}.csv"), header=None)
        ans_data = pd.read_csv(os.path.join(self.ans_base_path, f"ans_{Mean.__name__}.csv"), header=None)
        self.processtest(my_data, ans_data)

    def test_median(self):
        from src.rapython.unsupervised import median
        median(self.input_file_path, os.path.join(self.test_base_path, f"my_{median.__name__}.csv"))
        from Medium import Medium
        Medium(self.input_file_path, os.path.join(self.ans_base_path, f"ans_{Medium.__name__}.csv"))
        my_data = pd.read_csv(os.path.join(self.test_base_path, f"my_{median.__name__}.csv"), header=None)
        ans_data = pd.read_csv(os.path.join(self.ans_base_path, f"ans_{Medium.__name__}.csv"), header=None)
        self.processtest(my_data, ans_data)

    def test_mork(self):
        from src.rapython.unsupervised import mork_heuristic
        mork_heuristic(self.input_file_path, os.path.join(self.test_base_path, f"my_{mork_heuristic.__name__}.csv"))
        from MorK_Heuristic_Maximum import Mork_Heuristic
        Mork_Heuristic(self.input_file_path, os.path.join(self.ans_base_path, f"ans_{Mork_Heuristic.__name__}.csv"))
        my_data = pd.read_csv(os.path.join(self.test_base_path, f"my_{mork_heuristic.__name__}.csv"), header=None)
        ans_data = pd.read_csv(os.path.join(self.ans_base_path, f"ans_{Mork_Heuristic.__name__}.csv"), header=None)
        self.processtest(my_data, ans_data)

    def test_rrf(self):
        from src.rapython.unsupervised import rrf
        rrf(self.input_file_path, os.path.join(self.test_base_path, f"my_{rrf.__name__}.csv"))
        from RRF import RRF
        RRF(self.input_file_path, os.path.join(self.ans_base_path, f"ans_{RRF.__name__}.csv"))
        my_data = pd.read_csv(os.path.join(self.test_base_path, f"my_{rrf.__name__}.csv"), header=None)
        ans_data = pd.read_csv(os.path.join(self.ans_base_path, f"ans_{RRF.__name__}.csv"), header=None)
        self.processtest(my_data, ans_data)


class UnsupervisedTestVsMatlabCase(unittest.TestCase):
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

    def test_cg(self):
        ans_file_path = "..\\full_lists\\ans\\result-simulation-dataset-CG.mat"
        from src.rapython.unsupervised.cg import cg
        cg(self.input_file_path, "my_results\\my_cg.csv")
        test_data = pd.read_csv("my_results\\my_cg.csv", header=None)
        self.processtest(ans_file_path, test_data, InputType.SCORE)

    def test_combmax(self):
        from src.rapython.unsupervised import combmax
        combmax(self.input_file_path, "my_results\\my_combmax.csv")
        ans_file_path = "..\\full_lists\\ans\\result-simulation-dataset-CombMAX.mat"
        test_data = pd.read_csv("my_results\\my_combmax.csv", header=None)
        self.processtest(ans_file_path, test_data, InputType.SCORE)

    def test_combmin(self):
        ans_file_path = "..\\full_lists\\ans\\rank-result-simulation-dataset-CombMIN.mat"
        test_data = pd.read_csv("..\\full_lists\\results\\combmin.csv", header=None)
        self.processtest(ans_file_path, test_data, InputType.RANK)

    def test_dowdall(self):
        ans_file_path = "..\\full_lists\\ans\\rank-result-simulation-dataset-Dowdall.mat"
        test_data = pd.read_csv("..\\full_lists\\results\\dowdall.csv", header=None)
        self.processtest(ans_file_path, test_data, InputType.RANK)

    def test_er(self):
        ans_file_path = "..\\full_lists\\ans\\rank-result-simulation-dataset-ER.mat"
        test_data = pd.read_csv("..\\full_lists\\results\\er.csv", header=None)
        self.processtest(ans_file_path, test_data, InputType.RANK)

    def test_hpa(self):
        from src.rapython.unsupervised import hpa
        hpa(self.input_file_path, "my_results\\my_hpa.csv", InputType.RANK)
        ans_file_path = "..\\full_lists\\ans\\rank-result-simulation-dataset-hparunme.mat"
        test_data = pd.read_csv("my_results\\my_hpa.csv", header=None)
        self.processtest(ans_file_path, test_data, InputType.RANK)

    def test_irank(self):
        ans_file_path = "..\\full_lists\\ans\\rank-result-simulation-dataset-iRANK.mat"
        test_data = pd.read_csv("..\\full_lists\\results\\irank.csv", header=None)
        self.processtest(ans_file_path, test_data, InputType.RANK)

    def test_mean(self):
        from src.rapython.unsupervised import mean
        mean(self.input_file_path, "my_results\\my_mean.csv")
        ans_file_path = "..\\full_lists\\ans\\result-simulation-dataset-Mean.mat"
        test_data = pd.read_csv("my_results\\my_mean.csv", header=None)
        self.processtest(ans_file_path, test_data, InputType.SCORE)

    def test_postndcg(self):
        ans_file_path = "..\\full_lists\\ans\\rank-result-simulation-dataset-postNDCG.mat"
        test_data = pd.read_csv("..\\full_lists\\results\\postndcg.csv", header=None)
        self.processtest(ans_file_path, test_data, InputType.RANK)

    def test_bordacount(self):
        ans_file_path = "..\\full_lists\\ans\\result-simulation-dataset-BordaCount.mat"
        test_data = pd.read_csv("..\\full_lists\\results\\bordacount.csv", header=None)
        self.processtest(ans_file_path, test_data, InputType.SCORE)

    def test_comanz(self):
        ans_file_path = "..\\full_lists\\ans\\result-simulation-dataset-CombANZ.mat"
        test_data = pd.read_csv("..\\full_lists\\results\\combanz.csv", header=None)
        self.processtest(ans_file_path, test_data, InputType.SCORE)

    def test_combmed(self):
        ans_file_path = "..\\full_lists\\ans\\result-simulation-dataset-CombMED.mat"
        test_data = pd.read_csv("..\\full_lists\\results\\combmed.csv", header=None)
        self.processtest(ans_file_path, test_data, InputType.SCORE)

    def test_combmnz(self):
        from src.rapython.unsupervised import combmnz
        combmnz(self.input_file_path, "my_results\\my_combmnz.csv")
        ans_file_path = "..\\full_lists\\ans\\result-simulation-dataset-CombMNZ.mat"
        test_data = pd.read_csv("my_results\\my_combmnz.csv", header=None)
        self.processtest(ans_file_path, test_data, InputType.SCORE)

    def test_comsum(self):
        from src.rapython.unsupervised import combsum
        combsum(self.input_file_path, "my_results\\my_combsum.csv")
        test_data = pd.read_csv("my_results\\my_combsum.csv", header=None)
        ans_file_path = "..\\full_lists\\ans\\result-simulation-dataset-CombSUM.mat"
        self.processtest(ans_file_path, test_data, InputType.SCORE)

    def test_dibra(self):
        ans_file_path = "..\\full_lists\\ans\\result-simulation-dataset-DIBRA.mat"
        test_data = pd.read_csv("..\\full_lists\\results\\dibra.csv", header=None)
        self.processtest(ans_file_path, test_data, InputType.SCORE)

    def test_median(self):
        ans_file_path = "..\\full_lists\\ans\\result-simulation-dataset-Medium.mat"
        test_data = pd.read_csv("..\\full_lists\\results\\median.csv", header=None)
        self.processtest(ans_file_path, test_data, InputType.SCORE)

    def test_rrf(self):
        ans_file_path = "..\\full_lists\\ans\\result-simulation-dataset-rrf.mat"
        test_data = pd.read_csv("..\\full_lists\\results\\rrf.csv", header=None)
        self.processtest(ans_file_path, test_data, InputType.SCORE)


def get_first_key(file_path):
    mat_data = loadmat(file_path)
    keys = list(mat_data.keys())
    filtered_keys = [key for key in keys if not key.startswith('__')]
    return filtered_keys[0]


if __name__ == '__main__':
    unittest.main()
