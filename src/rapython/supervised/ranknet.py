"""
RankNet Implementation for Rank Aggregation.
"""

import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.rapython.common.constant import InputType
from src.rapython.datatools import csv_load
from src.rapython.evaluation.evaluation import Evaluation

__all__ = ['RankNet']


class RankNet:
    """
    RankNet algorithm for learning to rank using pairwise comparisons.
    
    This class implements the RankNet algorithm which learns a ranking function
    by minimizing a pairwise cross-entropy loss.
    
    Parameters
    ----------
    hidden_units : list of int, optional
        List of hidden layer sizes in the neural network. Default is [10].
    learning_rate : float, optional
        Learning rate for Adam optimizer. Default is 0.001.
    epochs_per_query : int, optional
        Number of training epochs for each query. Default is 5.
    batch_size : int, optional
        Batch size for training. Default is 32.
    """
    
    def __init__(self, 
                 hidden_units=[10], 
                 learning_rate=0.001, 
                 epochs_per_query=5,
                 batch_size=32):
        """
        Initialize RankNet with hyperparameters.
        """
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs_per_query = epochs_per_query
        self.batch_size = batch_size
        
        # Storage for trained models
        self.query_models = {}
        self.average_model_state = None
        self.average_model = None
        
        # Mappings
        self.voter_name_mapping = None
        self.voter_name_reverse_mapping = None
        self.voter_num = None
        self.query_mapping = None
    
    def _build_model(self, input_dim):
        """
        Build the neural network model.
        
        Parameters
        ----------
        input_dim : int
            Dimension of input features (number of voters).
            
        Returns
        -------
        torch.nn.Sequential
            The neural network model.
        """
        layers = []
        prev_dim = input_dim
        for units in self.hidden_units:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.Tanh())
            prev_dim = units
        layers.append(nn.Linear(prev_dim, 1, bias=False))
        return nn.Sequential(*layers)
    
    @staticmethod
    def _pairwise_loss(score_i, score_j, label):
        """
        Calculate pairwise cross-entropy loss for RankNet.
        """
        diff = score_i - score_j
        pred_prob = torch.sigmoid(diff)
        loss = -(label * torch.log(pred_prob + 1e-8) + 
                 (1 - label) * torch.log(1 - pred_prob + 1e-8))
        return torch.mean(loss)
    
    @staticmethod
    def _handle_partial_list(rank_matrix):
        """
        Handle partial lists by assigning the maximum rank + 1 to unrated items.
        """
        item_num = rank_matrix.shape[0]
        for k in range(rank_matrix.shape[1]):
            if np.isnan(rank_matrix[:, k]).all():
                rank_matrix[:, k] = item_num
            else:
                max_rank = np.nanmax(rank_matrix[:, k])
                rank_matrix[:, k] = np.nan_to_num(rank_matrix[:, k], nan=max_rank + 1)
        return rank_matrix
    
    def _ranks_to_features(self, rank_matrix):
        features = 1.0 / (rank_matrix + 1e-6)
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        return features.astype(np.float32)
    
    def _build_rank_matrix(self, base_data, item_mapping, item_num):
        rank_matrix = np.full((item_num, self.voter_num), np.nan)
        
        for _, row in base_data.iterrows():
            voter_name = row['Voter Name']
            item_code = row['Item Code']
            item_rank = row['Item Rank']
            
            voter_idx = self.voter_name_mapping[voter_name]
            item_idx = item_mapping[item_code]
            rank_matrix[item_idx, voter_idx] = item_rank
        
        return rank_matrix
    
    def _generate_pairs(self, feature_matrix, rel_vector):
        item_num = feature_matrix.shape[0]
        pairs = []
        labels = []
        
        for i in range(item_num):
            for j in range(i + 1, item_num):
                if rel_vector[i] > rel_vector[j]:
                    pairs.append((feature_matrix[i], feature_matrix[j]))
                    labels.append(1.0)
                elif rel_vector[i] < rel_vector[j]:
                    pairs.append((feature_matrix[j], feature_matrix[i]))
                    labels.append(1.0)
        
        if len(pairs) == 0:
            return None, None, None
        
        left_features = np.array([p[0] for p in pairs])
        right_features = np.array([p[1] for p in pairs])
        labels_array = np.array(labels)
        
        return left_features, right_features, labels_array
    
    def train(self, train_file_path, train_rel_path, input_type):
        """
        Train the RankNet model using pairwise learning to rank.
        
        Parameters
        ----------
        train_file_path : str
            File path to the training base data.
        train_rel_path : str
            File path to the relevance data.
        input_type : InputType
            Specifies the format of the input data. Must be InputType.RANK.
            
        Returns
        -------
        None
        """
        input_type = InputType.check_input_type(input_type)
        
        train_base_data, train_rel_data, unique_queries = csv_load(
            train_file_path, train_rel_path, input_type
        )
        
        train_base_data.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']
        train_rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']
        unique_voter_names = train_base_data['Voter Name'].unique()
        self.voter_num = len(unique_voter_names)
        self.voter_name_mapping = {name: i for i, name in enumerate(unique_voter_names)}
        self.voter_name_reverse_mapping = {i: name for i, name in enumerate(unique_voter_names)}
        self.query_mapping = {name: i for i, name in enumerate(unique_queries)}
        
        successful_queries = 0
        for query in tqdm(unique_queries, desc="Training queries"):
            base_data = train_base_data[train_base_data['Query'] == query]
            rel_data = train_rel_data[train_rel_data['Query'] == query]
            
            unique_items = base_data['Item Code'].unique()
            item_num = len(unique_items)
            item_mapping = {name: i for i, name in enumerate(unique_items)}
            
            rank_matrix = self._build_rank_matrix(base_data, item_mapping, item_num)
            rank_matrix = self._handle_partial_list(rank_matrix)
            rel_vector = np.zeros(item_num)
            
            for _, row in rel_data.iterrows():
                item_code = row['Item Code']
                item_relevance = row['Relevance']
                if item_code in item_mapping:
                    item_idx = item_mapping[item_code]
                    rel_vector[item_idx] = item_relevance
            
            feature_matrix = self._ranks_to_features(rank_matrix)
            left_features, right_features, labels = self._generate_pairs(feature_matrix, rel_vector)
            
            if left_features is None:
                continue
            
            left_tensor = torch.FloatTensor(left_features)
            right_tensor = torch.FloatTensor(right_features)
            label_tensor = torch.FloatTensor(labels).unsqueeze(1)
            
            dataset = TensorDataset(left_tensor, right_tensor, label_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            model = self._build_model(self.voter_num)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            model.train()
            for epoch in range(self.epochs_per_query):
                for left_batch, right_batch, label_batch in dataloader:
                    optimizer.zero_grad()
                    score_left = model(left_batch)
                    score_right = model(right_batch)
                    loss = self._pairwise_loss(score_left, score_right, label_batch)
                    loss.backward()
                    optimizer.step()
            
            query_id = self.query_mapping[query]
            self.query_models[query_id] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            successful_queries += 1
        
        if self.query_models:
            first_model = next(iter(self.query_models.values()))
            self.average_model_state = {}
            for key in first_model:
                self.average_model_state[key] = torch.zeros_like(first_model[key])
            
            for model_params in self.query_models.values():
                for key in model_params:
                    self.average_model_state[key] += model_params[key]
            
            for key in self.average_model_state:
                self.average_model_state[key] /= len(self.query_models)
            
            print(f"Successfully trained {successful_queries} out of {len(unique_queries)} queries")
    
    def test(self, test_file_path, test_output_path, using_average_w=True):
        """
        Test the model and generate rankings for test data.
        
        Parameters
        ----------
        test_file_path : str
            File path to the test data.
        test_output_path : str
            File path where the output CSV will be written.
        using_average_w : bool, optional
            Whether to use the average model across queries. Default is True.
            
        Returns
        -------
        None
        """
        test_data, unique_test_queries = csv_load(test_file_path, InputType.RANK)
        test_data.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']
        
        with open(test_output_path, 'w', newline='') as file:
            writer = csv.writer(file)
            
            for query in tqdm(unique_test_queries, desc="Testing"):
                query_data = test_data[test_data['Query'] == query]
                
                unique_items = query_data['Item Code'].unique()
                item_num = len(unique_items)
                item_mapping = {name: i for i, name in enumerate(unique_items)}
                item_code_reverse = {v: k for k, v in item_mapping.items()}
                
                rank_matrix = self._build_rank_matrix(query_data, item_mapping, item_num)
                rank_matrix = self._handle_partial_list(rank_matrix)
                feature_matrix = self._ranks_to_features(rank_matrix)
                
                if using_average_w or query not in self.query_mapping:
                    if self.average_model is None and self.average_model_state is not None:
                        self.average_model = self._build_model(self.voter_num)
                        self.average_model.load_state_dict(self.average_model_state)
                    model = self.average_model
                else:
                    query_id = self.query_mapping[query]
                    model = self._build_model(self.voter_num)
                    model.load_state_dict(self.query_models[query_id])
                
                model.eval()
                with torch.no_grad():
                    features = torch.FloatTensor(feature_matrix)
                    scores = model(features).squeeze().numpy()
                
                ranked_indices = np.argsort(scores)[::-1]
                for rank, item_idx in enumerate(ranked_indices, start=1):
                    writer.writerow([query, item_code_reverse[item_idx], rank])

                ranked_indices = np.argsort(scores)[::-1]
                item_code_reverse_mapping = {v: k for k, v in item_mapping.items()}
                
                for rank, item_idx in enumerate(ranked_indices, start=1):
                    item_code = item_code_reverse_mapping[item_idx]
                    writer.writerow([query, item_code, rank])
