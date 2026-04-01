"""
ListNet Implementation for Rank Aggregation.
"""

import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.rapython.common.constant import InputType
from src.rapython.datatools import csv_load
from src.rapython.evaluation.evaluation import Evaluation

__all__ = ['ListNet']


class ListNet:
    """
    ListNet algorithm for learning to rank using listwise comparisons.
    
    This class implements the ListNet algorithm which learns a ranking function
    by minimizing the cross-entropy loss between the predicted and true score
    distributions (softmax over scores).
    
    Parameters
    ----------
    hidden_units : list of int, optional
        List of hidden layer sizes in the neural network. Default is [10].
    learning_rate : float, optional
        Learning rate for Adam optimizer. Default is 0.001.
    epochs_per_query : int, optional
        Number of training epochs for each query. Default is 10.
    eps : float, optional
        Small constant for numerical stability. Default is 1e-10.
    """
    
    def __init__(self, 
                 hidden_units=[10], 
                 learning_rate=0.001, 
                 epochs_per_query=10,
                 eps=1e-10):
        """
        Initialize ListNet with hyperparameters.
        """
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs_per_query = epochs_per_query
        self.eps = eps
        
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
        layers = []
        prev_dim = input_dim
        for units in self.hidden_units:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.Tanh())
            prev_dim = units
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)
    
    @staticmethod
    def _listnet_loss(y_pred, y_true, eps=1e-10):
        """
        Calculate ListNet loss (listwise cross-entropy).
        
        The loss computes the cross-entropy between the softmax distributions
        of predicted scores and true relevance scores.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted scores of shape (batch_size, num_items).
        y_true : torch.Tensor
            True relevance scores of shape (batch_size, num_items).
        eps : float, optional
            Small constant for numerical stability.
            
        Returns
        -------
        torch.Tensor
            Mean loss value.
        """
        y_pred = y_pred.clone()
        y_true = y_true.clone()
        
        preds_smax = F.softmax(y_pred, dim=1)
        true_smax = F.softmax(y_true, dim=1)
        
        preds_smax = torch.clamp(preds_smax, min=eps, max=1.0 - eps)
        preds_log = torch.log(preds_smax)
        
        loss = -torch.sum(true_smax * preds_log, dim=1)
        return torch.mean(loss)
    
    @staticmethod
    def _handle_partial_list(rank_matrix):
        """
        Handle partial lists by assigning the maximum rank + 1 to unrated items.
        
        Parameters
        ----------
        rank_matrix : numpy.ndarray
            Rank matrix with NaN for missing rankings.
            
        Returns
        -------
        numpy.ndarray
            Rank matrix with all NaN values replaced.
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
    
    @staticmethod
    def _preprocess_relevances(rel_vector):
        """
        Preprocess relevance labels for listwise loss.
        
        Converts relevance labels to positive values for softmax calculation.
        
        Parameters
        ----------
        rel_vector : numpy.ndarray
            Relevance labels (can be negative or zero).
            
        Returns
        -------
        numpy.ndarray
            Transformed relevance labels (all positive).
        """
        # Transform: -1 → 1, 0 → 2, 1 → 3
        transformed = rel_vector + 2
        return transformed.astype(np.float32)
    
    def _build_rank_matrix(self, base_data, item_mapping, item_num):
        """
        Build the raw rank matrix from base data.
        
        Parameters
        ----------
        base_data : pandas.DataFrame
            Base data for a single query.
        item_mapping : dict
            Mapping from item code to item index.
        item_num : int
            Number of unique items.
            
        Returns
        -------
        numpy.ndarray
            Rank matrix of shape (item_num, voter_num), with NaN for missing rankings.
        """
        rank_matrix = np.full((item_num, self.voter_num), np.nan)
        
        for _, row in base_data.iterrows():
            voter_name = row['Voter Name']
            item_code = row['Item Code']
            item_rank = row['Item Rank']
            
            voter_idx = self.voter_name_mapping[voter_name]
            item_idx = item_mapping[item_code]
            rank_matrix[item_idx, voter_idx] = item_rank
        
        return rank_matrix
    
    def train(self, train_file_path, train_rel_path, input_type):
        """
        Train the ListNet model using listwise learning to rank.
        
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
            rel_vector = self._preprocess_relevances(rel_vector)
            features = torch.FloatTensor(feature_matrix).unsqueeze(0)
            relevances = torch.FloatTensor(rel_vector).unsqueeze(0)
            
            model = self._build_model(self.voter_num)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            model.train()
            for epoch in range(self.epochs_per_query):
                optimizer.zero_grad()
                scores = model(features.squeeze(0))
                scores = scores.reshape(1, -1)
                
                loss = self._listnet_loss(scores, relevances, self.eps)
                
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
