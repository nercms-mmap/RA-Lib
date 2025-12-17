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
    A class to implement the ListNet algorithm for learning-to-rank tasks.
    
    Hyperparameters:
    ---------------
    hidden_units : list, optional
        The structure of hidden layers in the neural network.
        Defaults to [10] (consistent with RankNet).

    learning_rate : float, optional
        Learning rate for optimization.

    epochs_per_query : int, optional
        Number of training epochs for each query.

    eps : float, optional
        Epsilon value for numerical stability in loss computation.
        Defaults to 1e-10.
    """

    def __init__(self, hidden_units=None, learning_rate=0.001, epochs_per_query=10, eps=1e-10):
        """
        Initializes the ListNet instance with specified parameters.

        Parameters:
        -----------
        hidden_units : list, optional
            List specifying the number of units in each hidden layer.
            Defaults to [10].

        learning_rate : float, optional
            Learning rate for the Adam optimizer. Defaults to 0.001.

        epochs_per_query : int, optional
            Number of training epochs for each query. Defaults to 10.

        eps : float, optional
            Small value to prevent numerical instability. Defaults to 1e-10.
        """
        if hidden_units is None:
            hidden_units = [10]
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs_per_query = epochs_per_query
        self.eps = eps
        self.model = None
        self.optimizer = None

        # Define attributes consistent with other models in the project
        self.voter_name_mapping = None
        self.voter_num = None
        self.query_mapping = None

    @staticmethod
    def _partial_to_full(rank_base_data_matrix):
        """
        Converts a rank base data matrix with potential missing values
        into a full list format.

        Parameters:
        -----------
        rank_base_data_matrix : numpy.ndarray
            A 2D numpy array of shape (voter_num, item_num) containing
            rankings, where missing values are represented by NaN.

        Returns:
        --------
        numpy.ndarray
            A modified rank base data matrix in full list format.
        """
        num_voters = rank_base_data_matrix.shape[0]

        for k in range(num_voters):
            if np.isnan(rank_base_data_matrix[k]).all():
                rank_base_data_matrix[k] = np.nan_to_num(rank_base_data_matrix[k], 
                                                         nan=rank_base_data_matrix.shape[1])
            else:
                max_rank = np.nanmax(rank_base_data_matrix[k])
                rank_base_data_matrix[k] = np.nan_to_num(rank_base_data_matrix[k], 
                                                         nan=max_rank + 1)

        return rank_base_data_matrix

    def _convert_to_matrix(self, base_data, rel_data=None):
        """
        Converts the provided base data into matrices suitable for ListNet.

        Parameters:
        -----------
        base_data : pandas.DataFrame
            A DataFrame containing the base data.

        rel_data : pandas.DataFrame, optional
            A DataFrame containing relevance data.

        Returns:
        --------
        tuple
            - score_base_data_matrix : numpy.ndarray
              A 2D numpy array storing Borda scores.
            - rel_data_matrix : numpy.ndarray or None
              A 1D numpy array storing relevance scores if rel_data is provided.
            - item_mapping : dict
              A mapping of item codes to indices.
        """
        unique_items = base_data['Item Code'].unique()
        item_num = len(unique_items)
        item_mapping = {name: i for i, name in enumerate(unique_items)}
        rank_base_data_matrix = np.full((self.voter_num, item_num), np.nan)

        for _, row in base_data.iterrows():
            voter_name = row['Voter Name']
            item_code = row['Item Code']
            item_rank = row['Item Rank']

            voter_index = self.voter_name_mapping[voter_name]
            item_index = item_mapping[item_code]
            rank_base_data_matrix[voter_index, item_index] = item_rank

        rank_base_data_matrix = self._partial_to_full(rank_base_data_matrix)
        
        # Convert ranks to Borda scores
        score_base_data_matrix = np.empty((self.voter_num, item_num))
        for k in range(self.voter_num):
            for i in range(item_num):
                score_base_data_matrix[k, i] = item_num - rank_base_data_matrix[k, i]

        if rel_data is None:
            return score_base_data_matrix, item_mapping
        else:
            rel_data_matrix = np.zeros(item_num)
            for _, row in rel_data.iterrows():
                item_code = row['Item Code']
                item_relevance = row['Relevance']
                item_index = item_mapping[item_code]
                rel_data_matrix[item_index] = item_relevance

            return score_base_data_matrix, rel_data_matrix, item_mapping

    def _build_model(self, input_dim):
        """
        Constructs the ListNet scoring network.

        Parameters:
        -----------
        input_dim : int
            Dimension of input features.

        Returns:
        --------
        torch.nn.Sequential
            The constructed neural network model.
        """
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
        Computes the ListNet loss.

        Parameters:
        -----------
        y_pred : torch.Tensor
            Predicted scores from the model, shape [batch_size, slate_length].

        y_true : torch.Tensor
            True relevance scores, shape [batch_size, slate_length].

        eps : float, optional
            Small value for numerical stability.

        Returns:
        --------
        torch.Tensor
            Computed loss value.
        """
        preds_smax = F.softmax(y_pred, dim=1)
        true_smax = F.softmax(y_true, dim=1)

        preds_smax = preds_smax + eps
        preds_log = torch.log(preds_smax)

        loss = -torch.sum(true_smax * preds_log, dim=1)
        return torch.mean(loss)

    def train(self, train_file_path, train_rel_path):
        """
        Trains the ListNet model using the provided training data.

        Parameters:
        -----------
        train_file_path : str
            The file path to the training base data.

        train_rel_path : str
            The file path to the relevance data.

        Returns:
        --------
        None
        """
        # Load data using project's standard method
        train_base_data, train_rel_data, unique_queries = csv_load(
            train_file_path, train_rel_path, InputType.RANK
        )
        
        train_base_data.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']
        train_rel_data.columns = ['Query', '0', 'Item Code', 'Relevance']

        unique_voter_names = train_base_data['Voter Name'].unique()
        self.voter_num = len(unique_voter_names)
        self.voter_name_mapping = {name: i for i, name in enumerate(unique_voter_names)}
        self.query_mapping = {name: i for i, name in enumerate(unique_queries)}

        # Build model
        input_dim = self.voter_num
        self.model = self._build_model(input_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Train for each query
        for query in tqdm(unique_queries):
            base_data = train_base_data[train_base_data['Query'] == query]
            rel_data = train_rel_data[train_rel_data['Query'] == query]

            # Convert to matrix format
            score_matrix, rel_vector, item_mapping = self._convert_to_matrix(base_data, rel_data)
            
            # Scale relevance scores
            rel_vector = np.nan_to_num(rel_vector, nan=0.0)
            rel_vector = rel_vector * 5.0

            # Prepare tensors for ListNet
            feature_matrix = score_matrix.T
            features = torch.FloatTensor(feature_matrix).unsqueeze(0)
            relevances = torch.FloatTensor(rel_vector).unsqueeze(0)

            # Training loop for this query
            self.model.train()
            for _ in range(self.epochs_per_query):
                self.optimizer.zero_grad()
                
                scores = self.model(features.squeeze(0))
                scores = scores.reshape(1, -1)

                loss = self._listnet_loss(scores, relevances, self.eps)
                loss.backward()
                self.optimizer.step()

    def test(self, test_file_path, test_output_path, using_average_w=True):
        """
        Tests the ListNet model on the provided test data.

        Parameters:
        -----------
        test_file_path : str
            The file path to the test data.

        test_output_path : str
            The file path where the output CSV file will be saved.

        using_average_w : bool, optional
            Kept for interface consistency (not used in ListNet).
            Defaults to True.

        Returns:
        --------
        None
        """
        test_data, unique_test_queries = csv_load(test_file_path, InputType.RANK)
        test_data.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']

        self.model.eval()
        with open(test_output_path, 'w', newline='') as file:
            writer = csv.writer(file)
            
            for query in tqdm(unique_test_queries):
                query_data = test_data[test_data['Query'] == query]
                
                # Convert to matrix format
                score_matrix, item_mapping = self._convert_to_matrix(query_data)
                feature_matrix = score_matrix.T

                # Predict scores
                with torch.no_grad():
                    features = torch.FloatTensor(feature_matrix)
                    scores = self.model(features).squeeze().numpy()

                # Generate rankings
                ranked_indices = np.argsort(scores)[::-1]
                item_code_reverse_mapping = {v: k for k, v in item_mapping.items()}
                
                for rank, item_idx in enumerate(ranked_indices, start=1):
                    item_code = item_code_reverse_mapping[item_idx]
                    writer.writerow([query, item_code, rank])
