"""
Competitive Graph Algorithm

This module implements the competitive graph approach for rank aggregation.
This implementation is based on the following reference:

Reference:
-----------
- Xiao, Y., Deng, H. Z., Lu, X., & Wu, J. (2021). Graph-based rank aggregation method for high-dimensional and partial rankings. Journal of the Operational Research Society, 72(1), 227-236.

Authors:
    Qi Deng, Shiwei Feng
Date:
    2023-10-20

Input Format:
-------------
The input to the algorithm should be in CSV file format with the following columns:
- Query: Does not require consecutive integers starting from 1.
- Voter Name: Allowed to be in string format.
- Item Code: Allowed to be in string format.
- Item Rank: Represents the rank given by each voter.

Output Format:
--------------
The final output of the algorithm will be in CSV file format with the following columns:
- Query: The same as the input.
- Item Code: The same as the input.
- Item Rank: The rank information (not the score information).
  - Note: The smaller the Item Rank, the higher the rank.
"""

import numpy as np

from src.rapython.common.constant import InputType
from src.rapython.datatools import *

__all__ = ['cg']


def cgagg(input_list):
    """
    Aggregate scores for items based on rankings provided by voters.
    Parameters
    ----------
    input_list : numpy.ndarray
        A 2D array where each row represents a voter's ranking of items.

    Returns
    -------
    numpy.ndarray
        An array containing the final ranks of the items after aggregation.
    """
    indegree = np.sum(input_list, axis=0) - input_list.shape[0]
    outdegree = input_list.shape[0] * input_list.shape[1] - indegree
    indegree += 1
    outdegree += 1
    score = outdegree / indegree
    rank = np.argsort(np.argsort(score)[::-1]) + 1
    return rank


def cg(input_file_path, output_file_path):
    """
    Process the input CSV file to aggregate rankings and output the results.
    Parameters
    ----------
    input_file_path : str
        Path to the input CSV file containing voting data.
    output_file_path : str
        Path to the output CSV file where the aggregated rankings will be saved.

    Returns
    -------
    None
    """
    df, unique_queries = csv_load(input_file_path, InputType.RANK)
    # Create an empty DataFrame to store results
    result = []

    for query in unique_queries:
        # Filter data for the current Query
        query_data = df[df['Query'] == query]

        item_code_reverse_mapping, _, _, _, input_lists = wtf_map(
            query_data)

        # Call the aggregation function to get ranks
        rank = cgagg(input_lists)

        # Add results to the result list
        for item_code_index, item_rank in enumerate(rank):
            item_code = item_code_reverse_mapping[item_code_index]
            new_row = [query, item_code, item_rank]
            result.append(new_row)

    # Write the results to the output CSV file
    save_as_csv(output_file_path, result)
