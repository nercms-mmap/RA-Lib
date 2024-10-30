"""
Comb* Family Algorithm

This module implements the Comb* family of algorithms for rank aggregation.
This implementation is based on the following reference:

Reference:
-----------
- Fox, E., & Shaw, J. (1994). Combination of multiple searches. NIST special publication SP, 243-243.

Authors:
    Qi Deng, Shiwei Feng
Date:
    2023-10-18

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

from src.rapython.datatools import *
from src.rapython.common.constant import InputType

__all__ = ['combanz']


def combanz_agg(input_list):
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
    item_num = input_list.shape[1]
    threshold = 20
    combanz_score = np.zeros(item_num)
    for item in range(item_num):
        rank = input_list[:, item]
        sr = -0.99 * rank / (threshold - 1) + 1 + 0.99 / (threshold - 1)
        sr[sr < 0.01] = 0
        srd = np.sum(sr)
        s = np.sum(sr > 0)
        combanz_score[item] = srd / s if s != 0 else 0

    rank_result = np.argsort(np.argsort(combanz_score)[::-1]) + 1
    return rank_result


def combanz(input_file_path, output_file_path):
    """
    Process the input CSV file to aggregate rankings and write the results to an output CSV file.
    Parameters
    ----------
    input_file_path : str
        Path to the input CSV file.
    output_file_path : str
        Path to the output CSV file.
    """
    df, unique_queries = csv_load(input_file_path, InputType.RANK)
    # Create an empty DataFrame to store results
    result = []

    for query in unique_queries:
        # Filter data for the current Query
        query_data = df[df['Query'] == query]

        item_code_reverse_mapping, _, _, _, input_lists = wtf_map(
            query_data)

        # Call function to get ranking information
        rank = combanz_agg(input_lists)

        # Add results to the result list
        for item_code_index, item_rank in enumerate(rank):
            item_code = item_code_reverse_mapping[item_code_index]
            new_row = [query, item_code, item_rank]
            result.append(new_row)

    # Write results to the output CSV file
    save_as_csv(output_file_path, result)
