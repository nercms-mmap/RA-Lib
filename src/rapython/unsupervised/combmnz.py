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
"""

import numpy as np

from src.rapython.datatools import *
from src.rapython.common.constant import InputType

__all__ = ['combmnz']


def combmnz_agg(input_list):
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
    combmnz_score = np.zeros(item_num)
    for item in range(item_num):
        rank = input_list[:, item]
        sr = -0.99 * rank / (threshold - 1) + 1 + 0.99 / (threshold - 1)
        sr[sr < 0.01] = 0
        srd = np.sum(sr)
        s = np.sum(sr > 0)
        combmnz_score[item] = srd * s

    rank_result = np.argsort(np.argsort(combmnz_score)[::-1]) + 1
    return rank_result


def combmnz(input_file_path, output_file_path):
    """
    Process the input CSV file to aggregate rankings and write results to an output CSV file.
    Parameters
    ----------
    input_file_path : str
        Path to the input CSV file.
    output_file_path : str
        Path to the output CSV file where aggregated rankings will be written.
    """
    df, unique_queries = csv_load(input_file_path, InputType.RANK)
    # Create an empty DataFrame to store results
    result = []

    for query in unique_queries:
        # Filter data for the current Query
        query_data = df[df['Query'] == query]

        item_code_reverse_mapping, _, _, _, input_lists = wtf_map(
            query_data)

        # Call function to get aggregated ranks
        rank = combmnz_agg(input_lists)

        # Add results to the result list
        for item_code_index, item_rank in enumerate(rank):
            item_code = item_code_reverse_mapping[item_code_index]
            new_row = [query, item_code, item_rank]
            result.append(new_row)

    # Write results to the output CSV file
    save_as_csv(output_file_path, result)
