%{
    borda_score - Calculate Borda scores based on voters' rankings and output results to a CSV file.

    This script serves as a wrapper for the underlying Python implementation of the Borda score algorithm.
    It loads the necessary Python module and calls the function to perform the rank aggregation using score rules.

    Input Format:
    -------------
    The input to this script should be a CSV file with the following columns:
    - Query: Does not require consecutive integers starting from 1.
    - Voter Name: Allowed to be in string format.
    - Item Code: Allowed to be in string format.
    - Item Rank: Represents the rank given by each voter.

    Output Format:
    --------------
    The final output of the algorithm will be a CSV file with the following columns:
    - Query: The same as the input.
    - Item Code: The same as the input.
    - Item Rank: The rank information (not the score information).
      - Note: The smaller the Item Rank, the higher the rank.

    Reference:
    ----------
    Boehmer, N., Bredereck, R., & Peters, D. (2023, June). Rank aggregation using scoring rules.
    In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 5, pp. 5515-5523).

    Authors:
        Qi Deng
    Date:
        2024-7-25
%}
function borda_score(input_file_path, output_file_path)
    % Calculate the Borda scores for items based on rankings provided by voters and write the results to a CSV file.
    %
    % Parameters
    % ----------
    % input_file_path : str
    %     Path to the input CSV file containing voting data.
    % output_file_path : str
    %     Path to the output CSV file where the results will be written.
    %
    % This function uses the Python module 'borda_score' to perform the calculations.

    bordamodule = py.importlib.import_module('borda_score');
    bordamodule.borda_score(input_file_path, output_file_path);
end
