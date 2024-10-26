%{
    bordacount - Aggregate ranks using the Borda Count method and output results to a CSV file.

    This script serves as a wrapper for the underlying Python implementation of the Borda Count algorithm.
    It loads the necessary Python module and calls the function to perform rank aggregation based on voting data.

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
    Borda, J. D. (1781). M'emoire sur les' elections au scrutin.
    Histoire de l'Acad'emie Royale des Sciences.

    Authors:
        Qi Deng
    Date:
        2024-7-25
%}
function bordacount(input_file_path, output_file_path)
    % Process a CSV file containing voting data and output the aggregated ranks using Borda count.
    %
    % Parameters
    % ----------
    % input_file_path : str
    %     Path to the input CSV file containing voting data.
    % output_file_path : str
    %     Path to the output CSV file where the aggregated ranks will be written.
    %
    % This function uses the Python module 'bordacount' to perform the calculations.

    bordamodule = py.importlib.import_module('bordacount');
    bordamodule.bordacount(input_file_path, output_file_path);
end
