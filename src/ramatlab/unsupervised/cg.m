%{
    cg - Aggregate rankings using the Competitive Graph method and output results to a CSV file.
    
    This script serves as a wrapper for the underlying Python implementation of the Competitive Graph algorithm.
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
    Xiao, Y., Deng, H. Z., Lu, X., & Wu, J. (2021). Graph-based rank aggregation method for high-dimensional and partial rankings. 
    Journal of the Operational Research Society, 72(1), 227-236.

    Authors: 
        Qi Deng
    Date: 
        2023-10-20
%}
function cg(input_file_path, output_file_path)
    % Process the input CSV file to aggregate rankings and output the results.
    %
    % Parameters
    % ----------
    % input_file_path : str
    %     Path to the input CSV file containing voting data.
    % output_file_path : str
    %     Path to the output CSV file where the aggregated rankings will be saved.
    %
    % Returns
    % -------
    % None
    %
    % This function uses the Python module 'cg' to perform the calculations.

    cgmodule = py.importlib.import_module('cg');
    cgmodule.cg(input_file_path, output_file_path);
end
