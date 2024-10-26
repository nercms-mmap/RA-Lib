%{
    dibra - Aggregate rankings using the DIBRA algorithm and output results to a CSV file.
    
    This function imports the Python module `dibra` and calls its `dibra` function to perform
    rank aggregation based on input data provided in a CSV file. The output is also saved in a 
    specified CSV format.

    Reference:
    ----------
    - Akritidis, L., Fevgas, A., Bozanis, P., & Manolopoulos, Y. (2022). An unsupervised distance-based model for weighted rank aggregation with list pruning. Expert Systems with Applications, 202, 117435.

    Authors:
    --------
    Qi Deng

    Date:
    -----
    2024-10-13
%}

function dibra(input_file_path, output_file_path, input_type)
    % dibra - Calls the Python DIBRA algorithm for rank aggregation.
    %
    % This function imports the Python module `dibra` and calls its `dibra` function.
    % It processes the input CSV file containing voting data and outputs the aggregated rankings 
    % to the specified output CSV file.
    %
    % Parameters
    % ----------
    % input_file_path : str
    %     Path to the input CSV file containing voting data. The input should be in CSV format 
    %     with the following columns:
    %     - Query: Does not require consecutive integers starting from 1.
    %     - Voter Name: Allowed to be in string format.
    %     - Item Code: Allowed to be in string format.
    %     - Item Score/Item Rank: Represents the score/rank given by each voter. 
    %
    % output_file_path : str
    %     Path to the output CSV file where the aggregated rankings will be written.
    %
    % input_type : InputType
    %     The type of input data, which determines the naming of the fourth column 
    %     as either 'Item Rank' or 'Item Score'. Defaults to InputType.RANK.
    
    dibramodule = py.importlib.import_module('dibra');
    dibramodule.dibra(input_file_path, output_file_path, input_type);
end
