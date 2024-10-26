%{
    er - Aggregate rankings using the ER algorithm and output results to a CSV file.
    
    This function imports the Python module `er` and calls its `er` function to perform
    rank aggregation based on input data provided in a CSV file. The output is also saved in a 
    specified CSV format.

    Reference:
    ----------
    - Mohammadi, M., & Rezaei, J. (2020). Ensemble ranking: Aggregation of rankings produced by different multi-criteria decision-making methods. Omega, 96, 102254.

    Authors:
    --------
    Qi Deng

    Date:
    -----
    2024-10-13
%}

function er(input_file_path, output_file_path, input_type)
    % er - Calls the Python ER algorithm for rank aggregation.
    %
    % This function imports the Python module `er` and calls its `er` function.
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
    %     The type of input data, which determines the interpretation of the fourth column 
    %     as either 'Item Rank' or 'Item Score'.
    
    ermodule = py.importlib.import_module('er');
    ermodule.er(input_file_path, output_file_path, input_type);
end
