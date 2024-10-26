%{
    irank - Aggregate rankings using the iRANk algorithm and output results to a CSV file.
    
    This function imports the Python module `irank` and calls its `irank` function to perform
    rank aggregation based on input data provided in a CSV file. The output is also saved in a 
    specified CSV format.

    Reference:
    ----------
    - Wei, F., Li, W., & Liu, S. (2010). iRANK: A rank‐learn‐combine framework for unsupervised ensemble ranking. Journal of the American Society for Information Science and Technology, 61(6), 1232-1243.

    Authors:
    --------
    Qi Deng

    Date:
    -----
    2024-10-11
%}

function irank(input_file_path, output_file_path, input_type)
    % irank - Calls the Python iRANk algorithm for rank aggregation.
    %
    % This function imports the Python module `irank` and calls its `irank` function.
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
    
    irankmodule = py.importlib.import_module('irank');
    irankmodule.irank(input_file_path, output_file_path, input_type);
end
