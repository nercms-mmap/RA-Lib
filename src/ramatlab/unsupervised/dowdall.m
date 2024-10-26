%{
    dowdall - Aggregate rankings using the Dowdall algorithm and output results to a CSV file.
    
    This function imports the Python module `dowdall` and calls its `dowdall` function to perform
    rank aggregation based on input data provided in a CSV file. The output is also saved in a 
    specified CSV format.

    Reference:
    ----------
    - Reilly, B. (2002). Social choice in the south seas: Electoral innovation and the borda count in the pacific island countries. International Political Science Review, 23(4), 355-372.

    Authors:
    --------
    Qi Deng

    Date:
    -----
    2023-10-19
%}

function dowdall(input_file_path, output_file_path)
    % dowdall - Calls the Python Dowdall algorithm for rank aggregation.
    %
    % This function imports the Python module `dowdall` and calls its `dowdall` function.
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
    %     - Item Rank: Represents the rank given by each voter. 
    %
    % output_file_path : str
    %     Path to the output CSV file where the aggregated rankings will be written.
    
    dowdallmodule = py.importlib.import_module('dowdall');
    dowdallmodule.dowdall(input_file_path, output_file_path);
end
