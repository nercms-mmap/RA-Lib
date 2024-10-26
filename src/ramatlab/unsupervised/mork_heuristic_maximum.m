%{
    mork_heuristic_maximum - Apply the Mork-H algorithm for rank aggregation and output results to a CSV file.
    
    This function imports the Python module `mork_heuristic_maximum` and calls its `mork_heuristic` function
    to compute rankings based on input data provided in a CSV file. The output 
    is saved in a specified CSV format.

    Reference:
    ----------
    - Azzini, I., & Munda, G. (2020). A new approach for identifying the Kemeny median ranking. European Journal of Operational Research, 281(2), 388-401.

    Authors:
    --------
    Qi Deng

    Date:
    -----
    2024-9-18
%}

function mork_heuristic_maximum(input_file_path, output_file_path)
    % mork_heuristic_maximum - Calls the Python Mork-H algorithm for rank aggregation.
    %
    % This function imports the Python module `mork_heuristic_maximum` and invokes its `mork_heuristic` function.
    % It processes the input CSV file containing voting data and outputs the 
    % aggregated rankings to the specified output CSV file.
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

    morkmodule = py.importlib.import_module('mork_heuristic_maximum');
    morkmodule.mork_heuristic(input_file_path, output_file_path);
end
