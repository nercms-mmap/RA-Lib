%{
    mean - Apply the Mean algorithm for rank aggregation and output results to a CSV file.
    
    This function imports the Python module `mean` and calls its `mean` function
    to compute the mean rankings based on input data provided in a CSV file. The output 
    is saved in a specified CSV format.

    Reference:
    ----------
    - Kaur, M., Kaur, P., & Singh, M. (2015, September). Rank aggregation using multi objective genetic algorithm. In 2015 1st International Conference on Next Generation Computing Technologies (NGCT) (pp. 836-840). IEEE.

    Authors:
    --------
    Qi Deng

    Date:
    -----
    2023-10-18
%}

function mean(input_file_path, output_file_path)
    % mean - Calls the Python Mean method for rank aggregation.
    %
    % This function imports the Python module `mean` and invokes its `mean`.
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

    meanmodule = py.importlib.import_module('mean');
    meanmodule.mean(input_file_path, output_file_path);
end
