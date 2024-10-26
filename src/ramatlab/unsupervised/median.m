%{
    median - Apply the Median algorithm for rank aggregation and output results to a CSV file.
    
    This function imports the Python module `median` and calls its `median` function
    to compute the median rankings based on input data provided in a CSV file. The output 
    is saved in a specified CSV format.

    Reference:
    ----------
    - Fagin, R., Kumar, R., & Sivakumar, D. (2003, June). Efficient similarity search and classification via rank aggregation. In Proceedings of the 2003 ACM SIGMOD international conference on Management of data (pp. 301-312).

    Authors:
    --------
    Qi Deng

    Date:
    -----
    2023-10-18
%}

function median(input_file_path, output_file_path)
    % median - Calls the Python Median method for rank aggregation.
    %
    % This function imports the Python module `median` and invokes its `median` function.
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

    medianmodule = py.importlib.import_module('median');
    medianmodule.median(input_file_path, output_file_path);
end
