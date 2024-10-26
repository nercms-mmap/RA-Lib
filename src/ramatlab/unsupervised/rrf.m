%{
    rrf - Apply the Reciprocal Rank Fusion (RRF) algorithm for rank aggregation and output results to a CSV file.

    This function imports the Python module `rrf` and calls its `rrf` function
    to compute rankings based on input data provided in a CSV file. The output 
    is saved in a specified CSV format.

    Reference:
    ----------
    - Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009, July). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval (pp. 758-759).

    Authors:
    --------
    Qi Deng

    Date:
    -----
    2023-10-19
%}

function rrf(input_file_path, output_file_path)
    % rrf - Calls the Python Reciprocal Rank Fusion (RRF) algorithm for rank aggregation.
    %
    % This function imports the Python module `rrf` and invokes its `rrf` function.
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

    rrfmodule = py.importlib.import_module('rrf');
    rrfmodule.rrf(input_file_path, output_file_path);
end
