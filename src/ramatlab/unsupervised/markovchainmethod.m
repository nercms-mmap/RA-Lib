%{
    markovchainmethod - Apply the Markov Chain algorithm for rank aggregation and output results to a CSV file.
    
    This function imports the Python module `markovchain` and calls its `markovchainmethod` function
    to perform rank aggregation based on input data provided in a CSV file. The output is saved in a 
    specified CSV format.

    Reference:
    ----------
    - Dwork, C., Kumar, R., Naor, M., & Sivakumar, D. (2001, April). Rank aggregation methods for the web. In Proceedings of the 10th international conference on World Wide Web (pp. 613-622).

    Authors:
    --------
    Qi Deng

    Date:
    -----
    2023-09-26
%}

function markovchainmethod(input_file_path, output_file_path, mc_type, max_iteration)
    % markovchainmethod - Calls the Python Markov Chain method for rank aggregation.
    %
    % This function imports the Python module `markovchain` and invokes its `markovchainmethod`.
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
    %
    % mc_type : McType
    %     The type of Markov Chain method to use (e.g., MC1, MC2, etc.).
    %
    % max_iteration : int, optional
    %     The maximum number of iterations for the power method. Defaults to 50 if not provided.

    if nargin < 4
        max_iteration = 50;
    end
    mcmodule = py.importlib.import_module('markovchain');
    mcmodule.markovchainmethod(input_file_path, output_file_path, mc_type, py.int(max_iteration));
end
