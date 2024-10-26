%{
    hpa - Aggregate rankings using the HPA algorithm and output results to a CSV file.
    
    This function imports the Python module `hpa` and calls its `hpa` function to perform
    rank aggregation based on input data provided in a CSV file. The output is also saved in a 
    specified CSV format.

    Reference:
    ----------
    - Fujita, S., Kobayashi, H., & Okumura, M. (2020). Unsupervised ensemble of ranking models for news comments using pseudo answers. In Advances in Information Retrieval: 42nd European Conference on IR Research, ECIR 2020, Lisbon, Portugal, April 14–17, 2020, Proceedings, Part II 42 (pp. 133-140). Springer International Publishing.

    Authors:
    --------
    Qi Deng

    Date:
    -----
    2024-10-13
%}

function hpa(input_file_path, output_file_path, input_type)
    % hpa - Calls the Python HPA algorithm for rank aggregation.
    %
    % This function imports the Python module `hpa` and calls its `hpa` function.
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
    
    hpamodule = py.importlib.import_module('hpa');
    hpamodule.hpa(input_file_path, output_file_path, input_type);
end
