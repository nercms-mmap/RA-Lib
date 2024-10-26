%{
    postndcg - Apply the PostNDCG algorithm for rank aggregation and output results to a CSV file.

    This function imports the Python module `postndcg` and calls its `postndcg` function
    to compute rankings based on input data provided in a CSV file. The output 
    is saved in a specified CSV format.

    Reference:
    ----------
    - Fujita, S., Kobayashi, H., & Okumura, M. (2020). Unsupervised ensemble of ranking models for news comments using pseudo answers. In Advances in Information Retrieval: 42nd European Conference on IR Research, ECIR 2020, Lisbon, Portugal, April 14–17, 2020, Proceedings, Part II 42 (pp. 133-140). Springer International Publishing.

    Authors:
    --------
    Qi Deng

    Date:
    -----
    2024-9-18
%}

function postndcg(input_file_path, output_file_path, input_type)
    % postndcg - Calls the Python PostNDCG algorithm for rank aggregation.
    %
    % This function imports the Python module `postndcg` and invokes its `postndcg` function.
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
    %     - Item Score/Item Rank: Represents the score/rank given by each voter. 
    %
    % output_file_path : str
    %     Path to the output CSV file where the aggregated rankings will be written.
    %
    % input_type : InputType
    %     The type of input data. It determines the naming of the fourth column, 
    %     which will either be 'Item Rank' or 'Item Score' based on this value.

    postndcgmodule = py.importlib.import_module('postndcg');
    postndcgmodule.postndcg(input_file_path, output_file_path, input_type);
end
