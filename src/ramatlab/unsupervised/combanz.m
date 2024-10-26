%{
    combanz - Aggregate rankings using the Comb* family of algorithms and output results to a CSV file.
    
    This function imports the Python module `combanz` and calls its `combanz` function to perform
    rank aggregation based on input data provided in a CSV file. The output is also saved in a 
    specified CSV format.

    Reference:
    ----------
    - Fox, E., & Shaw, J. (1994). Combination of multiple searches. NIST special publication SP, 243-243.

    Authors:
    --------
    Qi Deng

    Date:
    -----
    2023-10-18

    Input Format:
    -------------
    The input to the algorithm should be in CSV file format with the following columns:
    - Query: Does not require consecutive integers starting from 1.
    - Voter Name: Allowed to be in string format.
    - Item Code: Allowed to be in string format.
    - Item Rank: Represents the rank given by each voter.

    Output Format:
    --------------
    The final output of the algorithm will be in CSV file format with the following columns:
    - Query: The same as the input.
    - Item Code: The same as the input.
    - Item Rank: The rank information (not the score information).
      - Note: The smaller the Item Rank, the higher the rank.
%}

function combanz(input_file_path, output_file_path)
    % combanz - Calls the Python Comb* family algorithm for rank aggregation.
    %
    % This function imports the Python module `combanz` and calls its `combanz` function.
    % It processes the input CSV file containing voting data and outputs the aggregated rankings 
    % to the specified output CSV file.
    %
    % Parameters
    % ----------
    % input_file_path : str
    %     Path to the input CSV file containing voting data.
    % output_file_path : str
    %     Path to the output CSV file where the aggregated rankings will be saved.
    
    combanzmodule = py.importlib.import_module('combanz');
    combanzmodule.combanz(input_file_path, output_file_path);
end
