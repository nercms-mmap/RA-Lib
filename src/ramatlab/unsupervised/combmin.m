%{
    combmin - Aggregate rankings using the Comb* family of algorithms and output results to a CSV file.
    
    This function imports the Python module `combmin` and calls its `combmin` function to perform
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
%}

function combmin(input_file_path, output_file_path)
    % combmin - Calls the Python Comb* family algorithm for rank aggregation.
    %
    % This function imports the Python module `combmin` and calls its `combmin` function.
    % It processes the input CSV file containing voting data and outputs the aggregated rankings 
    % to the specified output CSV file.
    %
    % Parameters
    % ----------
    % input_file_path : str
    %     Path to the input CSV file containing voting data.
    % output_file_path : str
    %     Path to the output CSV file where the aggregated rankings will be written.
    
    combminmodule = py.importlib.import_module('combmin');
    combminmodule.combmin(input_file_path, output_file_path);
end
