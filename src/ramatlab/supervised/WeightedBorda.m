%{
    WeightedBorda - Class implementing the Weighted Borda algorithm for ranking aggregation.

    This class provides a MATLAB interface to the Python WeightedBorda implementation, allowing
    users to train and test the model using specified datasets.

    Reference:
    ----------
    - Subbian, K., & Melville, P. (2011, October). Supervised rank aggregation for predicting influencers in Twitter. In 2011 IEEE Third International Conference on Privacy, Security, Risk and Trust and 2011 IEEE Third International Conference on Social Computing (pp. 661-665). IEEE.

    Author:
    -------
    Qi Deng

    Date:
    -----
    2023-12-26

    Input Format for Training Data:
    -------------------------------
    1. File 1: train_rel_data
       - Format: CSV
       - Columns: Query | 0 | Item | Relevance

    2. File 2: train_base_data
       - Format: CSV
       - Columns: Query | Voter Name | Item Code | Item Rank

    Notes:
    ------
    - Query does not need to be consecutive integers starting from 1.
    - Voter Name and Item Code are allowed to be in string format.

    Output Format:
    --------------
    - The final output of the algorithm will be in CSV format with the following columns:
      - Query | Item Code | Item Rank
      - Note: The output contains ranking information, not score information.

    Input Format for Testing Data:
    -------------------------------
    1. File 1: test_data
       - Format: CSV
       - Columns: Query | Voter Name | Item Code | Item Rank

    Notes:
    ------
    - Query does not need to be consecutive integers starting from 1.
    - Voter Name and Item Code are allowed to be in string format.

    Additional Details:
    -------------------
    1. The data input accepts full lists; partial lists will be treated as having the lowest rank.
    2. Smaller Item Rank values indicate higher rankings.
    3. The voters in the training and testing datasets are the same.
%}

classdef WeightedBorda
    properties
        % Property to hold an instance of the Python WeightedBorda class
        pyObj
    end

    methods
        % Constructor to create a Python object of WeightedBorda
        function obj = WeightedBorda(topk, is_partial_list)
            % Create an instance of the Python WeightedBorda class.
            %
            % Parameters:
            %   topk : int or None, optional
            %       The number of top items to consider. Defaults to None.
            %   is_partial_list : bool, optional
            %       A flag indicating if partial lists are used. Defaults to true.
            %
            % Returns:
            %   obj : WeightedBorda
            %       An instance of the WeightedBorda class.

            if nargin < 2
                is_partial_list = true;  % Default parameter
            end
            if nargin < 1
                topk = py.None;  % Use None if topk is not provided
            end
            % Create an instance of the Python class
            obj.pyObj = py.importlib.import_module('src.rapython.supervised.weighted_borda').WeightedBorda(topk, is_partial_list);
        end

        % Call the Python train method
        function train(obj, train_file_path, train_rel_path)
            % Train the model using the provided training data.
            %
            % Parameters:
            %   train_file_path : str
            %       The file path to the training base data (CSV).
            %   train_rel_path : str
            %       The file path to the relevance data (CSV).
            %
            % Returns:
            %   None
            %       The method updates the internal state of the class with
            %       calculated weights for the voters.

            % Ensure the input file paths are in string format
            train_file_path = convertCharsToStrings(train_file_path);
            train_rel_path = convertCharsToStrings(train_rel_path);

            % Call the train method of the Python object
            obj.pyObj.train(train_file_path, train_rel_path);
        end

        % Call the Python test method
        function test(obj, test_file_path, test_output_path, using_average_w)
            % Test the model on the provided test data and write results to the specified output location.
            %
            % Parameters:
            %   test_file_path : str
            %       The file path to the test data (CSV).
            %   test_output_path : str
            %       The file path where the output CSV file will be saved.
            %   using_average_w : bool, optional
            %       A flag indicating whether to use average weights for scoring. Defaults to true.
            %
            % Returns:
            %   None
            %       The method writes the ranking results to the specified output location.

            if nargin < 4
                using_average_w = true;  % Default parameter
            end
            % Ensure the input file paths are in string format
            test_file_path = convertCharsToStrings(test_file_path);
            test_output_path = convertCharsToStrings(test_output_path);

            % Call the test method of the Python object
            obj.pyObj.test(test_file_path, test_output_path, using_average_w);
        end
    end
end
