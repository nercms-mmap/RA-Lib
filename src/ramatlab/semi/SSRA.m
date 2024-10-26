%{
    SSRA - Class implementing the Semi-supervised Ranking Aggregation (SSRA) algorithm.

    This class provides a MATLAB interface to the Python SSRA implementation, allowing
    users to train and test the model using specified datasets.

    Reference:
    ----------
    - Chen, S., Wang, F., Song, Y., & Zhang, C. (2008, October). Semi-supervised ranking aggregation.
      In Proceedings of the 17th ACM conference on Information and knowledge management (pp. 1427-1428).

    Author:
    -------
    Qi Deng

    Date:
    -----
    2024-01-18

    Training Data Input Format:
    ---------------------------
    1. train_rel_data (Relevance Data):
       - Format: CSV
       - Columns: Query | 0 | Item | Relevance

    2. train_base_data (Base Ranking Data):
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

    Test Data Input Format:
    -----------------------
    1. test_data:
       - Format: CSV
       - Columns: Query | Voter Name | Item Code | Item Rank

    Notes:
    ------
    - Query does not need to be consecutive integers starting from 1.
    - Voter Name and Item Code are allowed to be in string format.

    Additional Details:
    -------------------
    1. Input data accepts full lists; partial lists will be treated as having the lowest rank.
    2. Smaller Item Rank values indicate higher rankings.
%}

classdef SSRA
    % SSRA: A class for the Semi-Supervised Ranking Aggregation Algorithm.
    %
    % This class provides methods to train and test the semi-supervised ranking
    % aggregation model using Python's implementation.
    %
    % Properties:
    %   pyObj : Python object instance representing the SSRA class in Python.
    %
    % Methods:
    %   train: Trains the model using the specified training data.
    %   test: Tests the model using the specified test data and outputs results.

    properties
        pyObj  % Instance of the Python object
    end

    methods
        % Constructor
        function obj = SSRA()
            % Creates an instance of the Python SSRA class.
            obj.pyObj = py.importlib.import_module('src.rapython.semi.ssra').SSRA();
        end

        % Calls the train method of the Python class
        function obj = train(obj, train_file_path, train_rel_path, input_type, alpha, beta, constraints_rate, is_partial_list)
            % Trains the model using the provided training data.
            %
            % Parameters:
            %   train_file_path : string
            %       - The file path to the training base data (ranking data).
            %   train_rel_path : string
            %       - The file path to the relevance data (ground truth relevance scores).
            %   input_type : enum
            %       - Specifies the format or type of the input data.
            %   alpha : float, optional
            %       - A hyperparameter controlling the influence of the quadratic form
            %         in the objective function (default is 0.03).
            %   beta : float, optional
            %       - A hyperparameter regulating the L2 norm of the weight vector
            %         in the objective function (default is 0.1).
            %   constraints_rate : float, optional
            %       - Proportion of supervisory information used (default is 0.3).
            %   is_partial_list : bool, optional
            %       - Indicates whether the training data contains a partial list of items
            %         (default is true).
            %
            % Returns:
            %   obj : SSRA
            %       - The modified SSRA object with updated model weights.

            if nargin < 8
                is_partial_list = true; % Default value
            end
            if nargin < 7
                constraints_rate = 0.3; % Default value
            end
            if nargin < 6
                beta = 0.1; % Default value
            end
            if nargin < 5
                alpha = 0.03; % Default value
            end

            % Calls the train method of the Python class
            obj.pyObj.train(train_file_path, train_rel_path, input_type, alpha, beta, constraints_rate, is_partial_list);
        end

        % Calls the test method of the Python class
        function test(obj, test_file_path, test_output_path, using_average_w)
            % Tests the model using the provided test data and saves the results to a specified location.
            %
            % Parameters:
            %   test_file_path : string
            %       - The file path to the test data (ranking data).
            %   test_output_path : string
            %       - The file path where the output CSV file will be saved.
            %   using_average_w : bool, optional
            %       - Indicates whether to use average weights for scoring
            %         (default is true).
            %
            % Returns:
            %   None
            %       - The method writes the ranked results to a CSV file at the specified location.

            if nargin < 4
                using_average_w = true; % Default value
            end

            % Calls the test method of the Python class
            obj.pyObj.test(test_file_path, test_output_path, using_average_w);
        end
    end
end
