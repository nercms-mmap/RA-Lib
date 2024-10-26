%{
    CRF - Class implementing the CRF (Conditional Random Fields) algorithm for supervised preference aggregation.

    This class provides a MATLAB interface to the Python CRF implementation, enabling the training
    and testing of the model for ranking tasks.

    Reference:
    ----------
    - Volkovs, M. N., & Zemel, R. S. (2013, October). CRF framework for supervised preference aggregation.
      In Proceedings of the 22nd ACM international conference on Information & Knowledge Management (pp. 89-98).

    Authors:
    --------
    Qi Deng

    Date:
    -----
    2023-12-24

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
    1. The data input accepts partial lists.
    2. Smaller Item Rank values indicate higher rankings.
%}

classdef CRF
    properties
        pyObj  % Python object instance
    end

    methods
        % Constructor
        function obj = CRF()
            % CRF - Initializes the CRF object by creating an instance of the Python class.
            %
            % Returns
            % -------
            % obj : CRF
            %     An instance of the CRF class with the Python object initialized.

            % Create an instance of the Python class
            obj.pyObj = py.importlib.import_module('src.rapython.supervised.crf').CRF();
        end

        % Call the train method of the Python class
        function train(obj, train_file_path, train_rel_path, input_type, alpha, epsilon, epoch, loss_cut_off)
            % train - Trains the CRF model using the provided training data.
            %
            % Parameters
            % ----------
            % train_file_path : str
            %     The file path to the training base data (e.g., ranking data).
            %
            % train_rel_path : str
            %     The file path to the relevance data (e.g., ground truth relevance scores).
            %
            % input_type : InputType
            %     Specifies the format or type of the input data. InputType.RANK is recommended.
            %
            % alpha : float, optional
            %     The learning rate for weight updates. Default is 0.01.
            %
            % epsilon : int, optional
            %     The cut-off threshold for sampling items. Default is 5.
            %
            % epoch : int, optional
            %     The number of iterations for training. Default is 300.
            %
            % loss_cut_off : int or None, optional
            %     The cut-off for the loss computation (k in ndcg@k). If None, will use the number of relevant documents.
            %
            % Returns
            % -------
            % None
            %     This function updates the internal state of the model (i.e., the weights) but does not return a value.

            if nargin < 8
                loss_cut_off = py.None;  % Set to None if not provided
            end
            if nargin < 7
                epoch = 300;  % Default epoch
            end
            if nargin < 6
                epsilon = 5;  % Default epsilon
            end
            if nargin < 5
                alpha = 0.01;  % Default alpha
            end

            % Check the type of loss_cut_off
            if isnumeric(loss_cut_off)
                loss_cut_off = py.int(loss_cut_off);  % Convert to py.int
            end

            % Call the Python train method
            obj.pyObj.train(train_file_path, train_rel_path, input_type, alpha, py.int(epsilon), py.int(epoch), loss_cut_off);
        end

        % Call the test method of the Python class
        function test(obj, test_file_path, test_output_path)
            % test - Tests the CRF model using the provided test data and writes the ranked results to a CSV file.
            %
            % Parameters
            % ----------
            % test_file_path : str
            %     The file path to the test data (e.g., ranking data).
            %
            % test_output_path : str
            %     The file path where the ranked results will be written in CSV format.
            %
            % Returns
            % -------
            % None
            %     The function does not return a value but writes the ranked results to the specified CSV file.

            obj.pyObj.test(test_file_path, test_output_path);
        end
    end
end
