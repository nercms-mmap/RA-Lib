%{
    AggRankDE - Class implementing the AggRankDE algorithm for rank aggregation.

    This class serves as a MATLAB interface to the Python AggRankDE implementation.
    It allows for training and testing the rank aggregation model using differential evolution.

    Reference:
    ----------
    - Bałchanowski, M., & Boryczka, U. (2022). Aggregation of rankings using metaheuristics in recommendation systems. Electronics, 11(3), 369.

    Authors:
    --------
    Qi Deng

    Date:
    -----
    2023-12-19
%}

classdef AggRankDE
    properties
        pyObj  % Python object instance
    end

    methods
        % Constructor
        function obj = AggRankDE(np_val, max_iteration, cr, f, input_type, n)
            % AggRankDE - Initializes the AggRankDE object with specified parameters.
            %
            % Parameters
            % ----------
            % np_val : int, optional
            %     The size of the population (NP), representing the number of candidate solutions. Default is 50.
            %
            % max_iteration : int, optional
            %     The maximum number of iterations for the optimization process. Default is 100.
            %
            % cr : float, optional
            %     Crossover probability (CR) for the differential evolution algorithm. Default is 0.9.
            %
            % f : float, optional
            %     The amplification factor (F) for the differential evolution algorithm. Default is 0.5.
            %
            % input_type : InputType, optional
            %     Specifies whether the input data is treated as ranks or scores. Default is InputType.RANK.
            %
            % n : int, optional
            %     Represents the parameter for the fitness function. If set to None, assumes N is the number of relevant items. Default is None.

            if nargin < 6
                n = py.None;
            end
            if nargin < 5
                input_type = InputType.RANK;
            end
            if nargin < 4
                f = 0.5;
            end
            if nargin < 3
                cr = 0.9;
            end
            if nargin < 2
                max_iteration = 100;
            end
            if nargin < 1
                np_val = 50;
            end

            if isnumeric(n)
                n = py.int(n);  % Convert to py.int
            end

            % Create an instance of the Python class
            obj.pyObj = py.importlib.import_module('src.rapython.supervised.aggrankde').AggRankDE(py.int(np_val), py.int(max_iteration), cr, f, input_type, n);
        end

        % Call the train method of the Python class
        function train(obj, train_file_path, train_rel_path, input_type)
            % train - Trains the AggRankDE model using the provided training data.
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
            % Returns
            % -------
            % None
            %     The function does not return a value, but it trains the model and stores the
            %     resulting weights in `self.weights` and `self.average_weight`.

            if nargin < 4
                input_type = InputType.RANK;
            end
            obj.pyObj.train(train_file_path, train_rel_path, input_type);
        end

        % Call the test method of the Python class
        function test(obj, test_file_path, test_output_path, using_average_w)
            % test - Tests the AggRankDE model using the provided test data and writes the ranked results to a CSV file.
            %
            % Parameters
            % ----------
            % test_file_path : str
            %     The file path to the test data (e.g., ranking data).
            %
            % test_output_path : str
            %     The file path where the ranked results will be written in CSV format.
            %
            % using_average_w : bool, optional
            %     A flag indicating whether to use the average weights across queries for scoring.
            %     Default is true, which uses average weights.
            %
            % Returns
            % -------
            % None
            %     The function does not return a value but writes the ranked items to the CSV file specified by `test_output_path`.

            if nargin < 4
                using_average_w = true;
            end
            obj.pyObj.test(test_file_path, test_output_path, using_average_w);
        end
    end
end
