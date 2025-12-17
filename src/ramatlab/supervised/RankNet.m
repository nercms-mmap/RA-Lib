%{
    RankNet - MATLAB interface for RankNet algorithm for rank aggregation
%}

classdef RankNet
    properties
        pyObj
    end
    
    methods
        function obj = RankNet(hidden_units, learning_rate, epochs_per_query)
            % RankNet constructor - creates Python RankNet instance
            
            if nargin < 3
                epochs_per_query = 5;
            end
            if nargin < 2
                learning_rate = 0.001;
            end
            if nargin < 1
                hidden_units = [10];
            end
            
            % Convert MATLAB types to Python types
            if isnumeric(hidden_units)
                hidden_units = py.list(hidden_units);
            end
            
            % Import Python module and create RankNet instance
            % Follows the same pattern as AggRankDE, CRF, WeightedBorda
            obj.pyObj = py.importlib.import_module('src.rapython.supervised.ranknet').RankNet(hidden_units, learning_rate, epochs_per_query);
        end
        
        function train(obj, train_file_path, train_rel_path)
            % train - Train the RankNet model
            
            obj.pyObj.train(train_file_path, train_rel_path);
        end
        
        function test(obj, test_file_path, test_output_path, using_average_w)
            % test - Test the RankNet model and save results
            
            if nargin < 4
                using_average_w = true;
            end
            
            obj.pyObj.test(test_file_path, test_output_path, using_average_w);
        end
    end
end
