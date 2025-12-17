%{
    ListNet - MATLAB interface for ListNet algorithm for rank aggregation
%}

classdef ListNet
    properties
        pyObj
    end
    
    methods
        function obj = ListNet(hidden_units, learning_rate, epochs_per_query, eps)
            % ListNet constructor - creates Python ListNet instance
            
            if nargin < 4
                eps = 1e-10;
            end
            if nargin < 3
                epochs_per_query = 10;
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
            
            % Import Python module and create ListNet instance
            % Follows the same pattern as AggRankDE, CRF, WeightedBorda
            obj.pyObj = py.importlib.import_module('src.rapython.supervised.listnet').ListNet(hidden_units, learning_rate, epochs_per_query, eps);
        end
        
        function train(obj, train_file_path, train_rel_path)
            % train - Train the ListNet model
            
            obj.pyObj.train(train_file_path, train_rel_path);
        end
        
        function test(obj, test_file_path, test_output_path, using_average_w)
            % test - Test the ListNet model and save results
            
            if nargin < 4
                using_average_w = true;
            end
            
            obj.pyObj.test(test_file_path, test_output_path, using_average_w);
        end
    end
end
