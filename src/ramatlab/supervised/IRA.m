classdef IRA
    methods(Static)
        function ira(input_file_path, output_file_path, input_rel_path, k_set, iteration, error_rate, mode, input_type)
            
            % 调用 Python 的 ira 函数
            py.importlib.import_module('src.rapython.supervised.ira').ira(input_file_path, output_file_path, input_rel_path, py.int(k_set), py.int(iteration), error_rate, mode, input_type);
        end
    end
end
