classdef QI_IRA
    methods(Static)
        function qi_ira(input_file_path, output_file_path, input_rel_path, k_set, iteration, error_rate, input_type)
            
            % 调用 Python 的 qi_ira 函数
            py.importlib.import_module('src.rapython.supervised.qi_ira').qi_ira(input_file_path, output_file_path, input_rel_path, py.int(k_set), py.int(iteration), error_rate, input_type);
        end
    end
end
