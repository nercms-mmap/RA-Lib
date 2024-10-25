function markovchainmethod(input_file_path, output_file_path, mc_type, max_iteration)
    if nargin < 4
        max_iteration = int64(50);
    end
    disp(['Input file path: ', input_file_path]);
    disp(['Output file path: ', output_file_path]);
    disp(['MC type: ', char(mc_type)]);  % 确保 mc_type 为有效值
    disp(['Max iteration: ', num2str(max_iteration)]);  % 打印 max_iteration

    mcmodule = py.importlib.import_module('markovchain');
    mcmodule.markovchainmethod(input_file_path, output_file_path, mc_type, max_iteration);
end