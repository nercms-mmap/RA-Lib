function markovchainmethod(input_file_path, output_file_path, mc_type, max_iteration)
    if nargin < 4
        max_iteration = 50;
    end
    mcmodule = py.importlib.import_module('markovchain');
    mcmodule.markovchainmethod(input_file_path, output_file_path, mc_type, py.int(max_iteration));
end