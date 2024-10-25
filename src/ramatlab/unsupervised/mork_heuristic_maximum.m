function mork_heuristic_maximum(input_file_path, output_file_path)
    morkmodule = py.importlib.import_module('mork_heuristic_maximum');
    morkmodule.mork_heuristic(input_file_path, output_file_path);
end