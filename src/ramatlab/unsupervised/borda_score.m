function borda_score(input_file_path, output_file_path)
    bordamodule = py.importlib.import_module('borda_score');
    bordamodule.borda_score(input_file_path, output_file_path);
end
