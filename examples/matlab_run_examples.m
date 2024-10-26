function matlab_run_examples()
    % Set the path to the Python executable.
    PYTHON_PATH = "C:\\Users\\2021\\.conda\\envs\\pymatlab\\python.exe";

    % Initialize the Python environment.
    init_python(PYTHON_PATH);
    add_envpath();  % Add necessary environment paths.

    % Define the input and output file paths.
    input_file_path = '..\\test\\full_lists\\data\\simulation_test.csv';
    output_base_path = 'results';  % Base path for output files.

    % Check if the output directory exists; if not, create it.
    if ~exist(output_base_path, 'dir')
        mkdir(output_base_path);  % Create the directory if it doesn't exist.
    end

    % Call unsupervised methods with the input and output paths.
    call_unsupervised_methods(input_file_path, output_base_path);

    % Define training file paths.
    train_file_path = '..\\test\\full_lists\\data\\simulation_train.csv';
    train_rel_path = '..\\test\\full_lists\\data\\simulation_train_rel.csv';

    % Call supervised methods with the training and test file paths.
    call_supervised_methods(train_file_path, train_rel_path, input_file_path, output_base_path);

    % Call semi-supervised methods with the training and test file paths.
    call_semi_methods(train_file_path, train_rel_path, input_file_path, output_base_path);

    % Define the relative path for input data.
    input_rel_path = '..\\test\\full_lists\\data\\simulation_test_rel.csv';

    % Call IRA methods with the input and output paths.
    call_ira_methods(input_file_path, output_base_path, input_rel_path);
end

function call_semi_methods(train_file_path, train_rel_path, test_file_path, output_base_path)
    % Initialize the SSRA method and train it using the training data.
    fprintf('Initializing SSRA method...\n');
    ssra = SSRA();
    fprintf('Training SSRA with training data: %s and %s...\n', train_file_path, train_rel_path);
    ssra.train(train_file_path, train_rel_path, InputType.RANK);
    fprintf('SSRA training completed.\n');

    % Define the output path for the SSRA results.
    test_output_path = fullfile(output_base_path, 'ssra.csv');
    fprintf('Testing SSRA method and saving results to: %s...\n', test_output_path);
    ssra.test(test_file_path, test_output_path);  % Test the SSRA method.
    fprintf('SSRA testing completed.\n');
end

function call_ira_methods(input_file_path, output_base_path, input_rel_path)
    fprintf('Calling QI_IRA method...\n');
    % Call QI_IRA method and save the output.
    output_path = fullfile(output_base_path, 'qi_ira.csv');
    fprintf('Running QI_IRA with input file: %s, output path: %s, and rel path: %s...\n', input_file_path, output_path, input_rel_path);
    QI_IRA.qi_ira(input_file_path, output_path, input_rel_path, 3, 2, 0.02, InputType.RANK);
    fprintf('QI_IRA completed. Results saved to: %s\n', output_path);

    fprintf('Calling IRA method with rank input type...\n');
    % Call IRA method with rank input type and save the output.
    output_path = fullfile(output_base_path, 'ira_r.csv');
    fprintf('Running IRA (Rank) with input file: %s, output path: %s, and rel path: %s...\n', input_file_path, output_path, input_rel_path);
    IRA.ira(input_file_path, output_path, input_rel_path, 3, 2, 0.02, IRAType.IRA_RANK, InputType.RANK);
    fprintf('IRA (Rank) completed. Results saved to: %s\n', output_path);

    fprintf('Calling IRA method with score input type...\n');
    % Call IRA method with score input type and save the output.
    output_path = fullfile(output_base_path, 'ira_s.csv');
    fprintf('Running IRA (Score) with input file: %s, output path: %s, and rel path: %s...\n', input_file_path, output_path, input_rel_path);
    IRA.ira(input_file_path, output_path, input_rel_path, 3, 2, 0.02, IRAType.IRA_SCORE, InputType.RANK);
    fprintf('IRA (Score) completed. Results saved to: %s\n', output_path);
end

function call_supervised_methods(train_file_path, train_rel_path, test_file_path, output_base_path)
    fprintf('Initializing AggRankDE method...\n');
    % Initialize AggRankDE method and train it using the training data.
    aggRankDE = AggRankDE();
    fprintf('Training AggRankDE with training data: %s and %s...\n', train_file_path, train_rel_path);
    aggRankDE.train(train_file_path, train_rel_path, InputType.RANK);
    fprintf('AggRankDE training completed.\n');

    % Define the output path for AggRankDE results.
    test_output_path = fullfile(output_base_path, 'aggrankde.csv');
    fprintf('Testing AggRankDE method and saving results to: %s...\n', test_output_path);
    aggRankDE.test(test_file_path, test_output_path);  % Test the AggRankDE method.
    fprintf('AggRankDE testing completed.\n');

    fprintf('Initializing CRF method...\n');
    % Initialize CRF method and train it using the training data.
    crf = CRF();
    fprintf('Training CRF with training data: %s and %s...\n', train_file_path, train_rel_path);
    crf.train(train_file_path, train_rel_path, InputType.RANK, 0.01, 5, 2);
    fprintf('CRF training completed.\n');

    % Define the output path for CRF results.
    test_output_path = fullfile(output_base_path, 'crf.csv');
    fprintf('Testing CRF method and saving results to: %s...\n', test_output_path);
    crf.test(test_file_path, test_output_path);  % Test the CRF method.
    fprintf('CRF testing completed.\n');

    fprintf('Initializing Weighted Borda method...\n');
    % Initialize Weighted Borda method and train it using the training data.
    weightedBorda = WeightedBorda();
    fprintf('Training Weighted Borda with training data: %s and %s...\n', train_file_path, train_rel_path);
    weightedBorda.train(train_file_path, train_rel_path);
    fprintf('Weighted Borda training completed.\n');

    % Define the output path for Weighted Borda results.
    test_output_path = fullfile(output_base_path, 'weighted_borda.csv');
    fprintf('Testing Weighted Borda method and saving results to: %s...\n', test_output_path);
    weightedBorda.test(test_file_path, test_output_path);  % Test the Weighted Borda method.
    fprintf('Weighted Borda testing completed.\n');
end


function call_unsupervised_methods(input_file_path, output_base_path)
    % Define the method names and corresponding output filenames.
    methods = {
        'markovchainmethod', 'borda_score', 'cg', 'combanz', ...
        'combmax', 'combmed', 'combmin', 'combsum', ...
        'dibra', 'dowdall', 'er', 'hpa', ...
        'irank', 'bordacount', 'mean', ...
        'median', 'mork_heuristic', 'postndcg', 'rrf'
    };

    % Call each method and save the output results.
    for i = 1:length(methods)
        method_name = methods{i};
        output_file_path = fullfile(output_base_path, [method_name, '.csv']);

        % Dynamically call the method based on its name.
        switch method_name
            case 'bordacount'
                bordacount(input_file_path, output_file_path);  % Call bordacount method.
            case 'borda_score'
                borda_score(input_file_path, output_file_path);  % Call borda_score method.
            case 'cg'
                cg(input_file_path, output_file_path);  % Call cg method.
            case 'combanz'
                combanz(input_file_path, output_file_path);  % Call combanz method.
            case 'combmax'
                combmax(input_file_path, output_file_path);  % Call combmax method.
            case 'combmed'
                combmed(input_file_path, output_file_path);  % Call combmed method.
            case 'combmin'
                combmin(input_file_path, output_file_path);  % Call combmin method.
            case 'combsum'
                combsum(input_file_path, output_file_path);  % Call combsum method.
            case 'dibra'
                dibra(input_file_path, output_file_path, InputType.RANK);  % Call dibra method.
            case 'dowdall'
                dowdall(input_file_path, output_file_path);  % Call dowdall method.
            case 'er'
                er(input_file_path, output_file_path, InputType.RANK);  % Call er method.
            case 'hpa'
                hpa(input_file_path, output_file_path, InputType.RANK);  % Call hpa method.
            case 'irank'
                irank(input_file_path, output_file_path, InputType.RANK);  % Call irank method.
            case 'markovchainmethod'
                markovchainmethod(input_file_path, output_file_path, McType.MC1);  % Call markovchainmethod.
            case 'mean'
                mean(input_file_path, output_file_path);  % Call mean method.
            case 'median'
                median(input_file_path, output_file_path);  % Call median method.
            case 'mork_heuristic'
                mork_heuristic_maximum(input_file_path, output_file_path);  % Call mork_heuristic method.
            case 'postndcg'
                postndcg(input_file_path, output_file_path, InputType.RANK);  % Call postndcg method.
            case 'rrf'
                rrf(input_file_path, output_file_path);  % Call rrf method.
            otherwise
                fprintf('Function %s not recognized.\n', method_name);  % Print warning for unrecognized function.
        end
        fprintf('Finished %s()\n', method_name);  % Indicate method completion.
    end

    fprintf('All functions processed successfully!\n');  % Indicate all functions have been processed.
end

function init_python(PYTHON_PATH)
    % init_python - Set up the Python environment and package paths.
    pyenv('Version', PYTHON_PATH);  % Set the Python executable path.

    % Get the current file path.
    currentFilePath = mfilename('fullpath');
    [exampleDir, ~, ~] = fileparts(currentFilePath);

    % Get the project root directory.
    [projectDir, ~, ~] = fileparts(exampleDir);

    % Define necessary paths for the project.
    srcDir = fullfile(projectDir, 'src');
    unsupervisedpath = fullfile(projectDir, 'src', 'rapython', 'unsupervised');
    commonPath = fullfile(projectDir, 'src', 'rapython', 'common');

    % Define paths to add to the system path.
    pathsToAdd = {projectDir, srcDir, unsupervisedpath, commonPath};

    % Iterate through each path and check if it is in the system path.
    for i = 1:length(pathsToAdd)
        path_to_add = pathsToAdd{i};

        % If the path is not in the system path, add it.
        if ~any(strcmp(py.sys.path, path_to_add))
            py.sys.path().append(path_to_add);
        end
    end

    % Install required Python packages.
    install_requirements(srcDir);  % Call the function to install dependencies.
end

function install_requirements(srcDir)
    % install_requirements - Install project dependencies using pip.
    try
        % Install dependencies listed in requirements.txt.
        
        % Define the full path for requirements.txt.
        requirementsFile = fullfile(srcDir, 'requirements.txt');
        
        % Install dependencies listed in requirements.txt.
        system(['pip install -r "', requirementsFile, '"']);  
        disp('All dependencies installed successfully.');
    catch ME
        warning(ME.identifier, 'Error occurred while installing dependencies: %s', ME.message);  % Print warning for errors.
    end
end

function add_envpath()
    % Get the current script directory path.
    currentDir = fileparts(mfilename('fullpath'));

    % Get the parent directory path.
    projectDir = fileparts(currentDir);

    % Get the full path for "..\src\ramatlab".
    targetDir = fullfile(projectDir, 'src', 'ramatlab');

    % Define paths for unsupervised, supervised, and semi directories.
    unsupervisedDir = fullfile(targetDir, 'unsupervised');
    supervisedDir = fullfile(targetDir, 'supervised');
    semiDir = fullfile(targetDir, 'semi');
    commonDir = fullfile(targetDir, 'common');

    % Check and add paths if they are not already present.
    if ~any(strcmp(path, unsupervisedDir))
        addpath(unsupervisedDir);  % Add unsupervised path.
    end

    if ~any(strcmp(path, supervisedDir))
        addpath(supervisedDir);  % Add supervised path.
    end

    if ~any(strcmp(path, semiDir))
        addpath(semiDir);  % Add semi path.
    end

    if ~any(strcmp(path, commonDir))
        addpath(commonDir);  % Add common path.
    end
end
