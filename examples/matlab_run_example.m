function matlab_run_example()
    PYTHON_PATH = "C:\Users\2021\.conda\envs\pymatlab\python.exe";
    

    init_python(PYTHON_PATH);
    add_envpath();


    input_file_path = '..\\test\\full_lists\\data\\simulation_test.csv';
    output_base_path = 'results';  % 输出基础路径

    % 检查目录是否存在，如果不存在则创建它
    if ~exist(output_base_path, 'dir')
        mkdir(output_base_path);
    end


    call_unsupervised_methods(input_file_path, output_base_path);
    call_supervised_methods(input_file_path, output_base_path)
    
end

function call_supervised_methods(input_file_path, output_base_path)
end

function call_unsupervised_methods(input_file_path, output_base_path)

    
    % 定义方法名和文件名
    methods = {
        'markovchainmethod', 'borda_score', 'cg', 'combanz', ...
        'combmax', 'combmed', 'combmin', 'combsum', ...
        'dibra', 'dowdall', 'er', 'hpa', ...
        'irank', 'bordacount', 'mean', ...
        'median', 'mork_heuristic', 'postndcg', 'rrf'
    };
    
    % 调用每个方法
    for i = 1:length(methods)
        method_name = methods{i};
        output_file_path = fullfile(output_base_path, [method_name, '.csv']);
        
        % 根据函数名动态调用函数
        switch method_name
            case 'bordacount'
                bordacount(input_file_path, output_file_path);
            case 'borda_score'
                borda_score(input_file_path, output_file_path);
            case 'cg'
                cg(input_file_path, output_file_path);
            case 'combanz'
                combanz(input_file_path, output_file_path);
            case 'combmax'
                combmax(input_file_path, output_file_path);
            case 'combmed'
                combmed(input_file_path, output_file_path);
            case 'combmin'
                combmin(input_file_path, output_file_path);
            case 'combsum'
                combsum(input_file_path, output_file_path);
            case 'dibra'
                dibra(input_file_path, output_file_path, InputType.RANK); 
            case 'dowdall'
                dowdall(input_file_path, output_file_path);
            case 'er'
                er(input_file_path, output_file_path, InputType.RANK); 
            case 'hpa'
                hpa(input_file_path, output_file_path, InputType.RANK); 
            case 'irank'
                irank(input_file_path, output_file_path, InputType.RANK);
            case 'markovchainmethod'
                markovchainmethod(input_file_path, output_file_path, McType.MC1);
            case 'mean'
                mean(input_file_path, output_file_path);
            case 'median'
                median(input_file_path, output_file_path);
            case 'mork_heuristic'
                mork_heuristic_maximum(input_file_path, output_file_path);
            case 'postndcg'
                postndcg(input_file_path, output_file_path, InputType.RANK); 
            case 'rrf'
                rrf(input_file_path, output_file_path);
            otherwise
                fprintf('Function %s not recognized.\n', method_name);
        end
        fprintf('Finished %s()\n', method_name);
    end

    fprintf('All functions processed successfully!\n');
end

function init_python(PYTHON_PATH)

    % initPython - 设置 Python 环境和包路径
    pyenv('Version', PYTHON_PATH);  % 替换为实际的 Python 可执行文件路径


    % 获取src路径
    currentFilePath = mfilename('fullpath');
    [exampleDir, ~, ~] = fileparts(currentFilePath);
    % 项目根目录
    [projectDir, ~, ~] = fileparts(exampleDir);
    
    srcDir = fullfile(projectDir, 'src');
    unsupervisedpath = fullfile(projectDir, 'src', 'rapython', 'unsupervised');
    commonPath = fullfile(projectDir, 'src', 'rapython', 'common');
    % supervisedpath = fullfile(projectDir, 'src', 'supervised');
    % semipath = fullfile(projectDir, 'src', 'semi');
    % 定义路径
    pathsToAdd = {projectDir, srcDir, unsupervisedpath, commonPath};
    
    % 遍历每个路径并检查是否在系统搜索路径中
    for i = 1:length(pathsToAdd)
        path_to_add = pathsToAdd{i};
        
        % 如果路径不在系统搜索路径中，则添加该路径
        if ~any(strcmp(py.sys.path, path_to_add))
            py.sys.path().append(path_to_add);
        end

    end

end
function add_envpath()
    % 获取当前脚本所在的目录路径
    currentDir = fileparts(mfilename('fullpath'));
    % 获取父级目录路径
    projectDir = fileparts(currentDir);
    % 获取 "..\src\ramatlab" 的完整路径
    targetDir = fullfile(projectDir, 'src', 'ramatlab');

    unsupervisedDir = fullfile(targetDir, 'unsupervised');
    supervisedDir = fullfile(targetDir, 'supervised');
    semiDir = fullfile(targetDir, 'semi');
    commonDir = fullfile(targetDir, 'common');
    % 检查并添加路径
    if ~any(strcmp(path, unsupervisedDir))
        addpath(unsupervisedDir);
    end
    
    if ~any(strcmp(path, supervisedDir))
        addpath(supervisedDir);
    end
    
    if ~any(strcmp(path, semiDir))
        addpath(semiDir);
    end

    if ~any(strcmp(path, commonDir))
        addpath(commonDir);
    end

end
