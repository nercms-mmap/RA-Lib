classdef CRF
    properties
        pyObj  % Python 对象实例
    end
    
    methods
        % 构造函数
        function obj = CRF()
            % 创建 Python 类的实例
            obj.pyObj = py.importlib.import_module('src.rapython.supervised.crf').CRF();
        end
        
        % 调用 Python 类的 train 方法
        function train(obj, train_file_path, train_rel_path, input_type, alpha, epsilon, epoch, loss_cut_off)
            if nargin < 8
                loss_cut_off = py.None;
            end
            if nargin < 7
                epoch = 300;
            end
            if nargin < 6
                epsilon = 5;
            end
            if nargin < 5
                alpha = 0.01;
            end
           % 检查 loss_cut_off 的类型
            if isnumeric(loss_cut_off)
                loss_cut_off = py.int(loss_cut_off);  % 转换为 py.int
            end
            obj.pyObj.train(train_file_path, train_rel_path, input_type, alpha, py.int(epsilon), py.int(epoch), loss_cut_off);
        end
        
        % 调用 Python 类的 test 方法
        function test(obj, test_file_path, test_output_path)
            obj.pyObj.test(test_file_path, test_output_path);
        end
    end
end
