classdef McType
    properties (Constant)
        % 直接访问枚举成员
        MC1 = py.importlib.import_module('src.rapython.unsupervised.markovchain').McType.MC1;
        MC2 = py.importlib.import_module('src.rapython.unsupervised.markovchain').McType.MC2;
        MC3 = py.importlib.import_module('src.rapython.unsupervised.markovchain').McType.MC3;
        MC4 = py.importlib.import_module('src.rapython.unsupervised.markovchain').McType.MC4;
    end
end
