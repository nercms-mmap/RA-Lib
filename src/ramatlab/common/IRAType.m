classdef IRAType
    properties (Constant)
        % 直接访问枚举成员
        IRA_RANK = py.importlib.import_module('src.rapython.supervised.ira').MethodType.IRA_RANK;
        IRA_SCORE = py.importlib.import_module('src.rapython.supervised.ira').MethodType.IRA_SCORE;
    end
end
