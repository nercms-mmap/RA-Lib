classdef InputType
    properties (Constant)
        % 直接访问枚举成员
        RANK = py.importlib.import_module('src.rapython.common.constant').InputType.RANK;
        SCORE = py.importlib.import_module('src.rapython.common.constant').InputType.SCORE;
    end
end
